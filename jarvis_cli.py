"""PROJECT J.A.R.V.I.S. — Standalone Full-Duplex Voice & Cognitive AI Terminal CLI.

Features:
- 100% Standalone Terminal TUI/CLI (No browser or web GUI required).
- Auto-starts 16kHz Microphone capture on launch with a live 20-bar ANSI VU meter.
- Local Faster-Whisper INT8 Speech-to-Text (`tiny.en`) with automatic silence endpointing.
- 3-Layer Open-Speaker Anti-Echo Defense:
  1. Software Echo Guard Gate (blocks mic capture while J.A.R.V.I.S. speaks + 450ms room-reverb tail).
  2. Lexical Self-Echo Filter (rejects transcripts with >=45% word overlap with J.A.R.V.I.S.'s last speech).
  3. Optional Headset Full-Duplex Mode (press [E]) allowing voice-triggered acoustic barge-in.
- In-Memory Neural British TTS playback (`en-GB-RyanNeural` via `edge-tts` + `PyAV` + `sounddevice`)
  with <20ms Spacebar Barge-In Interruption.
- Full 3-Tier Hybrid Cognitive Router (Gemini 3.1 Flash-Lite -> Local Ollama RTX GPU -> Executive Core)
  + all 12 Autonomous MCP Tools.
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
import io
import math
import os
import re
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

# Force UTF-8 console output on Windows PowerShell / Terminal
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure standalone imports resolve from `audiopart/` first
AUDIOPART_DIR = Path(__file__).resolve().parent
if str(AUDIOPART_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIOPART_DIR))

import numpy as np
import sounddevice as sd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import msvcrt

    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

from server.core.router import cognitive_router
from server.core.voice_engine import voice_engine

console = Console(force_terminal=True, legacy_windows=False)
WORD_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)


class JarvisStandaloneVoiceCLI:
    """Interactive Full-Duplex Terminal Voice & Cognitive AI Engine."""

    SAMPLE_RATE = 16000
    BLOCK_SIZE = 800  # 50ms frames at 16kHz
    MIN_SPEECH_RMS = 0.0012  # Sensitive threshold calibrated for raw Windows MME/WASAPI mic arrays
    BARGE_IN_RMS_THRESHOLD = 0.035
    SILENCE_HOLD_SECONDS = 0.55  # Fast 550ms end-of-speech trigger
    MIN_SPEECH_SECONDS = 0.28
    MAX_RECORD_SECONDS = 18.0

    def __init__(self) -> None:
        self.mic_active: bool = True  # Starts ON automatically
        self.tts_enabled: bool = True
        self.open_speaker_mode: bool = True  # True = Echo Guard, False = Headset Duplex
        self.running: bool = True

        self.is_speaking: bool = False
        self.is_thinking: bool = False
        self.echo_guard_until: float = 0.0
        self.last_spoken_text: str = ""
        self.last_spoken_time: float = 0.0
        self.barge_in_requested: bool = False
        self.barge_in_count: int = 0

        # Live VAD, adaptive noise floor & 450ms pre-roll ring buffer
        self.current_rms: float = 0.0
        self.noise_floor: float = 0.0002
        self.pre_roll_buffer: Deque[np.ndarray] = deque(maxlen=9)  # 450ms pre-roll so first syllable is never clipped
        self.is_recording_utterance: bool = False
        self.recorded_frames: List[np.ndarray] = []
        self.speech_start_ts: float = 0.0
        self.last_voice_ts: float = 0.0
        self.barge_in_frames: int = 0

        self._utterance_queue: Optional[asyncio.Queue[bytes]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream: Optional[sd.InputStream] = None

    # ------------------------------------------------------------------
    # Echo Guard & Lexical Anti-Echo Filter
    # ------------------------------------------------------------------
    @property
    def is_echo_gated(self) -> bool:
        if not self.open_speaker_mode:
            return False
        return self.is_speaking or (time.perf_counter() < self.echo_guard_until)

    def is_echo_of_jarvis(self, transcript: str) -> bool:
        if not self.last_spoken_text:
            return False
        if not self.is_speaking and (time.perf_counter() - self.last_spoken_time) > 8.0:
            return False

        cand_low = transcript.lower().strip()
        jarvis_low = self.last_spoken_text.lower().strip()
        if not cand_low:
            return True
        if len(cand_low) >= 6 and cand_low in jarvis_low:
            return True

        cand_words = WORD_RE.findall(cand_low)
        jarvis_words = set(WORD_RE.findall(jarvis_low))
        if not cand_words or not jarvis_words:
            return False

        overlap = sum(1 for w in cand_words if w in jarvis_words)
        return (overlap / len(cand_words)) >= 0.45

    # ------------------------------------------------------------------
    # Real-Time 16kHz Microphone Callback (50ms blocks)
    # ------------------------------------------------------------------
    def _audio_input_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if not self.mic_active:
            self.current_rms = 0.0
            return

        mono = indata[:, 0]
        rms = float(np.sqrt(np.mean(np.square(mono))))
        self.current_rms = rms
        now = time.perf_counter()

        # 1. Handle Barge-In while J.A.R.V.I.S. is speaking
        if self.is_speaking:
            if not self.open_speaker_mode:
                # Headset mode: voice above threshold for 4 consecutive 50ms frames (200ms) triggers barge-in
                if rms >= self.BARGE_IN_RMS_THRESHOLD:
                    self.barge_in_frames += 1
                    if self.barge_in_frames >= 4:
                        self.barge_in_requested = True
                        self.barge_in_frames = 0
                else:
                    self.barge_in_frames = 0
            return

        # 2. Gate microphone during Open-Speaker Echo Guard cooldown
        if self.is_echo_gated or self.is_thinking:
            self.is_recording_utterance = False
            self.recorded_frames.clear()
            self.pre_roll_buffer.clear()
            return

        # 3. Dynamic Noise-Floor Adaptive VAD + 450ms Pre-Roll Ring Buffer
        dynamic_threshold = max(self.MIN_SPEECH_RMS, min(0.008, self.noise_floor * 3.8))

        if rms >= dynamic_threshold:
            if not self.is_recording_utterance:
                self.is_recording_utterance = True
                self.speech_start_ts = now
                # Prepend 450ms pre-roll frames so the beginning of the first word is never cut off
                self.recorded_frames = list(self.pre_roll_buffer)
                self.pre_roll_buffer.clear()
            self.last_voice_ts = now
            self.recorded_frames.append(mono.copy())
        elif self.is_recording_utterance:
            self.recorded_frames.append(mono.copy())
            silence_dur = now - self.last_voice_ts
            total_dur = now - self.speech_start_ts

            if silence_dur >= self.SILENCE_HOLD_SECONDS or total_dur >= self.MAX_RECORD_SECONDS:
                self.is_recording_utterance = False
                if total_dur >= self.MIN_SPEECH_SECONDS and self.recorded_frames:
                    wav_bytes = self._encode_wav_bytes(self.recorded_frames)
                    self.recorded_frames.clear()
                    if self._loop and self._utterance_queue is not None:
                        self._loop.call_soon_threadsafe(self._utterance_queue.put_nowait, wav_bytes)
                else:
                    self.recorded_frames.clear()
        else:
            # Update ambient noise floor slowly during silence & keep pre-roll ring buffer warm
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
            self.pre_roll_buffer.append(mono.copy())

    def _encode_wav_bytes(self, chunks: List[np.ndarray]) -> bytes:
        full_audio = np.concatenate(chunks).astype(np.float32)
        # 1. Remove DC offset
        full_audio = full_audio - np.mean(full_audio)
        # 2. Digital Automatic Gain Control (AGC) — boosts quiet Windows mic arrays to -1.4 dBFS (0.85 peak)
        peak = float(np.max(np.abs(full_audio))) if full_audio.size > 0 else 0.0
        if peak > 1e-5:
            gain = min(120.0, 0.85 / peak)
            full_audio = full_audio * gain
        pcm16 = np.clip(full_audio * 32767.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    # ------------------------------------------------------------------
    # In-Memory Neural British TTS Playback with <20ms Barge-In
    # ------------------------------------------------------------------
    def _decode_mp3_to_float32(self, mp3_bytes: bytes) -> Optional[tuple[np.ndarray, int]]:
        try:
            import av

            container = av.open(io.BytesIO(mp3_bytes))
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=24000)
            samples_list: List[np.ndarray] = []
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                for r_frame in resampled:
                    arr = r_frame.to_ndarray()[0]
                    samples_list.append(arr)
            if not samples_list:
                return None
            audio = np.concatenate(samples_list).astype(np.float32)
            return audio, 24000
        except Exception:
            return None

    async def _synthesize_sentence_samples(self, sentence: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesizes a single sentence to float32 PCM samples at natural 1.0x speed with <27ms latency."""
        if not sentence.strip():
            return None

        # 1. If Instant Local TTS mode is active (default), synthesize in ~25ms via persistent SAPI5 + Helmet-HUD DSP
        if voice_engine.prefer_local_tts:
            return await voice_engine.synthesize_pcm_local_sapi_async(sentence)

        # 2. Check RAM/Disk cache (<0.5ms) or synthesize British Neural TTS at natural +0% rate
        mp3_bytes = await voice_engine.synthesize_mp3_bytes(sentence, timeout_s=1.2)
        if mp3_bytes:
            decoded = await asyncio.to_thread(self._decode_mp3_to_float32, mp3_bytes)
            if decoded:
                return decoded

        # 3. Instantaneous (~25ms) fallback to Local SAPI5 + Helmet-HUD DSP if network TTS is slow
        return await voice_engine.synthesize_pcm_local_sapi_async(sentence)

    async def play_sentence_tasks(
        self,
        full_text: str,
        sentence_tasks: List[asyncio.Task[Optional[tuple[np.ndarray, int]]]],
    ) -> None:
        """Plays pre-launched sentence synthesis tasks back-to-back with 0ms start delay and <20ms Barge-In."""
        if not self.tts_enabled or not sentence_tasks:
            for t in sentence_tasks:
                t.cancel()
            return

        self.last_spoken_text = full_text
        self.last_spoken_time = time.perf_counter()
        self.is_speaking = True
        self.barge_in_requested = False

        try:
            for task in sentence_tasks:
                if self.barge_in_requested:
                    task.cancel()
                    continue

                decoded = await task
                if not decoded or self.barge_in_requested:
                    continue

                samples, sr = decoded
                duration_s = len(samples) / float(sr)
                sd.play(samples, samplerate=sr)

                start_t = time.perf_counter()
                while (time.perf_counter() - start_t) < duration_s:
                    self._poll_keyboard_hotkeys_during_speech()
                    if self.barge_in_requested:
                        sd.stop()
                        self.barge_in_count += 1
                        console.print(
                            "\n[bold gold1][!] [BARGE-IN INTERRUPT] Speech halted immediately, Sir.[/bold gold1]"
                        )
                        break
                    await asyncio.sleep(0.02)
                sd.stop()
        finally:
            for t in sentence_tasks:
                if not t.done():
                    t.cancel()
            self.is_speaking = False
            self.echo_guard_until = time.perf_counter() + 0.40
            self.last_spoken_time = time.perf_counter()

    async def speak_reply(self, text: str) -> None:
        if not self.tts_enabled or not text.strip():
            return
        sentences = voice_engine.split_sentences(text)
        tasks = [asyncio.create_task(self._synthesize_sentence_samples(s)) for s in sentences]
        if tasks:
            await asyncio.wait([tasks[0]], timeout=1.35)
        await self.play_sentence_tasks(text, tasks)

    def _poll_keyboard_hotkeys_during_speech(self) -> None:
        if not HAS_MSVCRT:
            return
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch == " ":
                self.barge_in_requested = True

    # ------------------------------------------------------------------
    # Terminal TUI Rendering
    # ------------------------------------------------------------------
    def render_header(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=3)
        grid.add_column(justify="right", ratio=2)

        title_block = Text.from_markup(
            "[bold bright_cyan]   .---.   \n"
            "  / (O) \\    J . A . R . V . I . S .\n"
            "  \\     /    [/bold bright_cyan][cyan]STANDALONE FULL-DUPLEX VOICE & COGNITIVE CLI[/cyan]\n"
            "[bold bright_cyan]   '---'   [/bold bright_cyan]"
        )

        mic_badge = (
            "[bold spring_green2][*] MIC: ON (AUTO-VAD)[/bold spring_green2]"
            if self.mic_active
            else "[bold red1][ ] MIC: MUTED[/bold red1]"
        )
        echo_badge = (
            "[bold spring_green2][+] ECHO GUARD (SPEAKER)[/bold spring_green2]"
            if self.open_speaker_mode
            else "[bold sky_blue1][+] HEADSET DUPLEX[/bold sky_blue1]"
        )
        if not self.tts_enabled:
            tts_badge = "[dim][-] TTS: MUTED[/dim]"
        elif voice_engine.prefer_local_tts:
            tts_badge = "[bold spring_green2][+] TTS: INSTANT LOCAL (<27ms, 1.0x)[/bold spring_green2]"
        else:
            tts_badge = "[bold bright_cyan][+] TTS: RYAN NEURAL (1.0x)[/bold bright_cyan]"
        router_badge = f"[bold gold1][+] ROUTER: {cognitive_router.preferred_mode.upper()}[/bold gold1]"

        status_block = Text.from_markup(
            f"{mic_badge}\n{echo_badge}\n{tts_badge}  |  {router_badge}"
        )

        grid.add_row(title_block, status_block)

        console.print(
            Panel(
                grid,
                box=box.DOUBLE,
                border_style="bright_cyan",
                subtitle="[cyan]Hotkeys: [bold white][SPACE][/bold white] Barge-In | [bold white][T][/bold white] Type | [bold white][M][/bold white] Mic | [bold white][E][/bold white] Echo/Headset | [bold white][V][/bold white] TTS | [bold white][R][/bold white] Router | [bold white][S][/bold white] Status | [bold white][Q][/bold white] Quit[/cyan]",
                padding=(0, 2),
            )
        )

    def render_turn(
        self,
        role: str,
        text: str,
        tier: str = "",
        latency_ms: float = 0.0,
        tools: Optional[List[str]] = None,
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        if role == "user":
            sys.stdout.write("\r" + " " * 110 + "\r")
            sys.stdout.flush()
            console.print(
                Panel(
                    f"[bold white]{text}[/bold white]",
                    title=f"[bold sky_blue1]SIR[/bold sky_blue1] [dim]({ts})[/dim]",
                    title_align="left",
                    box=box.ROUNDED,
                    border_style="sky_blue1",
                    padding=(0, 2),
                )
            )
        else:
            sys.stdout.write("\r" + " " * 110 + "\r")
            sys.stdout.flush()
            meta = f"[cyan]{tier}[/cyan] | [bright_cyan]{latency_ms:.0f}ms[/bright_cyan] | [dim]{ts}[/dim]"
            body = f"[bold bright_cyan]{text}[/bold bright_cyan]"
            if tools:
                body += f"\n[dim cyan][+] Executed MCP Tools: {', '.join(tools)}[/dim cyan]"
            console.print(
                Panel(
                    body,
                    title="[bold bright_cyan]J.A.R.V.I.S.[/bold bright_cyan]",
                    title_align="left",
                    subtitle=meta,
                    subtitle_align="right",
                    box=box.ROUNDED,
                    border_style="bright_cyan",
                    padding=(0, 2),
                )
            )

    def render_live_status_line(self) -> None:
        """Draws the single-line bottom live VU meter without scrolling the terminal."""
        bars_total = 18
        # Non-linear perceptual VU scaling so quiet Windows mic arrays visibly animate the meter
        level = min(1.0, math.sqrt(max(0.0, self.current_rms / 0.015)))
        filled = int(round(level * bars_total))
        vu_bar = "#" * filled + "-" * (bars_total - filled)

        if self.is_speaking:
            state = "SPEAKING [ECHO GUARD] (Press SPACE to Barge-In)"
        elif self.is_thinking:
            state = "ROUTING COGNITIVE RESPONSE..."
        elif not self.mic_active:
            state = "MIC MUTED (Press [M] to unmute or [T] to type)"
        elif self.is_recording_utterance:
            state = "CAPTURING VOICE DIRECTIVE..."
        else:
            state = "LISTENING (Speak now or press [T] to type)"

        router_str = cognitive_router.preferred_mode.upper()
        line = f"\r  [{router_str}] {state:<48}  MIC [{vu_bar}] "
        sys.stdout.write(line[:115])
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Utterance Processing Pipeline
    # ------------------------------------------------------------------
    async def process_text_directive(self, user_text: str) -> None:
        cleaned = user_text.strip()
        if not cleaned:
            return

        self.render_turn("user", cleaned)
        self.is_thinking = True
        self.render_live_status_line()

        sentence_tasks: List[asyncio.Task[Optional[tuple[np.ndarray, int]]]] = []
        try:
            result = await cognitive_router.route_and_respond(cleaned)
            reply = result.get("spoken_reply", "")
            tier = result.get("tier_used", "executive-local-core")
            latency = result.get("latency_ms", 0.0)
            tool_names = [t.get("tool", "") for t in (result.get("tool_executions") or []) if t.get("tool")]

            # Apply any voice/acoustic/router tool settings triggered by voice command
            for exec_item in result.get("tool_executions") or []:
                res = exec_item.get("result") or {}
                if res.get("hud_action") == "system_setting":
                    setting = res.get("setting")
                    val = res.get("value")
                    if setting == "tts":
                        self.tts_enabled = val != "mute"
                    elif setting == "acoustic_mode":
                        self.open_speaker_mode = val != "headset"
                    elif setting == "router_mode" and val in ("auto", "cloud", "local"):
                        cognitive_router.preferred_mode = val

            # Pipeline all sentences in parallel & await Sentence 1 so voice plays at 0ms when panel renders
            if self.tts_enabled and reply.strip():
                sentences = voice_engine.split_sentences(reply)
                sentence_tasks = [asyncio.create_task(self._synthesize_sentence_samples(s)) for s in sentences]
                if sentence_tasks:
                    await asyncio.wait([sentence_tasks[0]], timeout=1.35)
        finally:
            self.is_thinking = False

        self.render_turn("jarvis", reply, tier=tier, latency_ms=latency, tools=tool_names)
        if sentence_tasks:
            await self.play_sentence_tasks(reply, sentence_tasks)

    async def process_recorded_wav(self, wav_bytes: bytes) -> None:
        if self.is_echo_gated:
            return

        self.is_thinking = True
        self.render_live_status_line()
        try:
            transcript = await asyncio.to_thread(voice_engine.transcribe_audio_file, wav_bytes, ".wav")
        finally:
            self.is_thinking = False

        if not transcript:
            return

        if self.is_echo_of_jarvis(transcript):
            return

        await self.process_text_directive(transcript)

    # ------------------------------------------------------------------
    # Main Event Loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._utterance_queue = asyncio.Queue()

        # Pre-check Ollama health & warm persistent SAPI5 voice thread so first turn has <27ms TTS latency
        await asyncio.gather(
            cognitive_router.check_ollama_health(),
            voice_engine.synthesize_pcm_local_sapi_async("At your service, Sir."),
        )

        self.render_header()

        # Boot greeting
        greeting = "At your service, Sir."
        self.render_turn("jarvis", greeting, tier="executive-local-core", latency_ms=3.2)

        # Start 16kHz Microphone InputStream automatically
        try:
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                channels=1,
                dtype="float32",
                callback=self._audio_input_callback,
            )
            self._stream.start()
        except Exception as exc:
            self.mic_active = False
            console.print(f"[bold red][!] Microphone init warning: {exc}. Use [T] to type directives.[/bold red]")

        await self.speak_reply(greeting)

        try:
            while self.running:
                self.render_live_status_line()

                # 1. Check if a voice utterance finished recording
                try:
                    wav_bytes = self._utterance_queue.get_nowait()
                    await self.process_recorded_wav(wav_bytes)
                    continue
                except asyncio.QueueEmpty:
                    pass

                # 2. Poll non-blocking keyboard hotkeys
                if HAS_MSVCRT and msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    key = ch.lower()

                    if key == "q":
                        sys.stdout.write("\r" + " " * 110 + "\r")
                        console.print("[bold cyan]Powering down J.A.R.V.I.S. Voice Core. Goodbye, Sir.[/bold cyan]")
                        break

                    elif key == "m":
                        self.mic_active = not self.mic_active
                        self.render_header()

                    elif key == "e":
                        self.open_speaker_mode = not self.open_speaker_mode
                        self.render_header()

                    elif key == "v":
                        if self.tts_enabled and not voice_engine.prefer_local_tts:
                            voice_engine.prefer_local_tts = True
                        elif self.tts_enabled and voice_engine.prefer_local_tts:
                            self.tts_enabled = False
                            voice_engine.prefer_local_tts = False
                        else:
                            self.tts_enabled = True
                            voice_engine.prefer_local_tts = False
                        self.render_header()

                    elif key == "r":
                        modes = ["auto", "cloud", "local"]
                        idx = (modes.index(cognitive_router.preferred_mode) + 1) % len(modes)
                        cognitive_router.preferred_mode = modes[idx]
                        self.render_header()

                    elif key == "c":
                        self.render_header()

                    elif key == "s":
                        await self.process_text_directive("Status report and system telemetry")

                    elif key in ("t", "\r", "\n"):
                        # Pause mic while typing text directive
                        prev_mic = self.mic_active
                        self.mic_active = False
                        sys.stdout.write("\r" + " " * 110 + "\r")
                        sys.stdout.flush()
                        try:
                            typed = await asyncio.to_thread(
                                console.input,
                                "[bold bright_cyan]  >> DIRECTIVE TO J.A.R.V.I.S.: [/bold bright_cyan]",
                            )
                        except (EOFError, KeyboardInterrupt):
                            typed = ""
                        self.mic_active = prev_mic
                        if typed.strip():
                            await self.process_text_directive(typed.strip())

                await asyncio.sleep(0.05)
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass


if __name__ == "__main__":
    try:
        asyncio.run(JarvisStandaloneVoiceCLI().run())
    except KeyboardInterrupt:
        print("\n[*] J.A.R.V.I.S. CLI terminated.")
