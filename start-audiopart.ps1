<#
.SYNOPSIS
    Project J.A.R.V.I.S. -- 100% Standalone Voice & Cognitive AI Terminal CLI Launcher
    Works from ANY folder or drive. Automatically resolves or bootstraps its own
    Python .venv, starts the local Ollama GPU daemon, and launches jarvis_cli.py.
#>

$ErrorActionPreference = "Stop"
$AudioPartDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $AudioPartDir

# 1. Resolve Python .venv (checks local audiopart\.venv first, then parent .venv)
$VenvPython = "$AudioPartDir\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython) -and (Test-Path "$RootDir\.venv\Scripts\python.exe")) {
    $VenvPython = "$RootDir\.venv\Scripts\python.exe"
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[*] Bootstrapping standalone Python .venv inside $AudioPartDir\.venv ..." -ForegroundColor Cyan
    python -m venv "$AudioPartDir\.venv"
    $VenvPython = "$AudioPartDir\.venv\Scripts\python.exe"
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r "$AudioPartDir\requirements.txt"
}

# 2. Ensure Ollama GPU daemon is warm in background if installed
$OllamaExe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $OllamaExe -and (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
    $OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
}

$SpawnedOllamaProc = $null
if ($OllamaExe) {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $alreadyRunning = $false
    try {
        $tcp.Connect("127.0.0.1", 11434)
        $tcp.Close()
        $alreadyRunning = $true
    } catch {
        $alreadyRunning = $false
    }

    if (-not $alreadyRunning) {
        Write-Host "[*] Launching local Ollama AI engine in background..." -ForegroundColor Cyan
        $SpawnedOllamaProc = Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden -PassThru
        Start-Sleep -Milliseconds 800
    }
}

# 3. Launch Standalone J.A.R.V.I.S. Rich Terminal Voice CLI
Push-Location $AudioPartDir
try {
    & $VenvPython "$AudioPartDir\jarvis_cli.py"
} finally {
    if ($SpawnedOllamaProc -and -not $SpawnedOllamaProc.HasExited) {
        Write-Host "`n[*] Terminating spawned Ollama background process tree (PID: $($SpawnedOllamaProc.Id))..." -ForegroundColor Yellow
        & taskkill /F /T /PID $SpawnedOllamaProc.Id 2>$null | Out-Null
    }
    Pop-Location
}
