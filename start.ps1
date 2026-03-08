$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional microphone support (Windows):
# pip install pipwin
# pipwin install pyaudio

streamlit run app.py
