$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.11 environment' }
}
uv sync --extra dev --extra research
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.venv\Scripts\python.exe' -m dentalselfie doctor
if ($LASTEXITCODE -ne 0) { throw 'Doctor failed' }
