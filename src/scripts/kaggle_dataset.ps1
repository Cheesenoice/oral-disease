param(
    [Parameter(Mandatory=$true)][ValidateSet('code','data')][string]$Kind,
    [ValidateSet('Create','Version')][string]$Mode = 'Create',
    [string]$Message = 'Update DentalSelfie artifacts'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot
& '.venv\Scripts\python.exe' -m dentalselfie package $Kind
if ($LASTEXITCODE -ne 0) { throw 'Packaging failed; nothing uploaded' }
# CLI 2.2.3 on this Windows machine failed upload with a relative -p path.
$stagePath = (Resolve-Path -LiteralPath ".venv/work/kaggle/staging/$Kind").Path
if ($Mode -eq 'Create') {
    $publicationConfig = Get-Content -LiteralPath 'src/configs/kaggle.json' -Raw | ConvertFrom-Json
    if ($publicationConfig.private) {
        kaggle datasets create -p $stagePath
    } else {
        kaggle datasets create -p $stagePath --public
    }
} else {
    kaggle datasets version -p $stagePath -m $Message
}
if ($LASTEXITCODE -ne 0) { throw 'Kaggle upload failed; inspect error before retrying' }
