$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$taskRoot = $PSScriptRoot
$studyRoot = Split-Path (Split-Path $taskRoot -Parent) -Parent
$codeRoot = Join-Path $studyRoot '03_code'
Set-Location -LiteralPath $codeRoot
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' -u -m field_generation_v1
exit $LASTEXITCODE
