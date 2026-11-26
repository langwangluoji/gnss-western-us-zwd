param([Parameter(Mandatory=$true)][int]$TrainingProcessId)
$taskProject = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskProject
$env:PYTHONPATH = Join-Path $taskProject 'src'
$env:PYTHONIOENCODING = 'utf-8'
Wait-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' -u tools/report_generator_mechanisms.py --members 16
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' -u tools/confirm_generator_mechanisms.py
exit $LASTEXITCODE
