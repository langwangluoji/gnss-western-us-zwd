param([Parameter(Mandatory=$true)][int]$TrainingProcessId)
$taskProject = Split-Path -Parent $PSScriptRoot
$taskRun = Join-Path $taskProject 'results/20261123_enhanced_encoder_generative_v03'
Wait-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
$taskResult = Join-Path $taskRun 'generators_results.json'
if (Test-Path -LiteralPath $taskResult) {
    $taskState = Get-Content -LiteralPath $taskResult -Raw | ConvertFrom-Json
    if ($taskState.status -eq 'COMPLETE') {
        Set-Location -LiteralPath $taskProject
        & 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' -u tools/report_independent_generators.py
        exit $LASTEXITCODE
    }
}
Write-Output 'Training is incomplete; final report was not fabricated. Resume the same registered training entry.'
