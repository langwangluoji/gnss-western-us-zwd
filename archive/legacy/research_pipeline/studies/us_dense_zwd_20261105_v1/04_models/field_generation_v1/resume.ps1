$taskStatus = Join-Path $PSScriptRoot 'status.json'
if (Test-Path -LiteralPath $taskStatus) {
    $taskState = Get-Content -LiteralPath $taskStatus -Raw | ConvertFrom-Json
    if ($taskState.pid -and (Get-Process -Id $taskState.pid -ErrorAction SilentlyContinue)) {
        throw 'Recorded training process is still alive; do not launch a duplicate queue.'
    }
}
$taskFlag = Join-Path $PSScriptRoot 'PAUSE'
if (Test-Path -LiteralPath $taskFlag) { Remove-Item -LiteralPath $taskFlag }
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($taskLog in @('queue.stdout.log', 'queue.stderr.log')) {
    $taskLogPath = Join-Path $PSScriptRoot $taskLog
    if ((Test-Path -LiteralPath $taskLogPath) -and (Get-Item -LiteralPath $taskLogPath).Length -gt 0) {
        Copy-Item -LiteralPath $taskLogPath -Destination (Join-Path $PSScriptRoot ($taskStamp + '.' + $taskLog))
    }
}
$taskScript = Join-Path $PSScriptRoot 'run_queue.ps1'
Start-Process powershell -WindowStyle Hidden -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+$taskScript+'"')) -RedirectStandardOutput (Join-Path $PSScriptRoot 'queue.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'queue.stderr.log')
