$taskFlag = Join-Path $PSScriptRoot 'PAUSE'
Set-Content -LiteralPath $taskFlag -Value 'pause at safe boundary' -Encoding utf8
Write-Output 'Pause requested. Inspect status.json for paused state before resuming.'
