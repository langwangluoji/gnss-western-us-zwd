$taskLog = Join-Path $PSScriptRoot 'queue.stdout.log'
$taskCommand = "Get-Content -LiteralPath '$taskLog' -Tail 30 -Wait"
$taskEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($taskCommand))
Start-Process powershell.exe -WindowStyle Normal -ArgumentList @('-NoExit','-EncodedCommand',$taskEncoded)
