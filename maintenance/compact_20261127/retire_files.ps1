param(
    [Parameter(Mandatory=$true)][string]$ExpectedPlanSha256,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$allowedRoots = @{project=$taskRoot; cache_d='D:\SITP_cache'}
$planPath = Join-Path $PSScriptRoot 'delete_plan.csv'
$planHash = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($planHash -ne $ExpectedPlanSha256.ToLowerInvariant()) { throw 'Deletion manifest hash mismatch' }
$authorization = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'delete_authorization.json') -Raw | ConvertFrom-Json
if ($planHash -ne $authorization.delete_plan_sha256) { throw 'Authorization plan mismatch' }
$archivePath = Join-Path $PSScriptRoot 'remote_archive.json'
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $authorization.archive_receipt_sha256) { throw 'Archive receipt changed' }
$remote = Get-Content -LiteralPath $archivePath -Raw | ConvertFrom-Json
$rows = @(Import-Csv -LiteralPath $planPath)
$usResults = 'artifacts/legacy_layout/research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/'
$protectedRuns = @('20261123_terrain_diffusion_v10_data','20261125_temporal_continuous_data','20261125_p14_continuous_fusion','20261125_observed_innovation_precision','20261125_innovation_calendar15','20261126_native_weather_cache_v2','20261126_native_weather_means','20261126_within_forecast_increment','20261126_directed_lag_v2','20261126_best_base_diffusion_fusion')
function Get-CheckedTarget($row) {
    if (-not $allowedRoots.ContainsKey($row.scope)) { throw 'Unknown filesystem scope' }
    $rel = $row.path.Replace('\','/')
    if ([IO.Path]::IsPathRooted($rel) -or $rel.Contains(':') -or @($rel.Split('/') | Where-Object {$_ -eq '..'}).Count) { throw 'Unsafe relative path' }
    if (@($rel.Split('/') | Where-Object {$_ -in @('.git','build','__pycache__','.pytest_cache','.superpowers')}).Count) { throw 'Protected directory' }
    $basePath = [IO.Path]::GetFullPath($allowedRoots[$row.scope]).TrimEnd('\')
    $target = [IO.Path]::GetFullPath((Join-Path $basePath $rel))
    if (-not $target.StartsWith($basePath+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Target outside allowed root' }
    if ($row.scope -eq 'project') {
        if (-not $rel.StartsWith($usResults)) { throw 'Outside retired US results' }
        $run = $rel.Substring($usResults.Length).Split('/')[0]
        if ($run -in $protectedRuns) { throw 'Core experiment is protected' }
    } else {
        $allowedD = $rel -match '^europe_dense_zwd_20261104_v1/(physics|validation)/[^/]+\.h5$' -or
                    $rel -match '^europe_ref_gan_20261103_v1/expanded_prepared/[^/]+\.h5$' -or
                    $rel -eq 'europe_ref_gan_20261103_v1/frozen_2023/train_reference_utc.h5.partial'
        if (-not $allowedD) { throw 'Outside retired D cache files' }
    }
    $extended = '\\?\'+$target
    $info = Get-Item -LiteralPath $extended -Force
    if ($info.PSIsContainer) { throw 'Only individual files may be removed' }
    if (($info.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse file' }
    $cursor = $info.Directory
    $reachedRoot = $false
    while ($null -ne $cursor) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse point in target path' }
        $normal = $cursor.FullName.Replace('\\?\','').TrimEnd('\')
        if ($normal -eq $basePath) { $reachedRoot = $true; break }
        $cursor = $cursor.Parent
    }
    if (-not $reachedRoot) { throw 'Allowed root not reached' }
    if ($info.Length -ne [long]$row.size) { throw 'File size changed' }
    if ((Get-FileHash -LiteralPath $extended -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw 'File hash changed' }
    if ($row.action -eq 'archive_model') {
        $model = $remote.models.PSObject.Properties[$row.sha256].Value
        if ($null -eq $model -or -not $model.verified_remote) { throw 'Model archive is not verified' }
    } elseif ($row.action -ne 'retire_derived') { throw 'No retirement authorization' }
    return $extended
}
# Validate the entire batch before the first deletion.
$checked = 0
foreach ($row in $rows) {
    $null = Get-CheckedTarget $row
    $checked++
    if ($checked % 100 -eq 0) { Write-Output "Validated $checked / $($rows.Count)" }
}
Write-Output "Preflight passed: $checked individual files"
if (-not $Execute) { return }
$logPath = Join-Path $PSScriptRoot 'deletion_log.jsonl'
if (Test-Path -LiteralPath $logPath) { throw 'Deletion log already exists; review partial run before resuming' }
$removed = 0; $releasedBytes = [long]0
foreach ($row in $rows) {
    $target = Get-CheckedTarget $row
    Remove-Item -LiteralPath $target -Force
    if (Test-Path -LiteralPath $target) { throw 'File still exists after removal' }
    @{scope=$row.scope;path=$row.path;size=[long]$row.size;sha256=$row.sha256;action=$row.action;removed=$true} |
        ConvertTo-Json -Compress | Add-Content -LiteralPath $logPath -Encoding utf8
    $removed++; $releasedBytes += [long]$row.size
    if ($removed % 100 -eq 0) { Write-Output "Removed $removed / $($rows.Count)" }
}
@{status='completed';files=$removed;bytes=$releasedBytes;manifest_sha256=$planHash;recursive_deletion=$false} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'deletion_receipt.json') -Encoding utf8
Write-Output "Removed $removed files, $releasedBytes bytes"
