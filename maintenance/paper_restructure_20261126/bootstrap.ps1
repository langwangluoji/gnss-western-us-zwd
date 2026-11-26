param([string]$Root = (Get-Location).Path)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
function Junction([string]$LinkRelative,[string]$TargetRelative) {
    $link = [IO.Path]::GetFullPath((Join-Path $taskRoot $LinkRelative))
    $target = [IO.Path]::GetFullPath((Join-Path $taskRoot $TargetRelative))
    foreach ($candidate in @($link,$target)) { if (-not $candidate.StartsWith($taskRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw "Out of scope: $candidate" } }
    if ((Test-Path -LiteralPath $target) -and (Test-Path -LiteralPath $link)) {
        $existing = Get-Item -LiteralPath $link -Force
        if ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            $bound = [IO.Path]::GetFullPath(@($existing.Target)[0]).TrimEnd('\')
            if (-not $bound.Equals($target.TrimEnd('\'),[StringComparison]::OrdinalIgnoreCase)) { throw "Existing compatibility link has different target: $link" }
        } else { throw "Compatibility path already contains a real directory: $link" }
    }
    if ((Test-Path -LiteralPath $target) -and -not (Test-Path -LiteralPath $link)) {
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($link)) -Force | Out-Null
        New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    }
}
Junction 'research_pipeline' 'artifacts/legacy_layout/research_pipeline'
$study = 'artifacts/legacy_layout/research_pipeline/studies/us_dense_zwd_20261105_v1'
$oldEngine = "$study/03_code/gnss_field_server"
foreach ($name in @('tools','tests','configs','slurm','docs')) { Junction "$oldEngine/$name" "research/engine/$name" }
Junction "$oldEngine/src" 'src'
Junction 'research/engine/src' 'src'
if (Test-Path -LiteralPath (Join-Path $taskRoot "$oldEngine/results")) { Junction 'research/engine/results' "$oldEngine/results" }
else { Junction 'research/engine/results' 'experiments/us' }
foreach ($name in @('build','local_validation')) { Junction "research/engine/$name" "$oldEngine/$name" }
foreach ($entry in (Get-ChildItem -LiteralPath (Join-Path $taskRoot 'research') -Directory)) {
    if ($entry.Name -match '202611(23|25|26)$') { Junction "maintenance/$($entry.Name)" "research/$($entry.Name)" }
}
foreach ($studyName in @('europe_transfer_20261126','europe_vmf3_20261126')) {
    foreach ($name in @('cache','runs','probability','evaluation')) { Junction "research/$studyName/$name" "artifacts/campaigns/$studyName/$name" }
}
