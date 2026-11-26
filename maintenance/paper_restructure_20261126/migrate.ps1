param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$taskRoot = (Get-Location).Path.TrimEnd('\')
if ((Test-Path -LiteralPath (Join-Path $taskRoot 'artifacts/legacy_layout/research_pipeline')) -and
    (Test-Path -LiteralPath (Join-Path $taskRoot 'research/engine/tools'))) {
    Write-Output 'Physical migration already completed; use bootstrap.ps1 and verification.'
    return
}
$engineRelative = 'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
function Checked([string]$Relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $taskRoot $Relative))
    if (-not $absolute.StartsWith($taskRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Out of scope: $absolute" }
    return $absolute
}
function Relocate([string]$From, [string]$To) {
    $source = Checked $From; $target = Checked $To
    if (-not (Test-Path -LiteralPath $source)) { return }
    if (Test-Path -LiteralPath $target) { throw "Destination already exists: $target" }
    if ($Execute) {
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($target)) -Force | Out-Null
        Move-Item -LiteralPath $source -Destination $target
    }
    Write-Output "$From -> $To"
}
if (-not (Test-Path -LiteralPath (Checked 'maintenance/paper_restructure_20261126/inventory_before.local.csv'))) { throw 'Inventory required first' }
foreach ($name in @('src','tools','tests','configs','slurm','docs')) {
    $destination = if ($name -eq 'src') { 'src' } else { "research/engine/$name" }
    Relocate "$engineRelative/$name" $destination
}
foreach ($name in @('README.md','AGENTS.md','pyproject.toml','PERFORMANCE_BENCHMARK_REPORT.md','PERFORMANCE_CRITIC_REPORT.md','PERFORMANCE_EVALUATION_REPORT.md')) {
    Relocate "$engineRelative/$name" "research/engine/$name"
}
Relocate 'research_pipeline' 'artifacts/legacy_layout/research_pipeline'
$oldMaintenance = Get-ChildItem -LiteralPath (Checked 'maintenance') -Force | Where-Object { $_.Name -notin @('paper_restructure_20261126','__pycache__','.pytest_cache') }
foreach ($entry in $oldMaintenance) { Relocate "maintenance/$($entry.Name)" "research/$($entry.Name)" }
foreach ($study in @('europe_transfer_20261126','europe_vmf3_20261126')) {
    foreach ($name in @('cache','runs','probability','evaluation')) {
        Relocate "research/$study/$name" "artifacts/campaigns/$study/$name"
    }
}
