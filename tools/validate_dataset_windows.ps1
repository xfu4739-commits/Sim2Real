[CmdletBinding()]
param(
    [string]$DatasetRoot = "C:\Users\10115\Datasets\Sim2Real-Fire",
    [ValidateSet("structure", "sample", "full")]
    [string]$Level = "full",
    [string]$Config = "",
    [string]$Report = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Validator = Join-Path $PSScriptRoot "validate_dataset.py"

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $RepoRoot "config\config.yaml"
}
if ([string]::IsNullOrWhiteSpace($Report)) {
    $Report = Join-Path $RepoRoot "output\dataset_validation_windows.json"
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $Python) {
    $Python = Get-Command py -ErrorAction SilentlyContinue
}
if ($null -eq $Python) {
    throw "Python not found on PATH (tried 'python' and 'py'). Install dependencies from requirements.txt."
}

& $Python.Source $Validator `
    --dataset-root $DatasetRoot `
    --config $Config `
    --level $Level `
    --report $Report
exit $LASTEXITCODE
