param(
    [Parameter(Mandatory = $true)][int]$Gpu,
    [Parameter(Mandatory = $true)][string]$Arm,
    [Parameter(Mandatory = $true)][string]$WaitPath,
    [string]$SeedsCsv = ""
)

$ErrorActionPreference = "Stop"
$repo = "C:\Users\ksl11\Desktop\Code\Projects\unlv\UNLV-Research\Phase-1"
$conda = "C:\Users\ksl11\miniconda3\Scripts\conda.exe"
$logRoot = "D:\UNLV-Research\code_5m_corpus_v2\final_replay_v1\evalplus_natural_v1\logs"
$logPath = Join-Path $logRoot "active_evalplus_$Arm`_gpu$Gpu.log"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
while (-not (Test-Path $WaitPath)) {
    Start-Sleep -Seconds 60
}

Set-Location $repo
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"

if ($Arm -eq "base_no_update") {
    & $conda run --no-capture-output -n research python -m external_evaluation.evalplus_generator --arm $Arm 2>&1 |
        Tee-Object -FilePath $logPath -Append
    exit $LASTEXITCODE
}

foreach ($seedText in $SeedsCsv.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)) {
    $seed = $seedText.Trim()
    & $conda run --no-capture-output -n research python -m external_evaluation.evalplus_generator --arm $Arm --seed $seed 2>&1 |
        Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
