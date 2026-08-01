param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1)]
    [int]$GpuIndex,

    [Parameter(Mandatory = $true)]
    [int]$WaitPid
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$python = 'C:\Users\ksl11\miniconda3\envs\research\python.exe'
$outputDir = Join-Path $projectDir 'outputs\code_livecodebench_confirmation_qwen3_4b'
$jobs = if ($GpuIndex -eq 0) {
    @(
        @{ arm = 'raw_full_natural'; seed = 197 },
        @{ arm = 'curated_v2_natural'; seed = 131 },
        @{ arm = 'curated_v2_natural'; seed = 197 }
    )
} else {
    @(
        @{ arm = 'raw_full_natural'; seed = 239 },
        @{ arm = 'raw_full_natural'; seed = 101 },
        @{ arm = 'curated_v2_natural'; seed = 101 },
        @{ arm = 'curated_v2_natural'; seed = 163 },
        @{ arm = 'curated_v2_natural'; seed = 239 }
    )
}

if (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Wait-Process -Id $WaitPid
}
$env:CUDA_VISIBLE_DEVICES = $GpuIndex
foreach ($job in $jobs) {
    $freeze = if ($job.seed -eq 101) {
        'configs\code_livecodebench_pilot_v1.json'
    } else {
        "configs\code_livecodebench_confirmation_v1\seed$($job.seed).json"
    }
    $stem = "$($job.arm)_seed$($job.seed)"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python '226_run_code_livecodebench_pilot.py' '--arm' $job.arm '--freeze' $freeze '--output-dir' 'outputs\code_livecodebench_confirmation_qwen3_4b' *> (Join-Path $outputDir "$stem.queue.log")
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) {
        throw "Generation failed for $stem"
    }
}
