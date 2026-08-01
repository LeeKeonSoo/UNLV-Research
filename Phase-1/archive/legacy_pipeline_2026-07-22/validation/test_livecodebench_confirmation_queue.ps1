$scriptPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'validation\run_livecodebench_confirmation_queue.ps1'
$source = Get-Content -LiteralPath $scriptPath -Raw

if ($source -notmatch '\$ErrorActionPreference = ''Continue''') {
    throw 'Queue must not promote native stderr warnings to terminating errors.'
}
if ($source -notmatch '\$exitCode = \$LASTEXITCODE') {
    throw 'Queue must capture the Python process exit code explicitly.'
}
if ($source -notmatch 'if \(\$exitCode -ne 0\)') {
    throw 'Queue must fail only when Python returns a nonzero exit code.'
}
if ($source -notmatch 'Get-Process -Id \$WaitPid') {
    throw 'Queue must tolerate a worker that finished before the queue starts.'
}
$gpu0Section = $source.Substring(
    $source.IndexOf('$jobs = if ($GpuIndex -eq 0)'),
    $source.IndexOf("} else {") - $source.IndexOf('$jobs = if ($GpuIndex -eq 0)')
)
if ($gpu0Section -match 'seed = 101') {
    throw 'Queue must not schedule the seed-101 source freeze on GPU 0.'
}

Write-Output '[livecodebench-confirmation-queue] stderr warning contract: pass'
