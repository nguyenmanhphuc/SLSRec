<#
.SYNOPSIS
    Gate A/B experiment (original SLSRec gate vs adaptive gates) on Windows.

.DESCRIPTION
    PowerShell port of scripts/run_gate_sweep.sh, for Windows PowerShell 5.1.
    Restartable: a run whose test_metrics.json already exists is skipped.

    Every variant is neutral-initialised, so all of them start from exactly the
    original SLSRec layer.  Keep -NumWorkers and -Extra identical across every
    variant, otherwise the runs are not comparable.

.EXAMPLE
    # see the commands without running anything
    .\scripts\run_gate_sweep.ps1 -DryRun

.EXAMPLE
    # the screening pass: 3 variants x 1 seed, capped at 60 epochs
    .\scripts\run_gate_sweep.ps1 -Variants sum,scalar,token -Seeds 42 -MaxEpochs 60 -ValIterations 200 -Jobs 3

.EXAMPLE
    # full comparison (slow on CPU -- see the time estimate it prints)
    .\scripts\run_gate_sweep.ps1 -Jobs 4
#>
param(
    [string]   $Python        = "C:\Users\computer\miniconda3\envs\SemADR\python.exe",
    [string[]] $Variants      = @('sum','scalar','channel','token','context','fusion'),
    [int[]]    $Seeds         = @(42,43,44),
    [int]      $Jobs          = 2,
    [string]   $Dataset       = 'ml-1m',
    [string]   $Granularity   = 'channel',
    [string]   $Extra         = '',
    [int]      $MaxEpochs     = 0,      # 0 = harness default (500)
    [int]      $ValIterations = 0,      # 0 = harness default (500 on ML-1M)
    [int]      $NumWorkers    = 2,      # keep constant across runs: it perturbs the RNG
    [string]   $ExpPrefix     = 'gate',   # change it to run a second sweep without collisions
    [string]   $LogDir        = '',
    [switch]   $DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

if (-not (Test-Path $Python)) { throw "Python not found: $Python  (pass -Python <path>)" }
if ($LogDir -eq '') { $LogDir = "runlogs\${ExpPrefix}s_$Dataset" }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

$FLAGS = @{
    'sum'     = "--gate_mode=sum"
    'scalar'  = "--gate_mode=scalar"
    'channel' = "--gate_mode=channel"
    'token'   = "--gate_mode=token --gate_granularity=$Granularity"
    'context' = "--gate_mode=context --gate_granularity=$Granularity"
    'fusion'  = "--fusion_mode=adaptive --fusion_source=token --gate_granularity=$Granularity"
    'both'    = "--gate_mode=token --fusion_mode=adaptive --gate_granularity=$Granularity"
}

# --- build the job list ---------------------------------------------------- #
$jobsToRun = @()
foreach ($v in $Variants) {
    if (-not $FLAGS.ContainsKey($v)) { throw "unknown variant: $v" }
    foreach ($s in $Seeds) {
        $exp = "${ExpPrefix}_${v}_${Dataset}_seed${s}"
        if (Test-Path "experiments\slsrec\$exp\test_metrics.json") {
            Write-Host "[skip] $exp -- already finished"
            continue
        }
        $a = "train.py --dataset_code=$Dataset --model_code=slsrec --seed=$s " +
             "$($FLAGS[$v]) --num_workers=$NumWorkers --exp_name=$exp"
        if ($MaxEpochs     -gt 0) { $a += " --num_epochs=$MaxEpochs" }
        if ($ValIterations -gt 0) { $a += " --val_iterations=$ValIterations" }
        if ($Extra -ne '')        { $a += " $Extra" }
        $jobsToRun += [pscustomobject]@{ Name = $exp; Args = $a }
    }
}

if ($jobsToRun.Count -eq 0) { Write-Host "[gates] nothing to do"; exit 0 }

if ($DryRun) {
    Write-Host "[gates] DryRun -- $($jobsToRun.Count) run(s), $Jobs at a time:`n"
    foreach ($j in $jobsToRun) { Write-Host "`"$Python`" $($j.Args)" }
    exit 0
}

# --- thread budget --------------------------------------------------------- #
# Without this every process grabs all cores and they thrash each other.
$cores   = [Environment]::ProcessorCount
$threads = [Math]::Max(1, [Math]::Floor($cores / $Jobs))
$env:OMP_NUM_THREADS = $threads
$env:MKL_NUM_THREADS = $threads
$env:PYTHONUNBUFFERED = '1'
Write-Host "[gates] $cores cores, $Jobs concurrent run(s), $threads torch threads each"

# --- warm the dataset cache in ONE process (parallel jobs would race) ------- #
Write-Host "[gates] warming the $Dataset dataset cache"
& $Python -c @"
import sys
sys.argv = ['warm', '--dataset_code=$Dataset', '--model_code=slsrec', '--num_workers=0']
from config import args, set_template
set_template(args)
from dataloader import dataloader_factory
tr, va, te = dataloader_factory(args)
print('[warm] %d train batches, %d items' % (len(tr), args.num_items))
"@
if (-not $?) { throw "dataset warm-up failed" }

Write-Host "[gates] $($jobsToRun.Count) run(s), logs in $LogDir\"
Write-Host "[gates] started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# --- throttled launcher ---------------------------------------------------- #
$running = @()
$failed  = @()
foreach ($j in $jobsToRun) {
    while (@($running | Where-Object { -not $_.Proc.HasExited }).Count -ge $Jobs) {
        Start-Sleep -Seconds 5
    }
    foreach ($done in @($running | Where-Object { $_.Proc.HasExited -and -not $_.Reported })) {
        $done.Reported = $true
        if ($done.Proc.ExitCode -ne 0) { $failed += $done.Name }
        Write-Host "[gates] finished $($done.Name) (exit $($done.Proc.ExitCode))"
    }
    $out = Join-Path $LogDir "$($j.Name).log"
    $err = Join-Path $LogDir "$($j.Name).err"
    Write-Host "[gates] launching $($j.Name)"
    $p = Start-Process -FilePath $Python -ArgumentList $j.Args -NoNewWindow -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    # PowerShell 5.1: touching .Handle caches it.  Without this .ExitCode comes
    # back $null after the process ends, and since $null -ne 0 is $true every
    # run would be reported as failed.
    $null = $p.Handle
    $running += [pscustomobject]@{ Name = $j.Name; Proc = $p; Reported = $false }
}

foreach ($r in $running) { $r.Proc.WaitForExit() }
foreach ($r in $running) {
    $code = $r.Proc.ExitCode
    # belt and braces: if the exit code is still unavailable, judge the run by
    # whether it actually produced its results file.
    if ($null -eq $code) {
        $code = if (Test-Path "experiments\slsrec\$($r.Name)\test_metrics.json") { 0 } else { 1 }
    }
    if ($code -ne 0 -and $failed -notcontains $r.Name) { $failed += $r.Name }
}

Write-Host "[gates] finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if ($failed.Count -gt 0) {
    Write-Host "[gates] FAILED: $($failed -join ', ')  -- check $LogDir\*.err"
    exit 1
}
Write-Host "[gates] all $($jobsToRun.Count) run(s) finished"
Write-Host "[gates] next: `"$Python`" scripts\compare_gates.py --gates --prefix $ExpPrefix"
