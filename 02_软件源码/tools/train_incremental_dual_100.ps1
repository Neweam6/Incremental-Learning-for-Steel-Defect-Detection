param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 60,
    [ValidateSet('control', 'dn', 'quality', 'dn_quality')]
    [string]$Variant = 'control',
    [string]$Python = 'python',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$baseCheckpoint = Join-Path $projectRoot 'MODEL/base_output_group1/checkpoint_phase0.pth'
if (-not (Test-Path -LiteralPath $baseCheckpoint -PathType Leaf)) {
    throw "Base checkpoint not found: $baseCheckpoint"
}
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$variantSuffix = if ($Variant -eq 'control') { '' } else { "_${Variant}" }
$outputPath = Join-Path $projectRoot "incremental_output_100_dual${variantSuffix}_$runStamp"
if (Test-Path -LiteralPath $outputPath) { throw "Output directory exists: $outputPath" }
$trainArgs = @(
    '-u', 'incremental_train.py',
    '--incremental_only', '--split_group', '1',
    '--incremental_arch', 'dual', '--resume', $baseCheckpoint,
    '--new_class_samples', '100', '--epochs_incremental', [string]$Epochs,
    '--warmstart_epochs', '0', '--seed', '42',
    '--dual_lr', '1e-4', '--dual_head_lr', '5e-4',
    '--cls_loss_coef', '2', '--bbox_loss_coef', '5', '--giou_loss_coef', '2',
    '--set_cost_class', '2', '--set_cost_bbox', '5', '--set_cost_giou', '2',
    '--replay_buffer_size', '50', '--post_nms_mode', 'perclass',
    '--accum_steps_inc', '2', '--batch_size', '6', '--num_workers', '2',
    '--device', 'cuda', '--output_dir', $outputPath
)
if ($Variant -in @('dn', 'dn_quality')) { $trainArgs += @('--dual_dn_groups', '4') }
if ($Variant -in @('quality', 'dn_quality')) { $trainArgs += '--dual_quality_loss' }
Write-Host "Output: $outputPath"
Write-Host "Architecture: dual; variant=$Variant; 100 fixed images/new class; epochs=$Epochs"
Write-Host 'Best checkpoints: best_overall_phase1.pth (COCO AP50); best_legacy_phase1.pth (historical metric)'
if ($DryRun) { Write-Output $Python; Write-Output $trainArgs; return }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $projectRoot
try {
    & $Python @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed: exit $LASTEXITCODE. Output: $outputPath" }
}
finally { Pop-Location }
