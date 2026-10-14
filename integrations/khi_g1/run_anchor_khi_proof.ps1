param(
    [Parameter(Mandatory = $true)]
    [string]$HumanVideo,

    [string]$KhiRoot = 'C:\Users\salio\Downloads\G1_COMPLETE_HANDOFF_20260915_01\source\g1-aprilcube-autonomy',
    [string]$PoseModel = 'C:\Users\salio\.cache\anchor-models\pose_landmarker_lite.task',
    [string]$Python = 'C:\Users\salio\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [string]$OutputDirectory = 'evidence\khi_g1'
)

$ErrorActionPreference = 'Stop'
$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PipelineSource = Join-Path $Repository 'pipeline\src'
$OutputDirectory = Join-Path $Repository $OutputDirectory
$Urdf = Join-Path $KhiRoot 'g1_29dof_pc2.urdf'
$HumanMotion = Join-Path $OutputDirectory 'human_motion_v1.json'
$G1Trajectory = Join-Path $OutputDirectory 'hcs_g1_retarget_v1.json'
$KhiReference = Join-Path $OutputDirectory 'khi_g1_reference.pkl'
$KhiObjectReference = Join-Path $OutputDirectory 'khi_object_reference.pkl'
$Manifest = Join-Path $OutputDirectory 'anchor_khi_proof_manifest_v1.json'
$Receipt = Join-Path $OutputDirectory 'anchor_khi_proof_receipt_v1.json'

foreach ($Required in @($HumanVideo, $KhiRoot, $PoseModel, $Python, $Urdf)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required path is missing: $Required"
    }
}

$RepositoryRoot = [System.IO.Path]::GetPathRoot($Repository)
$RepositoryDrive = [System.IO.DriveInfo]::GetDrives() |
    Where-Object { $_.Name -eq $RepositoryRoot } |
    Select-Object -First 1
if ($null -eq $RepositoryDrive) {
    throw "Could not resolve the filesystem drive for $Repository"
}
$FreeBytes = [double]$RepositoryDrive.AvailableFreeSpace
Write-Host ('FREE_DISK_GIB: ' + [math]::Round($FreeBytes / 1GB, 2))
if ($FreeBytes -lt 1GB) {
    throw "Less than 1 GiB is free; refusing to start even a no-copy proof run."
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$env:PYTHONPATH = $PipelineSource

function Invoke-Anchor {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Host ('COMMAND: ' + $Python + ' -m anchor.cli ' + ($Arguments -join ' '))
    & $Python -m anchor.cli @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Anchor command failed with exit code $LASTEXITCODE"
    }
}

Invoke-Anchor extract-human-pose --video $HumanVideo --model $PoseModel --output $HumanMotion --sample-fps 5
Invoke-Anchor retarget-g1 --human-motion $HumanMotion --urdf $Urdf --profile khi_tabletop_right --output $G1Trajectory
Invoke-Anchor export-khi-g1 --trajectory $G1Trajectory --output $KhiReference
Invoke-Anchor validate-khi-g1 $KhiReference
Invoke-Anchor export-khi-object --robot-reference $KhiReference --output $KhiObjectReference --relative-position-m 0.487 -0.180 -0.089
Invoke-Anchor validate-khi-pair --robot-reference $KhiReference --object-reference $KhiObjectReference
Invoke-Anchor proof-khi --khi-root $KhiRoot --human-video $HumanVideo --human-motion $HumanMotion --g1-trajectory $G1Trajectory --khi-reference $KhiReference --khi-object-reference $KhiObjectReference --output $Manifest
Invoke-Anchor proof-verify $Manifest --receipt $Receipt

Write-Host "PASS: $Receipt"

