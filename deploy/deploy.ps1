# Build with Cloud Build → push to Artifact Registry → deploy to Cloud Run + L4.
#
# Prereqs (one-time):
#   gcloud auth login
#   gcloud config set project sportsanalytics-495612
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com `
#       artifactregistry.googleapis.com storage.googleapis.com
#   gcloud artifacts repositories create tracker --repository-format=docker `
#       --location=us-central1                          # ok if already exists
#   # Bucket sportsanalytics-495612-clips already exists in this project.
#
# Usage:
#   .\deploy\deploy.ps1
#   .\deploy\deploy.ps1 -Tag v2.1
#   .\deploy\deploy.ps1 -MinInstances 1                  # keep L4 warm (~$510/mo)

param(
    [string] $ProjectId     = "sportsanalytics-495612",
    [string] $Bucket        = "sportsanalytics-495612-clips",
    [string] $Region        = "us-central1",
    [string] $Repo          = "tracker",
    [string] $Service       = "cricket-tracker",
    [string] $Tag           = "v3-physics",
    [int]    $MinInstances  = 0
)

$ErrorActionPreference = "Stop"
$Root  = Split-Path -Parent $PSScriptRoot
$Image = "$Region-docker.pkg.dev/$ProjectId/$Repo/$Service`:$Tag"

# 1. Build + push via Cloud Build (no local Docker needed).
Push-Location $Root
try {
    Write-Host "Building $Image via Cloud Build..." -ForegroundColor Cyan
    gcloud builds submit `
        --project=$ProjectId `
        --region=$Region `
        --tag=$Image `
        --machine-type=e2-highcpu-8 `
        --timeout=3600s `
        .
    if ($LASTEXITCODE -ne 0) { throw "Cloud Build failed" }
} finally {
    Pop-Location
}

# 2. Deploy to Cloud Run with L4 GPU.
Write-Host "Deploying $Service to Cloud Run + L4..." -ForegroundColor Cyan
gcloud run deploy $Service `
    --project=$ProjectId `
    --region=$Region `
    --image=$Image `
    --execution-environment=gen2 `
    --gpu=1 `
    --gpu-type=nvidia-l4 `
    --no-gpu-zonal-redundancy `
    --cpu=8 `
    --memory=32Gi `
    --no-cpu-throttling `
    --concurrency=1 `
    --min-instances=$MinInstances `
    --max-instances=1 `
    --timeout=3600 `
    --port=8080 `
    --no-allow-unauthenticated `
    --service-account=api-sa@$ProjectId.iam.gserviceaccount.com `
    --set-env-vars="GCS_BUCKET=$Bucket,SIGNED_URL_TTL_SECONDS=3600,MAX_UPLOAD_MB=500,MAX_CLIP_SECONDS=30,IDLE_SHUTDOWN_SECONDS=300,TRACKER_REJECT_SHADOWS=1,TRACKER_USE_MAGNUS=1,TRACKER_USE_DRAG=1"

if ($LASTEXITCODE -ne 0) { throw "Cloud Run deploy failed" }

$Url = gcloud run services describe $Service --project=$ProjectId --region=$Region --format="value(status.url)"
Write-Host ""
Write-Host "Deployed: $Url" -ForegroundColor Green
Write-Host "Access (IAM-protected):"
Write-Host "  curl -H `"Authorization: Bearer `$(gcloud auth print-identity-token)`" $Url/api/track/health"
Write-Host ""
Write-Host "To open to the public:"
Write-Host "  gcloud run services add-iam-policy-binding $Service --region=$Region --member=allUsers --role=roles/run.invoker"
