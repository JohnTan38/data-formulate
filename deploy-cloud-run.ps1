param(
  [string]$PROJECT_ID = "spheric-crow-424609-t1",
  [string]$REGION = "asia-southeast1",
  [string]$SERVICE_NAME = "data-formulator",
  [string]$IMAGE = "",
  [string]$INSTANCE_CONNECTION_NAME = "spheric-crow-424609-t1:asia-southeast1:data-formulator"
)

# Check gcloud is available
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
  Write-Error "gcloud CLI not found. Install Google Cloud SDK and ensure 'gcloud' is on PATH. See https://cloud.google.com/sdk/docs/install"
  exit 1
}
# Check Docker is available
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Error "Docker CLI not found. Install Docker and ensure 'docker' is on PATH. See https://docs.docker.com/get-docker/"
  exit 1
}
# prepare image variable
if (-not $IMAGE) { $IMAGE = "gcr.io/$PROJECT_ID/$SERVICE_NAME:latest" }

# Validate image looks correct (gcr.io/PROJECT/REPO:TAG)
if ($IMAGE -notmatch "^[^/:]+/[^/]+/[^:]+:.+$") {
  Write-Error "Invalid image name: '$IMAGE'. Use format gcr.io/PROJECT/REPO:TAG or pass -IMAGE explicitly."
  exit 1
}

Write-Host "Project: $PROJECT_ID"
Write-Host "Region: $REGION"
Write-Host "Service: $SERVICE_NAME"
Write-Host "Image: $IMAGE"
Write-Host "Cloud SQL instance: $INSTANCE_CONNECTION_NAME"
Write-Host ""

# 1. ensure gcloud is configured
gcloud auth login --quiet
gcloud config set project $PROJECT_ID

# 2. enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com sqladmin.googleapis.com secretmanager.googleapis.com --project $PROJECT_ID

# 3. read DB password securely and store in Secret Manager
Write-Host "Enter the Postgres DB password (will be stored in Secret Manager 'data-formulator-db-password'):"
$secure = Read-Host -AsSecureString
# convert SecureString to plain (used only to create secret; avoid storing plain in files)
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)

$tempFile = [System.IO.Path]::GetTempFileName()
Set-Content -Path $tempFile -Value $plain -NoNewline
Remove-Variable plain, ptr

# create or add secret
$secretName = "data-formulator-db-password"
$exists = & gcloud secrets describe $secretName --project $PROJECT_ID 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Creating secret $secretName in Secret Manager..."
  gcloud secrets create $secretName --replication-policy="automatic" --data-file="$tempFile" --project $PROJECT_ID
} else {
  Write-Host "Adding new version to secret $secretName..."
  gcloud secrets versions add $secretName --data-file="$tempFile" --project $PROJECT_ID
}
Remove-Item $tempFile -Force

# 4. build and push container image using Cloud Build
Write-Host "Building container and submitting to Google Cloud Build..."
gcloud builds submit --tag $IMAGE --project $PROJECT_ID
if ($LASTEXITCODE -ne 0) {
  Write-Error "gcloud builds submit failed. Aborting deploy."
  exit $LASTEXITCODE
}

# 5. deploy to Cloud Run, attach Cloud SQL, and mount secret as env var
Write-Host "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME `
  --image $IMAGE `
  --region $REGION `
  --platform managed `
  --allow-unauthenticated `
  --add-cloudsql-instances $INSTANCE_CONNECTION_NAME `
  --set-env-vars DB_HOST=/cloudsql/$INSTANCE_CONNECTION_NAME,DB_USER=postgres,DB_DATABASE=postgres,EXTERNAL_DB=true `
  --set-secrets DB_PASSWORD=$secretName:latest `
  --project $PROJECT_ID

if ($LASTEXITCODE -ne 0) {
  Write-Error "gcloud run deploy failed."
  exit $LASTEXITCODE
}

# 6. grant service account access to Cloud SQL and Secret Manager
$serviceSa = gcloud run services describe $SERVICE_NAME --region $REGION --format="value(spec.template.spec.serviceAccountName)" --project $PROJECT_ID
if ($serviceSa) {
  Write-Host "Granting roles/cloudsql.client and roles/secretmanager.secretAccessor to $serviceSa ..."
  gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$serviceSa" --role="roles/cloudsql.client" --project $PROJECT_ID
  gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$serviceSa" --role="roles/secretmanager.secretAccessor" --project $PROJECT_ID
} else {
  Write-Host "Could not determine Cloud Run service account. Please grant roles/cloudsql.client and roles/secretmanager.secretAccessor to the Cloud Run service account manually."
}

# 7. show service URL
$svcUrl = gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)" --project $PROJECT_ID
Write-Host ""
Write-Host "Deployment complete. Service URL: $svcUrl"
Write-Host "Note: Ensure py-src/data_formulator/db_manager.py treats DB_HOST starting with '/cloudsql/' as a unix socket path (e.g. pass host='/cloudsql/INSTANCE_CONNECTION_NAME' to psycopg2)."
# ...existing code...