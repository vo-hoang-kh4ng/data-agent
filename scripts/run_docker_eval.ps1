# Run Dockerized Outer Eval
# This script builds the Docker image and runs the evaluation container,
# passing the .env file so the Groq API key is available inside.

Write-Host "Building Docker image lambda_eval..." -ForegroundColor Cyan
docker build -t lambda_eval -f Dockerfile.eval .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nRunning Docker container lambda_eval..." -ForegroundColor Cyan
# Run the container, mount the current directory so reports are saved back to the host
docker run --rm --env-file .env -v "${PWD}:/app" lambda_eval
