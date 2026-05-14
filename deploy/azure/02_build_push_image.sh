#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT_DIR/deploy/azure/.env"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  # shellcheck disable=SC1091
  source "$ROOT_DIR/deploy/azure/env.example"
fi

cd "$ROOT_DIR"

ACR_LOGIN_SERVER="$(az acr show \
  --name "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query loginServer \
  --output tsv)"

TAG="${1:-latest}"
IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$TAG"
LATEST="$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"

echo "Logging in to ACR: $ACR_NAME"
az acr login --name "$ACR_NAME"

echo "Building and pushing linux/amd64 image for Azure Container Apps: $IMAGE"
docker buildx build \
  --platform linux/amd64 \
  -f deploy/Dockerfile \
  -t "$IMAGE" \
  -t "$LATEST" \
  --push \
  .

echo
echo "✅ Image pushed: $IMAGE"
echo "Next: run deploy/azure/03_create_or_update_app.sh $TAG"
