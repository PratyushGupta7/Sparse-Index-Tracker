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

TAG="${1:-latest}"

ACR_LOGIN_SERVER="$(az acr show \
  --name "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query loginServer \
  --output tsv)"

ACR_USERNAME="$(az acr credential show \
  --name "$ACR_NAME" \
  --query username \
  --output tsv)"

ACR_PASSWORD="$(az acr credential show \
  --name "$ACR_NAME" \
  --query "passwords[0].value" \
  --output tsv)"

REDIS_HOST="$(az redis show \
  --name "$REDIS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query hostName \
  --output tsv)"

REDIS_KEY="$(az redis list-keys \
  --name "$REDIS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query primaryKey \
  --output tsv)"

APPINSIGHTS_CONNECTION_STRING="$(az monitor app-insights component show \
  --app "$APP_INSIGHTS" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString \
  --output tsv)"

IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$TAG"
REDIS_URL="rediss://:${REDIS_KEY}@${REDIS_HOST}:6380/0"

COMMON_ENV_VARS=(
  "SIT_ENV=prod"
  "SIT_APP_VERSION=1.0.0"
  "SIT_GIT_SHA=$TAG"
  "SIT_DATA_DIR=/app/data"
  "SIT_BENCHMARKS_DIR=/app/benchmarks/_results"
  "SIT_ALLOWED_ORIGINS=$SIT_ALLOWED_ORIGINS"
  "SIT_REDIS_URL=secretref:redis-url"
  "SIT_REDIS_TTL_S=300"
  "SIT_RATE_LIMITS_ENABLED=true"
  "SIT_RATE_LIMIT_DEFAULT=120/minute"
  "SIT_RATE_LIMIT_INVEST=120/minute"
  "SIT_RATE_LIMIT_INVEST_LIVE=30/minute"
  "SIT_LIVE_UNIVERSE_MAX_TICKERS=120"
  "SIT_LIVE_UNIVERSE_CAP_THRESHOLD=750"
  "SIT_APPLICATIONINSIGHTS_CONNECTION_STRING=secretref:appinsights-connection-string"
)

if az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "Updating existing Container App: $CONTAINER_APP"
  az containerapp secret set \
    --name "$CONTAINER_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --secrets \
      "redis-url=$REDIS_URL" \
      "appinsights-connection-string=$APPINSIGHTS_CONNECTION_STRING" \
    --output none

  az containerapp update \
    --name "$CONTAINER_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --image "$IMAGE" \
    --set-env-vars "${COMMON_ENV_VARS[@]}" \
    --output table
else
  echo "Creating Container App: $CONTAINER_APP"
  az containerapp create \
    --name "$CONTAINER_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ACA_ENV" \
    --image "$IMAGE" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --target-port 8000 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 5 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --secrets \
      "redis-url=$REDIS_URL" \
      "appinsights-connection-string=$APPINSIGHTS_CONNECTION_STRING" \
    --env-vars "${COMMON_ENV_VARS[@]}" \
    --output table
fi

API_FQDN="$(az containerapp show \
  --name "$CONTAINER_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)"

echo
echo "✅ Container App ready."
echo "API URL: https://$API_FQDN"
echo
echo "Smoke test:"
echo "curl https://$API_FQDN/api/v1/health"
