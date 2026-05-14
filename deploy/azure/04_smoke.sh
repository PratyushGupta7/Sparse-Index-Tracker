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

API_FQDN="$(az containerapp show \
  --name "$CONTAINER_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)"

API_URL="https://$API_FQDN"

echo "API URL: $API_URL"
echo

set -x
curl --fail --show-error "$API_URL/api/v1/health"
curl --fail --show-error "$API_URL/api/v1/portfolio"
curl --fail --show-error "$API_URL/api/v1/methods/comparison" >/dev/null
curl --fail --show-error "$API_URL/api/v1/markets/cross-index" >/dev/null
set +x

echo
echo "✅ Smoke checks passed."
