#!/bin/bash
# scripts/smoke_test.sh
# ============================================================
# Post-Deployment Smoke Tests
# Verifies the inference service is healthy and making predictions.
#
# Usage:
#   bash scripts/smoke_test.sh [BASE_URL] [IMAGE_PATH]
#   bash scripts/smoke_test.sh https://api.petadoption.example.com
#   bash scripts/smoke_test.sh http://127.0.0.1:8080 tests/assets/sample.jpg
# ============================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_URL="${1:-http://localhost:8080}"
TEST_IMAGE_INPUT="${2:-}"
MAX_RETRIES=10
RETRY_INTERVAL=6    # seconds
PASS_COUNT=0
FAIL_COUNT=0

# ANSI colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ─── Helpers ─────────────────────────────────────────────────────────────────
pass() { echo -e "${GREEN}✓ PASS${NC} $1"; ((PASS_COUNT++)); }
fail() { echo -e "${RED}✗ FAIL${NC} $1"; ((FAIL_COUNT++)); }
info() { echo -e "${BLUE}→${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC}  $1"; }

header() {
  echo ""
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
}

# ─── Wait for Service ─────────────────────────────────────────────────────────
header "Waiting for Service at ${BASE_URL}"

for i in $(seq 1 $MAX_RETRIES); do
  if curl -sf --max-time 5 "${BASE_URL}/health" > /dev/null 2>&1; then
    pass "Service is reachable (attempt $i)"
    break
  fi
  if [ $i -eq $MAX_RETRIES ]; then
    fail "Service unreachable after ${MAX_RETRIES} attempts"
    exit 1
  fi
  warn "Attempt $i/${MAX_RETRIES} failed, retrying in ${RETRY_INTERVAL}s..."
  sleep $RETRY_INTERVAL
done

# ─── Test 1: Health Check ─────────────────────────────────────────────────────
header "Test 1: Health Endpoint"

HEALTH_RESPONSE=$(curl -sf --max-time 10 "${BASE_URL}/health")
HTTP_CODE=$(curl -o /dev/null -sw "%{http_code}" --max-time 10 "${BASE_URL}/health")

info "Response: ${HEALTH_RESPONSE}"
info "HTTP Code: ${HTTP_CODE}"

if [ "${HTTP_CODE}" = "200" ]; then
  pass "Health endpoint returns 200"
else
  fail "Health endpoint returned ${HTTP_CODE} (expected 200)"
fi

# Check model_loaded field
MODEL_LOADED=$(echo "${HEALTH_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded',''))" 2>/dev/null || echo "")
if [ "${MODEL_LOADED}" = "True" ] || [ "${MODEL_LOADED}" = "true" ]; then
  pass "Model is loaded"
else
  warn "Model may not be loaded yet (model_loaded=${MODEL_LOADED})"
fi

# Check status field
STATUS=$(echo "${HEALTH_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || echo "")
if [ "${STATUS}" = "healthy" ]; then
  pass "Service status is 'healthy'"
else
  fail "Service status is '${STATUS}' (expected 'healthy')"
fi

# ─── Test 1b: Readiness Check ─────────────────────────────────────────────────
header "Test 1b: Readiness Endpoint"

READY_CODE=$(curl -o /dev/null -sw "%{http_code}" --max-time 10 "${BASE_URL}/ready" || echo "000")
if [ "${READY_CODE}" = "200" ]; then
  pass "Readiness endpoint returns 200"
else
  fail "Readiness endpoint returned ${READY_CODE}"
fi

# ─── Test 2: Model Info ───────────────────────────────────────────────────────
header "Test 2: Model Info Endpoint"

INFO_CODE=$(curl -o /dev/null -sw "%{http_code}" --max-time 10 "${BASE_URL}/model/info")
if [ "${INFO_CODE}" = "200" ]; then
  pass "Model info endpoint returns 200"
else
  fail "Model info endpoint returned ${INFO_CODE}"
fi

# ─── Test 3: Prediction with Synthetic Image ─────────────────────────────────
header "Test 3: Prediction Endpoint"

GENERATED_IMAGE=0
if [ -n "${TEST_IMAGE_INPUT}" ]; then
  TEST_IMAGE="${TEST_IMAGE_INPUT}"
  if [ -f "${TEST_IMAGE}" ]; then
    info "Using provided test image: ${TEST_IMAGE}"
  else
    fail "Provided test image does not exist: ${TEST_IMAGE}"
    TEST_IMAGE=""
  fi
else
  # Create a minimal valid JPEG image using Python
  TEST_IMAGE="/tmp/smoke_test_image.jpg"
  python3 -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), 'RGB')
img.save('${TEST_IMAGE}', 'JPEG')
print('Synthetic test image created')
" 2>/dev/null || {
    # Fallback: create minimal JPEG without PIL
    python3 -c "
import struct, io
# Minimal 1x1 white JPEG
data = bytes([
  0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
  0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
  0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
  0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
  0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
  0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
  0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
  0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
  0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
  0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
  0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
  0x09,0x0A,0x0B,0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,0x00,0x3F,
  0x00,0xFB,0xD3,0xFF,0xD9
])
open('${TEST_IMAGE}', 'wb').write(data)
" 2>/dev/null
  }
  GENERATED_IMAGE=1
fi

if [ -n "${TEST_IMAGE}" ] && [ -f "${TEST_IMAGE}" ]; then
  PREDICT_RESPONSE=$(curl -sf --max-time 30 \
    -X POST \
    -F "file=@${TEST_IMAGE};type=image/jpeg" \
    "${BASE_URL}/predict" 2>&1 || echo "CURL_FAILED")

  if [ "${PREDICT_RESPONSE}" != "CURL_FAILED" ] && [ -n "${PREDICT_RESPONSE}" ]; then
    pass "Prediction endpoint returned a response"
    info "Response: ${PREDICT_RESPONSE}"

    # Validate label
    LABEL=$(echo "${PREDICT_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('label',''))" 2>/dev/null || echo "")
    if [ "${LABEL}" = "cat" ] || [ "${LABEL}" = "dog" ]; then
      pass "Prediction label is valid: '${LABEL}'"
    else
      fail "Prediction label is invalid: '${LABEL}'"
    fi

    # Validate confidence
    CONFIDENCE=$(echo "${PREDICT_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d.get('confidence',0); print('valid' if 0<=c<=1 else 'invalid')" 2>/dev/null || echo "invalid")
    if [ "${CONFIDENCE}" = "valid" ]; then
      pass "Confidence score is in valid range [0, 1]"
    else
      fail "Confidence score out of range"
    fi

    # Validate probabilities exist
    HAS_PROBS=$(echo "${PREDICT_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); p=d.get('probabilities',{}); print('yes' if 'cat' in p and 'dog' in p else 'no')" 2>/dev/null || echo "no")
    if [ "${HAS_PROBS}" = "yes" ]; then
      pass "Response contains cat and dog probabilities"
    else
      fail "Missing probabilities in response"
    fi
  else
    fail "Prediction endpoint failed or returned empty response"
  fi

  if [ "${GENERATED_IMAGE}" = "1" ]; then
    rm -f "${TEST_IMAGE}"
  fi
else
  fail "Could not create test image"
fi

# ─── Test 4: Metrics Endpoint ─────────────────────────────────────────────────
header "Test 4: Prometheus Metrics Endpoint"

METRICS_CODE=$(curl -o /dev/null -sw "%{http_code}" --max-time 10 "${BASE_URL}/metrics")
if [ "${METRICS_CODE}" = "200" ]; then
  pass "Metrics endpoint returns 200"
else
  fail "Metrics endpoint returned ${METRICS_CODE}"
fi

METRICS_BODY=$(curl -sf --max-time 10 "${BASE_URL}/metrics" || echo "")
if echo "${METRICS_BODY}" | grep -q "http_requests_total"; then
  pass "Metrics contain http_requests_total counter"
else
  fail "Missing http_requests_total metric"
fi

# ─── Test 5: Invalid Input Handling ──────────────────────────────────────────
header "Test 5: Error Handling"

# Send non-image file
INVALID_RESPONSE_CODE=$(curl -o /dev/null -sw "%{http_code}" --max-time 10 \
  -X POST \
  -F "file=@/etc/hostname;type=text/plain" \
  "${BASE_URL}/predict" 2>/dev/null || echo "000")

if [ "${INVALID_RESPONSE_CODE}" = "400" ]; then
  pass "Non-image upload correctly returns 400"
else
  warn "Non-image upload returned ${INVALID_RESPONSE_CODE} (expected 400)"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
header "Smoke Test Summary"

TOTAL=$((PASS_COUNT + FAIL_COUNT))
echo ""
echo -e "  Tests passed: ${GREEN}${PASS_COUNT}/${TOTAL}${NC}"
echo -e "  Tests failed: ${RED}${FAIL_COUNT}/${TOTAL}${NC}"
echo ""

if [ ${FAIL_COUNT} -gt 0 ]; then
  echo -e "${RED}❌ SMOKE TESTS FAILED — deployment validation unsuccessful${NC}"
  echo -e "   Check the inference service logs: kubectl logs -l app=cats-dogs-classifier -n mlops"
  exit 1
else
  echo -e "${GREEN}✅ ALL SMOKE TESTS PASSED — deployment validated successfully${NC}"
  exit 0
fi
