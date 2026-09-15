#!/usr/bin/env bash
# Publica o dashboard estático em S3 + CloudFront (bucket privado, OAC).
# Requer AWS CLI e um profile com acesso à conta. Rodar da raiz do repo.
set -euo pipefail

PROFILE="${AWS_PROFILE:-fiap-tech-challenge}"
REGION="us-east-1"
BUCKET="fiap-tc3-dashboard-286958704145"
DIST_ID="E2KTJLYG9E06DQ"
URL="https://dv2nlyojecknt.cloudfront.net"

python dashboard/gerar_dashboard.py
aws s3 cp dashboard/index.html "s3://$BUCKET/index.html" \
  --content-type "text/html; charset=utf-8" \
  --cache-control "max-age=300" \
  --profile "$PROFILE"
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" --profile "$PROFILE"
echo "Publicado: $URL"

# --- Provisionamento inicial (feito uma vez; mantido para reprodutibilidade) ---
# aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" --profile "$PROFILE"
# aws s3api put-public-access-block --bucket "$BUCKET" --profile "$PROFILE" \
#   --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
# OAC=$(aws cloudfront create-origin-access-control --origin-access-control-config \
#   "Name=tc3-dashboard-oac,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" \
#   --profile "$PROFILE" --query 'OriginAccessControl.Id' --output text)
# # colocar $OAC em OriginAccessControlId de dashboard/cloudfront-config.json, então:
# aws cloudfront create-distribution --distribution-config file://dashboard/cloudfront-config.json --profile "$PROFILE"
# # com o ARN da distribuição em dashboard/bucket-policy.json:
# aws s3api put-bucket-policy --bucket "$BUCKET" --policy file://dashboard/bucket-policy.json --profile "$PROFILE"
