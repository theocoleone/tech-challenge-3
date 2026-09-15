#!/usr/bin/env bash
# Deploy do dashboard estático do TC3 em S3 + CloudFront (OAC, bucket privado).
# Versiona o que o TC2 deixou como passo manual. Requer AWS CLI + profile com acesso.
set -euo pipefail

PROFILE="${AWS_PROFILE:-fiap-tech-challenge}"
REGION="us-east-1"
BUCKET="fiap-tc3-dashboard-286958704145"
DIST_ID="E2KTJLYG9E06DQ"                       # distribuição já provisionada
URL="https://dv2nlyojecknt.cloudfront.net"

# --- Publicação de rotina ---
python dashboard/gerar_dashboard.py
aws s3 cp dashboard/index.html "s3://$BUCKET/index.html" \
  --content-type "text/html; charset=utf-8" --profile "$PROFILE"
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" --profile "$PROFILE"
echo "Publicado: $URL"

# --- Provisionamento inicial (rodado uma única vez; documentado p/ reprodutibilidade) ---
# aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" --profile "$PROFILE"
# OAC=$(aws cloudfront create-origin-access-control --origin-access-control-config \
#   "Name=tc3-dashboard-oac,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" \
#   --profile "$PROFILE" --query 'OriginAccessControl.Id' --output text)
# # substituir ${OAC_ID} em dashboard/cloudfront-config.json e:
# aws cloudfront create-distribution --distribution-config file://dashboard/cloudfront-config.json --profile "$PROFILE"
# # aplicar bucket policy liberando Service=cloudfront.amazonaws.com com AWS:SourceArn = ARN da distribuição
