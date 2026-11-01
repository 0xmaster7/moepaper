#!/bin/bash
# Launch N one-time spot instances that run one pipeline stage and terminate themselves.
#
#   ./infra/launch_spot.sh extract 3          # 3 workers for Stage A
#   ./infra/launch_spot.sh safe 1             # 1 worker for SAFE embeddings
#   ./infra/launch_spot.sh idle 1             # an empty box for manual work (meta.db, benign collection)
#
# Requires: AWS CLI v2 configured for your account, region us-east-1, and the
# instance profile created in RUNBOOK step 1. Interrupted? Run the same command
# again - finished chunks are skipped, unfinished leases expire and get retried.
set -euo pipefail
STAGE="${1:?stage: extract|safe|idle}"
COUNT="${2:-1}"
BUCKET="${MOE_BUCKET:?export MOE_BUCKET=your-bucket}"
REPO="${MOE_REPO:?export MOE_REPO=https://github.com/<you>/moepaper.git}"
PROFILE="${MOE_INSTANCE_PROFILE:-moe-worker}"
REGION=us-east-1
# CPU-heavy, memory-light work: compute-optimised AMD/Intel, 32 vCPU. Several
# types widen the spot pool; the first one with capacity is used.
TYPES=(${MOE_TYPES:-c6a.8xlarge c7a.8xlarge c6i.8xlarge c5a.8xlarge})

AMI=$(aws ssm get-parameter --region $REGION \
  --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
  --query Parameter.Value --output text)
DIR="$(cd "$(dirname "$0")" && pwd)"
USERDATA=$(sed -e "s|__BUCKET__|$BUCKET|" -e "s|__STAGE__|$STAGE|" -e "s|__REPO__|$REPO|" "$DIR/bootstrap.sh")

for i in $(seq 1 "$COUNT"); do
  for T in "${TYPES[@]}"; do
    if aws ec2 run-instances --region $REGION --image-id "$AMI" --instance-type "$T" \
        --iam-instance-profile Name="$PROFILE" \
        --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=60,VolumeType=gp3,DeleteOnTermination=true}' \
        --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=moe-$STAGE-$i},{Key=project,Value=moe-paper}]" \
        --user-data "$USERDATA" --query 'Instances[0].InstanceId' --output text; then
      echo "launched $T for $STAGE"
      break
    fi
    echo "no spot capacity for $T, trying next type"
  done
done
echo "watch: python tools/progress.py   |   list: aws ec2 describe-instances --filters Name=tag:project,Values=moe-paper Name=instance-state-name,Values=running --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[0].Value]' --output table"
