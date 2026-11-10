#!/bin/bash
# Build the dataset on an "idle" cloud box, next to the data, and upload the result.
# Stage A output (tokens + call graphs) is large; building inside us-east-1 avoids
# downloading it to the laptop. Only the finished dataset (a few GB) leaves AWS.
#
# From the Mac, after `python infra/launch_fleet.py idle 1 --types m6a.8xlarge ...`:
#   aws ssm send-command --region us-east-1 --instance-ids <id> --document-name AWS-RunShellScript \
#     --parameters 'commands=["bash /opt/moe/repo/pipeline_v2/infra/remote_build.sh <bucket>"],executionTimeout=["7200"]'
set -euo pipefail
BUCKET="${1:?usage: remote_build.sh <bucket>}"
cd /opt/moe/repo/pipeline_v2
git pull -q || true
source /opt/moe/venv/bin/activate
export MOE__s3__bucket="$BUCKET"
python build/build_dataset.py --graphs
python - "$BUCKET" <<'EOF'
import pathlib, sys
import boto3
bucket = sys.argv[1]
s3 = boto3.client("s3", region_name="us-east-1")
for p in sorted(pathlib.Path("data_v2").iterdir()):
    if p.is_file():
        s3.upload_file(str(p), bucket, f"v2/dataset/{p.name}")
        print(f"uploaded {p.name}  {p.stat().st_size / 2**20:,.0f} MB", flush=True)
EOF
echo BUILD_DONE
