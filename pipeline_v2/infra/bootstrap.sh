#!/bin/bash
# EC2 user-data for Ubuntu 22.04: install everything, then run one pipeline stage.
# launch_spot.sh fills in the three variables below. The instance terminates
# itself when its stage has no work left (or after a spot interruption).
#
#   BUCKET  your S3 bucket            STAGE  extract | safe | idle
#   REPO    git URL of your fork that contains pipeline_v2/
set -euxo pipefail
BUCKET="__BUCKET__"
STAGE="__STAGE__"
REPO="__REPO__"
R2_TAG="5.9.8"                       # pinned: r2_extract.py was written against 5.9.x JSON output

exec > >(tee /var/log/moe-bootstrap.log) 2>&1
# if setup fails, leave 15 min to look around via SSM, then terminate (no idle billing)
trap 'echo "BOOTSTRAP FAILED"; shutdown -h +15' ERR
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q python3-venv python3-pip git build-essential pkg-config

# radare2 from source at a fixed tag (~5 min)
if ! command -v r2 >/dev/null; then
  git clone --depth 1 --branch "$R2_TAG" https://github.com/radareorg/radare2 /opt/radare2
  (cd /opt/radare2 && ./configure --prefix=/usr && make -j"$(nproc)" && make install)
  r2 -v
fi

# work area on the instance store / root volume (only code, models, logs - never samples)
mkdir -p /opt/moe && cd /opt/moe
[ -d repo ] || git clone --depth 1 "$REPO" repo
python3 -m venv venv && source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r repo/pipeline_v2/requirements.txt
pip install -q torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -q git+https://github.com/FutureComputing4AI/EMBER2024.git
pip install -q signify==0.8.1          # thrember needs signify.authenticode.SignedPEFile (gone in 0.9)
python -c "import thrember; print(\"thrember dim\", thrember.PEFeatureExtractor().dim)"
if [ "$STAGE" = "safe" ]; then pip install -q tensorflow-cpu==2.15.1; fi

# model assets you uploaded once to s3://$BUCKET/assets/ (see RUNBOOK step 2)
python - <<EOF
import boto3, os
s3 = boto3.client("s3", region_name="us-east-1")
dst = {"safe.pb": "repo/safe_final/data/safe.pb",
       "word2id.json": "repo/safe_final/data/i2v/word2id.json"}
for k, p in dst.items():
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        s3.download_file("$BUCKET", "assets/" + k, p)
    except Exception as e:
        print("asset", k, "not downloaded:", e)
EOF

export MOE__s3__bucket="$BUCKET"
cd repo/pipeline_v2
trap - ERR
set +e                               # whatever the stage does, still upload the log and terminate
case "$STAGE" in
  extract) python extract/worker.py      2>&1 | tee /var/log/moe-worker.log ;;
  safe)    python embed/embed_safe.py    2>&1 | tee /var/log/moe-worker.log ;;
  idle)    echo "idle: connect with SSM Session Manager and run things by hand"; exit 0 ;;
esac

# upload the log, then terminate (InstanceInitiatedShutdownBehavior=terminate)
python - <<EOF
import boto3, socket
boto3.client("s3").upload_file("/var/log/moe-worker.log", "$BUCKET", f"v2/logs/$STAGE-{socket.gethostname()}.log")
EOF
shutdown -h now
