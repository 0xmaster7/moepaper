#!/bin/bash
# All experiments on the idle box (data_v2/ is already there; graphs.npz needs ~10 GB RAM,
# more than a laptop has). results/ and the log are mirrored to s3://<bucket>/v2/results/
# every 10 min, so a spot interruption loses at most one run: on a new box this script
# first downloads results/ and run_ablations.py skips everything already finished.
#
# Start detached (survives the SSM command ending):
#   systemd-run --unit=moe-exp bash -c 'bash /opt/moe/repo/pipeline_v2/infra/remote_experiments.sh <bucket> > /var/log/moe-exp.log 2>&1'
set -uo pipefail
BUCKET="${1:?usage: remote_experiments.sh <bucket>}"
cd /opt/moe/repo/pipeline_v2
git pull -q || true
source /opt/moe/venv/bin/activate
export MOE__s3__bucket="$BUCKET"

[ -f data_v2/dataset.npz ] || python tools/s3sync.py down data_v2 v2/dataset/     # fresh box
python tools/s3sync.py down results v2/results/                                     # resume
mkdir -p results
push() { cp /var/log/moe-exp.log results/experiments.log 2>/dev/null; python tools/s3sync.py up results v2/results/; }
( while true; do sleep 600; push; done ) &
SYNC=$!

python ablations/run_ablations.py --sweep-dropout || { push; echo EXPERIMENTS_FAILED; exit 1; }
BEST=$(python -c "import json; print(json.load(open('results/dropout_sweep_val.json'))['best'])")
echo "expert dropout chosen on VAL: $BEST"
export MOE__train__expert_dropout="$BEST"
python ablations/run_ablations.py || { push; echo EXPERIMENTS_FAILED; exit 1; }
python report/make_tables.py
kill $SYNC 2>/dev/null
echo EXPERIMENTS_DONE
push
