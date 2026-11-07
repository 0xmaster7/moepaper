"""Launch N spot workers with an EC2 Fleet that picks the spot pools least likely
to be interrupted, across ALL availability zones and many 32-vCPU types.

launch_spot.sh takes the first type that has capacity, always in your default
zone, so every worker can land in the same crowded pool and get reclaimed
together. EC2 Fleet with the "price-capacity-optimized" strategy ranks every
(type x zone) pool by spare capacity and price and launches into the best ones.

    python infra/launch_fleet.py extract 3             # launch 3 now
    python infra/launch_fleet.py extract 3 --keep      # ...and keep 3 alive until the stage is done
    python infra/launch_fleet.py safe 1 --keep

Workers are spread over different zones (one fleet call per worker, each
avoiding the zones already used), so one crowded pool can't reclaim them all.

--keep runs on your Mac (keep it awake: caffeinate -i python infra/launch_fleet.py ...).
Every 5 min it counts live workers and replaces reclaimed ones, but only while
there are unclaimed chunks left; it exits when every chunk of the stage is done.
Ctrl-C stops the watcher only - running workers carry on.

Needs the same env vars as launch_spot.sh: MOE_BUCKET, MOE_REPO,
MOE_INSTANCE_PROFILE. Uses the same bootstrap.sh, so the workers behave exactly
the same (lease chunks, upload results, terminate themselves).
All types below have 32 vCPU, so N workers = 32*N vCPU against your spot quota.
"""
import argparse
import base64
import os
import sys
import time
from collections import Counter
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
TYPES = ["c6a.8xlarge", "c7a.8xlarge", "c6i.8xlarge", "c7i.8xlarge", "c5a.8xlarge",
         "m6a.8xlarge", "m7a.8xlarge", "m6i.8xlarge", "m7i.8xlarge", "m5a.8xlarge"]
AMI_PARAM = "/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "safe", "idle"])
    ap.add_argument("count", type=int, nargs="?", default=1)
    ap.add_argument("--types", nargs="+", default=TYPES)
    ap.add_argument("--keep", action="store_true", help="watch and replace reclaimed workers until the stage is done")
    ap.add_argument("--every", type=int, default=300, help="seconds between checks in --keep mode")
    args = ap.parse_args()
    try:
        bucket, repo = os.environ["MOE_BUCKET"], os.environ["MOE_REPO"]
    except KeyError as e:
        sys.exit(f"export {e.args[0]} first")
    profile = os.environ.get("MOE_INSTANCE_PROFILE", "moe-worker")

    ec2 = boto3.client("ec2", region_name=REGION)
    ami = boto3.client("ssm", region_name=REGION).get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    ud = (Path(__file__).resolve().parent / "bootstrap.sh").read_text()
    ud = ud.replace("__BUCKET__", bucket).replace("__STAGE__", args.stage).replace("__REPO__", repo)

    lt_name = f"moe-{args.stage}"
    data = {
        "ImageId": ami,
        "IamInstanceProfile": {"Name": profile},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "BlockDeviceMappings": [{"DeviceName": "/dev/sda1",
                                 "Ebs": {"VolumeSize": 60, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "UserData": base64.b64encode(ud.encode()).decode(),
        "TagSpecifications": [{"ResourceType": "instance",
                               "Tags": [{"Key": "Name", "Value": f"moe-{args.stage}"},
                                        {"Key": "project", "Value": "moe-paper"}]}],
    }
    try:
        r = ec2.create_launch_template(LaunchTemplateName=lt_name, LaunchTemplateData=data)
        version = r["LaunchTemplate"]["LatestVersionNumber"]
    except ClientError as e:
        if "AlreadyExists" not in e.response["Error"]["Code"]:
            raise
        r = ec2.create_launch_template_version(LaunchTemplateName=lt_name, LaunchTemplateData=data)
        version = r["LaunchTemplateVersion"]["VersionNumber"]

    subnets = ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
    if not subnets:
        sys.exit("no default subnets found in us-east-1 (default VPC missing?)")
    zone = {s["SubnetId"]: s["AvailabilityZone"] for s in subnets}
    lt = {"LaunchTemplateName": lt_name, "Version": str(version)}

    def launch(n, avoid_zones=()):
        """Launch n workers, one fleet call each, each in a zone not used yet."""
        used, got = set(avoid_zones), 0
        for _ in range(n):
            subs = [s for s in subnets if s["AvailabilityZone"] not in used] or subnets
            overrides = [{"InstanceType": t, "SubnetId": s["SubnetId"]} for t in args.types for s in subs]
            try:
                r = ec2.create_fleet(
                    Type="instant",
                    SpotOptions={"AllocationStrategy": "price-capacity-optimized"},
                    LaunchTemplateConfigs=[{"LaunchTemplateSpecification": lt, "Overrides": overrides}],
                    TargetCapacitySpecification={"TotalTargetCapacity": 1, "DefaultTargetCapacityType": "spot"},
                )
            except ClientError as e:
                print("create_fleet failed:", e.response["Error"]["Code"], e.response["Error"]["Message"][:150])
                break
            ok = False
            for grp in r.get("Instances", []):
                o = grp["LaunchTemplateAndOverrides"]["Overrides"]
                for iid in grp["InstanceIds"]:
                    z = zone.get(o.get("SubnetId"), "?")
                    print(f"launched {iid}  {o['InstanceType']:12s} {z}", flush=True)
                    used.add(z)
                    got += 1
                    ok = True
            if not ok:
                errs = Counter(e.get("ErrorCode", "?") for e in r.get("Errors", []))
                print("no worker launched this round:", dict(errs), flush=True)
                if "MaxSpotInstanceCountExceeded" in errs:
                    print("  (spot quota full - reclaimed workers can count for ~10 min)")
                break
        return got

    if not args.keep:
        n = launch(args.count)
        print(f"{n}/{args.count} workers launched")
        return

    # ---- watcher: keep `count` workers alive while unclaimed chunks remain
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import s3io                                     # noqa: E402
    from common.config import load_config                       # noqa: E402
    cfg = load_config()
    s3cfg = cfg["s3"]
    src = s3cfg["manifests_prefix"] + "chunks/" if args.stage == "extract" else s3cfg["features_prefix"]
    out = s3cfg["features_prefix"] if args.stage == "extract" else s3cfg["safe_prefix"]
    lease_dir = s3cfg["leases_prefix"] + ("A/" if args.stage == "extract" else "B/")

    def names(prefix):
        return {k.rsplit("/", 1)[-1] for k in s3io.list_keys(bucket, prefix, REGION) if k.endswith(".parquet")}

    def live_workers():
        res = ec2.describe_instances(Filters=[
            {"Name": "tag:Name", "Values": [f"moe-{args.stage}"]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]}])
        inst = [i for r in res["Reservations"] for i in r["Instances"]]
        return len(inst), {i["Placement"]["AvailabilityZone"] for i in inst}

    print(f"watching stage {args.stage}: keep {args.count} workers until every chunk is done (Ctrl-C to stop watching)")
    while True:
        todo = names(src)
        done = names(out)
        leased = len(list(s3io.list_keys(bucket, lease_dir, REGION)))
        free = len(todo - done) - leased
        alive, zones = live_workers()
        print(f"{time.strftime('%H:%M')}  done {len(done & todo)}/{len(todo)}  leased {leased}  "
              f"workers {alive}", flush=True)
        if todo and not (todo - done):
            print(f"stage {args.stage} complete - workers terminate themselves; watcher exits")
            return
        need = min(args.count - alive, max(free, 0))
        if need > 0:
            launch(need, avoid_zones=zones)
        time.sleep(args.every)

if __name__ == "__main__":
    main()
