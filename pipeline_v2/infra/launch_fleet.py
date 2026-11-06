"""Launch N spot workers with an EC2 Fleet that picks the spot pools least likely
to be interrupted, across ALL availability zones and many 32-vCPU types.

launch_spot.sh takes the first type that has capacity, always in your default
zone, so every worker can land in the same crowded pool and get reclaimed
together. EC2 Fleet with the "price-capacity-optimized" strategy ranks every
(type x zone) pool by spare capacity and price and launches into the best ones.

    python infra/launch_fleet.py extract 3
    python infra/launch_fleet.py safe 1

Needs the same env vars as launch_spot.sh: MOE_BUCKET, MOE_REPO,
MOE_INSTANCE_PROFILE. Uses the same bootstrap.sh, so the workers behave exactly
the same (lease chunks, upload results, terminate themselves).
All types below have 32 vCPU, so N workers = 32*N vCPU against your spot quota.
"""
import argparse
import base64
import os
import sys
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
    overrides = [{"InstanceType": t, "SubnetId": s["SubnetId"]} for t in args.types for s in subnets]
    print(f"{len(args.types)} types x {len(subnets)} zones = {len(overrides)} spot pools; asking for {args.count}")

    r = ec2.create_fleet(
        Type="instant",
        SpotOptions={"AllocationStrategy": "price-capacity-optimized"},
        LaunchTemplateConfigs=[{"LaunchTemplateSpecification": {"LaunchTemplateName": lt_name,
                                                                "Version": str(version)},
                                "Overrides": overrides}],
        TargetCapacitySpecification={"TotalTargetCapacity": args.count, "DefaultTargetCapacityType": "spot"},
    )
    zone = {s["SubnetId"]: s["AvailabilityZone"] for s in subnets}
    n = 0
    for grp in r.get("Instances", []):
        o = grp["LaunchTemplateAndOverrides"]["Overrides"]
        for iid in grp["InstanceIds"]:
            n += 1
            print(f"launched {iid}  {o['InstanceType']:12s} {zone.get(o.get('SubnetId'), '?')}")
    errs = Counter(e.get("ErrorCode", "?") for e in r.get("Errors", []))
    if errs and n < args.count:
        print("pools that could not launch (normal, AWS tries the next):", dict(errs))
    print(f"{n}/{args.count} workers launched")
    if n < args.count:
        print("short: if the errors say MaxSpotInstanceCountExceeded, wait ~10 min (terminated "
              "workers still count) and run again with the missing number")


if __name__ == "__main__":
    main()
