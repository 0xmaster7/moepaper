<#
Harvest benign PE files from a clean Windows EC2 instance into S3.

Run it through SSM Run Command from your Mac (no RDP, no open ports, survives
disconnects) - see RUNBOOK / the instructions you were given:
  aws ssm send-command --document-name AWS-RunPowerShellScript ...

What it does
  1. Installs AWS CLI v2 if the AMI doesn't have it.
  2. Walks System32, SysWOW64, Microsoft.NET, Program Files, WinSxS for
     .exe/.dll/.sys/.ocx/.cpl/.scr/.drv/.efi files (1 KB .. MaxMB), keeps only
     files that start with "MZ", de-duplicates by SHA-256.
  3. Copies each unique file to C:\harvest\files\<sha256> (local staging), then
     uploads everything with ONE "aws s3 sync" (fast; one aws.exe call per file
     would take hours).
  4. Uploads the index benign/index/windows_<os>.csv : sha256,group,origin,size,os
     group = "win:" + lower-case file name, so the same DLL from different Windows
     versions is one group and never straddles train/test.

Give the instance a 60 GB root volume. Microsoft files: keep private, publish hashes only.
Terminate the instance when done.
#>
param(
  [Parameter(Mandatory = $true)][string]$Bucket,
  [string]$Region = "us-east-1",
  [int]$MaxMB = 64
)
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# ---- AWS CLI
$aws = (Get-Command aws.exe -ErrorAction SilentlyContinue).Source
if (-not $aws) {
  Write-Output "installing AWS CLI v2"
  Start-Process msiexec.exe -Wait -ArgumentList "/i https://awscli.amazonaws.com/AWSCLIV2.msi /qn /norestart"
  $aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
}
& $aws --version

$os = Get-CimInstance Win32_OperatingSystem
$osTag = ("win" + $os.Version).Replace(".", "_")
$stage = "C:\harvest\files"
New-Item -ItemType Directory -Force $stage | Out-Null
$roots = @("C:\Windows\System32", "C:\Windows\SysWOW64", "C:\Windows\Microsoft.NET",
           "C:\Program Files", "C:\Program Files (x86)", "C:\Windows\WinSxS")
# .mui (resource-only language DLLs) left out on purpose: no code
$exts = @(".exe", ".dll", ".sys", ".ocx", ".cpl", ".scr", ".drv", ".efi")
$seen = @{}
$rows = New-Object System.Collections.Generic.List[string]
$rows.Add("sha256,group,origin,size,os")
$scanned = 0

foreach ($r in $roots) {
  Write-Output "scanning $r"
  Get-ChildItem -Path $r -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { ($exts -contains $_.Extension.ToLower()) -and $_.Length -gt 1024 -and $_.Length -lt ($MaxMB * 1MB) } |
    ForEach-Object {
      $scanned++
      try {
        $fs = [System.IO.File]::Open($_.FullName, 'Open', 'Read', 'ReadWrite')
        $hdr = New-Object byte[] 2
        [void]$fs.Read($hdr, 0, 2)
        $fs.Close()
        if ($hdr[0] -ne 0x4D -or $hdr[1] -ne 0x5A) { return }
        $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLower()
        if ($seen.ContainsKey($sha)) { return }
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $stage $sha) -ErrorAction Stop
        $seen[$sha] = 1
        $grp = "win:" + $_.Name.ToLower().Replace(",", "_")
        $origin = $_.FullName.Replace(",", "_")
        $rows.Add("$sha,$grp,$origin,$($_.Length),$osTag")
        if ($seen.Count % 2000 -eq 0) { Write-Output "$($seen.Count) unique PE files staged ($scanned scanned)" }
      } catch { }
    }
}
Write-Output "staged $($seen.Count) unique PE files from $scanned candidates; uploading"

& $aws s3 sync $stage "s3://$Bucket/benign/raw/windows/" --region $Region --only-show-errors
$csv = "C:\harvest\windows_$osTag.csv"
$rows | Set-Content -Encoding ascii $csv
& $aws s3 cp $csv "s3://$Bucket/benign/index/windows_$osTag.csv" --region $Region --only-show-errors
Write-Output "DONE: $($seen.Count) unique PE files from $osTag uploaded and indexed"
