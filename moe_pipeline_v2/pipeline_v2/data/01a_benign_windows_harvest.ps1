<#
Harvest benign PE files from a clean Windows EC2 instance straight into S3.

Launch one small Windows instance per OS version you want (diversity helps):
  Windows Server 2016 / 2019 / 2022 / 2025 Base AMIs (us-east-1), t3.large is enough.
Attach an IAM instance profile that can s3:PutObject to your bucket (see infra/iam-policy.json).
In PowerShell (AWS Tools / AWS CLI are preinstalled on Amazon Windows AMIs):

  Set-ExecutionPolicy -Scope Process Bypass
  .\01a_benign_windows_harvest.ps1 -Bucket CHANGE-ME-moe-paper

Files are uploaded as benign/raw/windows/<sha256> (original bytes, never modified)
plus an index CSV benign/index/windows_<os>.csv with: sha256,group,origin,size,os
group = lower-case file name, so the same DLL from different Windows versions is
one group and never straddles train/test.

These files are Microsoft-copyrighted: keep them private, publish hashes only.
Terminate the instance when done.
#>
param(
  [Parameter(Mandatory = $true)][string]$Bucket,
  [string]$Region = "us-east-1",
  [int]$MaxMB = 64
)

$os = (Get-CimInstance Win32_OperatingSystem)
$osTag = ("win" + $os.Version).Replace(".", "_")
$roots = @("C:\Windows\System32", "C:\Windows\SysWOW64", "C:\Windows\Microsoft.NET",
           "C:\Program Files", "C:\Program Files (x86)", "C:\Windows\WinSxS")
# .mui (resource-only language DLLs) left out on purpose: no code, trivially "benign-looking"
$exts = @(".exe", ".dll", ".sys", ".ocx", ".cpl", ".scr", ".drv", ".efi")
$seen = @{}
$rows = New-Object System.Collections.Generic.List[string]
$rows.Add("sha256,group,origin,size,os")
$tmpDir = Join-Path $env:TEMP "harvest"; New-Item -ItemType Directory -Force $tmpDir | Out-Null

foreach ($r in $roots) {
  Get-ChildItem -Path $r -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $exts -contains $_.Extension.ToLower() -and $_.Length -gt 1024 -and $_.Length -lt ($MaxMB * 1MB) } |
    ForEach-Object {
      try {
        $fs = [System.IO.File]::OpenRead($_.FullName)
        $hdr = New-Object byte[] 2; [void]$fs.Read($hdr, 0, 2); $fs.Close()
        if ($hdr[0] -ne 0x4D -or $hdr[1] -ne 0x5A) { return }          # not MZ
        $sha = (Get-FileHash -Algorithm SHA256 -Path $_.FullName).Hash.ToLower()
        if ($seen.ContainsKey($sha)) { return }
        $seen[$sha] = 1
        aws s3 cp $_.FullName "s3://$Bucket/benign/raw/windows/$sha" --region $Region --only-show-errors
        $grp = "win:" + $_.Name.ToLower()
        $origin = $_.FullName.Replace(",", "_")
        $rows.Add("$sha,$grp,$origin,$($_.Length),$osTag")
        if ($seen.Count % 500 -eq 0) { Write-Host "$($seen.Count) files uploaded" }
      } catch { }
    }
}
$csv = Join-Path $tmpDir "windows_$osTag.csv"
$rows | Set-Content -Encoding ascii $csv
aws s3 cp $csv "s3://$Bucket/benign/index/windows_$osTag.csv" --region $Region
Write-Host "done: $($seen.Count) unique PE files from $osTag"
