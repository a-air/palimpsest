# Publish to GitHub in one command (create the repo on the web first)
param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [string]$User = "a-air",
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$url = "git@github.com:$User/$Name.git"
Write-Host "target: $url"

# 1. preflight
& "$PSScriptRoot\preflight.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "[X] preflight failed"; exit 1 }

# 2. remote
$remotes = @(git remote)
if ($remotes -contains 'origin') {
    Write-Host "origin exists, updating url"
    git remote set-url origin $url
} else {
    git remote add origin $url
}

# 3. branch name
git branch -M main

if ($DryRun) {
    Write-Host "[dry-run] would push to $url"
    git remote -v
    exit 0
}

# 4. push
git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "[X] push failed. repo created? name correct? ssh key bound?" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] pushed: https://github.com/$User/$Name" -ForegroundColor Green
