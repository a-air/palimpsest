# Preflight: tests + privacy scan + tracked scope + clean tree
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Fail($msg) { Write-Host "[X] $msg" -ForegroundColor Red; exit 1 }
function Ok($msg)   { Write-Host "[OK] $msg" -ForegroundColor Green }

Write-Host "== 1/4 tests ==" -ForegroundColor Cyan
python tests/test_version_chain.py | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "version-chain tests failed" }
python tests/test_cli_smoke.py | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "cli smoke tests failed" }
Ok "both suites passed"

Write-Host "== 2/4 privacy scan ==" -ForegroundColor Cyan
$patterns = '??|??|??|??|??|??|??|??|??|??|??|??'
# 私人词表放在单独文件，避免本脚本自身被扫到
if (Test-Path "scripts/privacy_patterns.txt") {
    $patterns = (Get-Content "scripts/privacy_patterns.txt" -Encoding UTF8 | Where-Object { $_ -and -not $_.StartsWith('#') }) -join '|'
}
$hits = Get-ChildItem -Recurse -File -Include *.md,*.json,*.py,*.yml |
        Where-Object { $_.FullName -notmatch '\\archive\\|\\state\\|\\refined\\|\\.git\\' } |
        Select-String -Pattern $patterns -ErrorAction SilentlyContinue
if ($hits) { $hits | ForEach-Object { Write-Host "  $($_.Filename):$($_.LineNumber)" }; Fail "private content found" }
Ok "no private content"

Write-Host "== 3/4 tracked scope ==" -ForegroundColor Cyan
$tracked = git ls-files
foreach ($p in 'archive/', 'state/', 'refined/') {
    if ($tracked -match "^$p") { Fail "$p is tracked by git" }
}
Ok "runtime dirs not tracked"

Write-Host "== 4/4 working tree ==" -ForegroundColor Cyan
$dirty = git status --porcelain
if ($dirty) { Write-Host $dirty; Fail "working tree is dirty, commit first" }
Ok "clean"
git log --oneline -1

Write-Host ""
Write-Host "ready to publish." -ForegroundColor Green
