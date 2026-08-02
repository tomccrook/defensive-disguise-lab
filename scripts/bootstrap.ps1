param([switch]$VerifyGpu)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'Install Git for Windows from https://git-scm.com/download/win, then rerun.'
  }
  winget install --id Git.Git --exact --source winget
  throw 'Complete any Git installer/UAC prompt, reopen PowerShell, then rerun.'
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'Install uv 0.12.23 using https://docs.astral.sh/uv/getting-started/installation/.'
  }
  winget install --id astral-sh.uv --exact --version 0.12.23 --source winget
  throw 'Complete the installer, reopen PowerShell so PATH is refreshed, then rerun.'
}
if (((uv --version) -split '\s+')[1] -ne '0.12.23') { throw 'Use pinned uv 0.12.23 before syncing locks.' }
$nodeVersion = '24.21.0'
$toolsDir = Join-Path $PWD '.tools'
$nodeDir = Join-Path $toolsDir "node-v$nodeVersion-win-x64"
New-Item -ItemType Directory -Force $toolsDir | Out-Null
if (-not (Test-Path (Join-Path $nodeDir 'node.exe'))) {
  $archiveName = "node-v$nodeVersion-win-x64.zip"
  $archive = Join-Path $toolsDir $archiveName
  $checksums = Join-Path $toolsDir 'node-SHASUMS256.txt'
  Invoke-WebRequest "https://nodejs.org/dist/v$nodeVersion/$archiveName" -OutFile $archive
  Invoke-WebRequest "https://nodejs.org/dist/v$nodeVersion/SHASUMS256.txt" -OutFile $checksums
  $line = Get-Content $checksums | Where-Object { $_.EndsWith("  $archiveName") }
  if (-not $line) { throw 'Official checksum unavailable.' }
  $expected = ($line -split '\s+')[0]
  if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLower() -ne $expected) {
    throw 'Node checksum mismatch; do not run archive.'
  }
  Expand-Archive $archive -DestinationPath $toolsDir
}
$env:PATH = "$nodeDir;$env:PATH"
$env:UV_CACHE_DIR = Join-Path $PWD '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $toolsDir 'python'
$env:npm_config_cache = Join-Path $PWD '.cache/npm'
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $PWD '.cache/playwright'
uv python install 3.14.8 --no-bin
if ($LASTEXITCODE -ne 0) { throw 'Pinned Python installation failed.' }
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw 'Locked dependency install failed; preserve compatibility error.' }
npm.cmd ci --prefix apps/web
if ($LASTEXITCODE -ne 0) { throw 'Frontend lock install failed.' }
git --version
uv --version
node --version
npm.cmd --version
uv run --locked disguise-lab host --output artifacts/host-windows-cpu.json
if ($LASTEXITCODE -ne 0) { throw 'CPU workload failed; GPU training is not ready.' }
if ($VerifyGpu) {
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi }
  uv run --locked disguise-lab host --device cuda --output artifacts/host-windows-cuda.json
  if ($LASTEXITCODE -ne 0) {
    throw 'CUDA workload unverified. Check NVIDIA driver and official PyTorch wheel selector; retain CPU mode.'
  }
}
Write-Output 'Setup complete. Run uv run disguise-lab synthetic; then start API and frontend in separate terminals.'
