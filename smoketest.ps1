function Stop-Smoketest([string]$Message) {
    [Console]::Error.WriteLine("smoketest.ps1: $Message")
    exit 1
}

Set-Location -LiteralPath $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Stop-Smoketest "git chybi; nainstaluj git a repozitar naklonuj pres git clone."
}
$gitError = (git rev-parse --git-dir 2>&1)
if ($LASTEXITCODE -ne 0) {
    Stop-Smoketest "git adresar nepovazuje za repozitar ($gitError); naklonuj repo pres git clone, archiv ZIP nestaci, protoze run manifest potrebuje commit."
}

$engine = $env:CONTAINER_ENGINE
if (-not $engine) {
    if (Get-Command podman -ErrorAction SilentlyContinue) {
        $engine = "podman"
    } elseif (Get-Command docker -ErrorAction SilentlyContinue) {
        $engine = "docker"
    } else {
        Stop-Smoketest "chybi podman i docker; nainstaluj jeden z nich nebo nastav CONTAINER_ENGINE."
    }
}

if (-not (Get-Command $engine -ErrorAction SilentlyContinue)) {
    Stop-Smoketest "kontejnerovy engine $engine nebyl nalezen; nainstaluj ho nebo uprav CONTAINER_ENGINE."
}

$commit = (git rev-parse HEAD)
if ($LASTEXITCODE -ne 0) {
    Stop-Smoketest "git rev-parse HEAD selhal."
}
$status = (git status --porcelain)
if ($LASTEXITCODE -ne 0) {
    Stop-Smoketest "git status selhal."
}
$dirty = "false"
if ($status) {
    $dirty = "true"
}

& $engine build -f Containerfile --build-arg "MEDIPARSE_COMMIT=$commit" --build-arg "MEDIPARSE_DIRTY=$dirty" -t mediparse .
if ($LASTEXITCODE -ne 0) {
    Stop-Smoketest "sestaveni image selhalo."
}

$root = (Get-Location).Path
New-Item -ItemType Directory -Force -Path (Join-Path $root "build"), (Join-Path $root "logs") | Out-Null

& $engine run --rm --network none -v "${root}\build:/app/build" -v "${root}\logs:/app/logs" mediparse
if ($LASTEXITCODE -ne 0) {
    Stop-Smoketest "beh v kontejneru selhal."
}
