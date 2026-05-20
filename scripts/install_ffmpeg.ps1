param(
    [string]$Url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
    [string]$Archive = "ffmpeg.zip",
    [string]$InstallDir = "ffmpeg_tool"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $InstallDir -PathType Container)) {
    New-Item -ItemType Directory -Path $InstallDir | Out-Null
}

Write-Host "Downloading FFmpeg..."
Invoke-WebRequest -Uri $Url -OutFile $Archive

$extractDir = Join-Path $InstallDir "extract"
if (Test-Path -Path $extractDir) {
    Remove-Item -Path $extractDir -Recurse -Force
}

Expand-Archive -Path $Archive -DestinationPath $extractDir -Force

$ffmpegExe = Get-ChildItem -Path $extractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
if ($null -eq $ffmpegExe) {
    throw "Could not find ffmpeg.exe inside the downloaded archive."
}

Copy-Item -Path $ffmpegExe.FullName -Destination (Join-Path $InstallDir "ffmpeg.exe") -Force

$ffprobeExe = Get-ChildItem -Path $extractDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1
if ($null -ne $ffprobeExe) {
    Copy-Item -Path $ffprobeExe.FullName -Destination (Join-Path $InstallDir "ffprobe.exe") -Force
}

Write-Host "Installed FFmpeg at $(Join-Path $InstallDir 'ffmpeg.exe')"
