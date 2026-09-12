param(
    [int]$Framerate = 10,
    [string]$InputDir = "sim",
    [string]$Output = "wave.mp4",
    [int]$Frames = 0,
    [string]$Ffmpeg = "",
    [string]$Extension = "pgm"
)

$ErrorActionPreference = "Stop"

$Extension = $Extension.TrimStart(".")

if (-not (Test-Path -Path $InputDir -PathType Container)) {
    throw "Input directory '$InputDir' does not exist."
}

$firstFrame = Join-Path $InputDir ("frame_00000.{0}" -f $Extension)
if (-not (Test-Path -Path $firstFrame -PathType Leaf)) {
    throw "Expected first frame '$firstFrame' was not found."
}

if ($Ffmpeg -eq "") {
    $localFfmpeg = Join-Path "ffmpeg_tool" "ffmpeg.exe"
    if (Test-Path -Path $localFfmpeg -PathType Leaf) {
        $Ffmpeg = $localFfmpeg
    } else {
        $Ffmpeg = "ffmpeg"
    }
}

$ffmpegArgs = @(
    "-y",
    "-framerate", $Framerate,
    "-i", (Join-Path $InputDir ("frame_%05d.{0}" -f $Extension))
)

if ($Frames -gt 0) {
    $ffmpegArgs += @("-frames:v", $Frames)
}

$ffmpegArgs += @(
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    $Output
)

& $Ffmpeg @ffmpegArgs
