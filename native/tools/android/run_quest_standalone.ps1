param(
    [string]$UnityEditorRoot = "C:\Program Files\Unity\Hub\Editor\6000.3.9f1",
    [string]$DeviceSerial = "340YC10G7Y0X0N",
    [string]$ModelPath = "",
    [string]$InputPath = "",
    [int]$Width = 280,
    [int]$Height = 182,
    [string]$Encoder = "vits"
)

$ErrorActionPreference = "Continue"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $repositoryRoot "validation\distill_any_depth_vits.dad"
}
if ([string]::IsNullOrWhiteSpace($InputPath)) {
    $InputPath = Join-Path $repositoryRoot "validation\input_182x280.f32"
}
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$InputPath = (Resolve-Path -LiteralPath $InputPath).Path

$androidRoot = Join-Path $UnityEditorRoot "Editor\Data\PlaybackEngines\AndroidPlayer"
$cmake = Join-Path $androidRoot "SDK\cmake\3.22.1\bin\cmake.exe"
$adb = Join-Path $androidRoot "SDK\platform-tools\adb.exe"
$ndk = Join-Path $androidRoot "NDK"
foreach ($required in @($cmake, $adb, $ndk)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required Unity Android tool is missing: $required"
    }
}

$buildRoot = Join-Path $repositoryRoot ".BuildAndroid"
$ndkLink = Join-Path $buildRoot "unity-ndk"
$buildDirectory = Join-Path $buildRoot "quest-arm64-standalone"
$evidenceDirectory = Join-Path $buildRoot "evidence"
New-Item -ItemType Directory -Force -Path $buildRoot, $evidenceDirectory |
    Out-Null

if (Test-Path -LiteralPath $ndkLink) {
    $existing = Get-Item -LiteralPath $ndkLink
    if ($existing.LinkType -ne "Junction" -or
        $existing.Target[0] -ne (Resolve-Path -LiteralPath $ndk).Path) {
        throw "The existing NDK link does not point to Unity's NDK: $ndkLink"
    }
} else {
    New-Item -ItemType Junction -Path $ndkLink -Target $ndk | Out-Null
}

& $adb -s $DeviceSerial get-state | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "ADB device is unavailable: $DeviceSerial"
}

& $cmake -S $repositoryRoot -B $buildDirectory -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$ndkLink\build\cmake\android.toolchain.cmake" `
    "-DANDROID_NDK=$ndkLink" `
    -DANDROID_ABI=arm64-v8a `
    -DANDROID_PLATFORM=android-29 `
    -DANDROID_STL=c++_static `
    -DDAD_BUILD_TESTS=OFF `
    -DDAD_BUILD_TOOLS=ON `
    -DDAD_WITH_VULKAN=ON
if ($LASTEXITCODE -ne 0) {
    throw "Android CMake configuration failed"
}
& $cmake --build $buildDirectory --parallel
if ($LASTEXITCODE -ne 0) {
    throw "Android build failed"
}

$remoteRoot = "/data/local/tmp/dad-standalone"
& $adb -s $DeviceSerial shell "rm -rf $remoteRoot && mkdir -p $remoteRoot"
if ($LASTEXITCODE -ne 0) {
    throw "Could not prepare the probe-owned device directory"
}
foreach ($upload in @(
    @{ Local = Join-Path $buildDirectory "dad_run"; Remote = "dad_run" },
    @{
        Local = Join-Path $buildDirectory "libdistill_any_depth.so"
        Remote = "libdistill_any_depth.so"
    },
    @{ Local = $ModelPath; Remote = "model.dad" },
    @{ Local = $InputPath; Remote = "input.f32" }
)) {
    & $adb -s $DeviceSerial push $upload.Local "$remoteRoot/$($upload.Remote)"
    if ($LASTEXITCODE -ne 0) {
        throw "ADB upload failed: $($upload.Local)"
    }
}

& $adb -s $DeviceSerial shell `
    "chmod 755 $remoteRoot/dad_run && cd $remoteRoot && export LD_LIBRARY_PATH=. && /system/bin/time -p ./dad_run model.dad $Encoder $Width $Height input.f32 output.f32"
if ($LASTEXITCODE -ne 0) {
    throw "Distill AnyDepth inference failed on the Android device"
}

$outputPath = Join-Path $evidenceDirectory "quest_output_${Width}x${Height}.f32"
& $adb -s $DeviceSerial pull "$remoteRoot/output.f32" $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "Could not retrieve the Android depth output"
}
$hash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Android inference completed."
Write-Host "Output: $outputPath"
Write-Host "SHA-256: $hash"
