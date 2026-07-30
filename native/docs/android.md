# Standalone Android inference

The native Distill AnyDepth runtime can execute as an ordinary Android
arm64 process. It does not use Python, PyTorch, a subprocess-based worker, or
an inference framework. Model execution uses Vulkan and the converted
`.dad` weights.

The current standalone validation target is an ADB-shell executable. This is
deliberate: it validates the same C ABI and shared library that an Android
application or InferBridge will load, without involving an engine or UI.

## Quest 3S validation

Connect the headset with developer mode enabled, then run from PowerShell:

```powershell
tools/android/run_quest_standalone.ps1
```

The script:

1. uses the Android SDK, NDK, CMake, and ADB bundled with Unity 6000.3.9f1;
2. incrementally builds the arm64 shared library and `dad_run`;
3. uploads only to `/data/local/tmp/dad-standalone`;
4. executes one full Vulkan inference; and
5. retrieves the float32 depth map under `.BuildAndroid/evidence`.

`validation/distill_any_depth_vits.dad` and
`validation/input_182x280.f32` are the default local inputs. They are
development artifacts ignored by Git. Supply `-ModelPath`, `-InputPath`,
`-Width`, `-Height`, and `-Encoder` to use other compatible inputs.

The repository-local `.BuildAndroid/unity-ndk` junction avoids a Windows
8.3-path collision between `clang.exe` and `clang++.exe` under Unity's
space-containing installation directory. It points to the existing Unity
NDK; it does not copy or modify the SDK.

Android deployment intentionally excludes the Python worker backend.
InferBridge integration should load this shared library in-process and use
the same native model artifact.
