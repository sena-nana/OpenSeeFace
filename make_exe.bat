@echo off
echo Started
uv run python runtime-ort\scripts\wrap_preprocess.py --models-dir models
if errorlevel 1 python runtime-ort\scripts\wrap_preprocess.py --models-dir models
if errorlevel 1 echo warning: could not generate models\pre (need Python + onnx); --device gpu can generate them at first run
cargo build --release --features gpu --manifest-path runtime-ort\Cargo.toml --bin facetracker
if errorlevel 1 exit /b 1
if not exist dist\facetracker mkdir dist\facetracker
copy /Y runtime-ort\target\release\facetracker.exe dist\facetracker\facetracker.exe
xcopy /E /I /Y models dist\facetracker\models
copy /Y run.bat dist\facetracker\run.bat
echo Files should be available in dist\facetracker
echo Finished
