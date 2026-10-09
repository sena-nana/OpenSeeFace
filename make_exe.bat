@echo off
echo Started
cargo build --release --manifest-path runtime-ort\Cargo.toml --bin facetracker
if errorlevel 1 exit /b 1
if not exist dist\facetracker mkdir dist\facetracker
copy /Y runtime-ort\target\release\facetracker.exe dist\facetracker\facetracker.exe
xcopy /E /I /Y models dist\facetracker\models
copy /Y run.bat dist\facetracker\run.bat
echo Files should be available in dist\facetracker
echo Finished
