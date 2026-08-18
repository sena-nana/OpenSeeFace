%ECHO OFF

facetracker -l 1

echo Make sure that nothing is accessing your camera before you proceed.

set /p cameraNum=Select your camera from the list above and enter the corresponding number:

echo Default mode is used. Enter FPS:

set /p fps=Select the FPS:

facetracker -c %cameraNum% -F %fps% -v 3 -P 1 --discard-after 0 --scan-every 0 --no-3d-adapt 1 --max-feature-updates 900

pause
