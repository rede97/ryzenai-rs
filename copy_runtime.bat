@echo off
set DEST_DIR="./target/debug"

xcopy "%RYZEN_AI_INSTALLATION_PATH%\onnxruntime\bin\*" /E /I %DEST_DIR%
xcopy "%RYZEN_AI_INSTALLATION_PATH%\voe-4.0-win_amd64\vaip_config.json" %DEST_DIR%
