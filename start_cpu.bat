@echo off
call venv\Scripts\activate.bat
echo Starting Onnx API Endpoint on CPU...
python server.py --device cpu --threshold 0.2
pause
