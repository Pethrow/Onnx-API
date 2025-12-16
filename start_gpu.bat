@echo off
call venv\Scripts\activate.bat
echo Starting Onnx API Endpoint on GPU...
python server.py --device cuda --threshold 0.3
pause
