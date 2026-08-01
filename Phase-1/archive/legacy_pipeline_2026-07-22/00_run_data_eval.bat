@echo off
setlocal
python "%~dp0\00_run_data_eval.py"
exit /b %errorlevel%
