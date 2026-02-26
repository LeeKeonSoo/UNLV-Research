@echo off
setlocal EnableExtensions

:: Canonical run: final-validation defaults + memory-safe profile
python "%~dp0run_phase1_lowmem.py"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
