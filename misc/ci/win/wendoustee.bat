@echo off
if "%1"=="-a" (
  shift
) else (
  type nul > %1
)
for /f "tokens=1* delims=:" %%a in ('findstr /n "^"') do (
  echo.%%b
  echo.%%b >> %1
)
