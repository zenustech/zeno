@rem mode con cols=15 lines=1
@echo off
if "%1" == "h" goto begin
mshta vbscript:createobject("wscript.shell").run("""%~nx0"" h",0)(window.close)&&exit
:begin

set ScriptDir=%~dp0
set ScriptDir=%ScriptDir:~0,-1%
%ScriptDir%\start.bat