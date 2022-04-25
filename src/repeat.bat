@ECHO OFF

for /l %%i in (1,1,%1) do (python main.py info.json >> output.log)
