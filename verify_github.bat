@echo off
REM ===================================================================
REM  verify_github.bat
REM
REM  Clones the repository FRESH from GitHub into a temporary folder and
REM  runs preflight_check.py against that clone.
REM
REM  Why a fresh clone: checking the local working folder proves nothing
REM  about what is actually on GitHub. Lightning.ai will clone from
REM  GitHub, so GitHub is the only version that matters.
REM
REM  Usage:  double-click, or run  verify_github.bat  from CMD
REM ===================================================================

setlocal

set REPO_URL=https://github.com/hosseinzzare/neyshekar_asr.git
set WORKDIR=%TEMP%\neyshekar_verify

echo.
echo ===================================================================
echo  Verifying the GitHub version of the project
echo ===================================================================
echo.

REM ---- locate python ------------------------------------------------
set PYTHON=python
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    set PYTHON=py
    py --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python was not found on PATH.
        echo         Install Python or open a shell where 'python' works.
        goto :fail
    )
)

git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git was not found on PATH.
    goto :fail
)

REM ---- clean any previous attempt -----------------------------------
if exist "%WORKDIR%" (
    echo [1/3] Removing previous temporary clone...
    rmdir /s /q "%WORKDIR%"
)

REM ---- fresh clone --------------------------------------------------
echo [1/3] Cloning %REPO_URL%
echo       into %WORKDIR%
git clone --depth 1 "%REPO_URL%" "%WORKDIR%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git clone failed. Check the URL and your network connection.
    goto :fail
)

REM ---- show what we got ---------------------------------------------
echo.
echo [2/3] Latest commit on GitHub:
pushd "%WORKDIR%"
git log --oneline -1
popd
echo.

REM ---- run the checks -----------------------------------------------
echo [3/3] Running preflight checks against the clone...
echo.
pushd "%WORKDIR%"
%PYTHON% preflight_check.py
set CHECK_RESULT=%ERRORLEVEL%
popd

REM ---- verdict ------------------------------------------------------
echo.
if "%CHECK_RESULT%"=="0" (
    echo ===================================================================
    echo  SUCCESS - the GitHub version is correct.
    echo  Safe to clone it on Lightning.ai and start training.
    echo ===================================================================
) else (
    echo ===================================================================
    echo  FAILED - the GitHub version is NOT correct.
    echo.
    echo  Something did not get committed or pushed. Do NOT start training.
    echo  Back in your project folder, run:
    echo      git status
    echo      git add -A
    echo      git commit -m "fix: sync remaining changes"
    echo      git push origin main
    echo  then run this script again.
    echo ===================================================================
)

REM ---- cleanup ------------------------------------------------------
if exist "%WORKDIR%" rmdir /s /q "%WORKDIR%"
echo.
echo Temporary clone removed.
echo.
pause
exit /b %CHECK_RESULT%

:fail
echo.
pause
exit /b 1
