@echo off
REM Get system architecture using wmic
FOR /F "tokens=2 delims==" %%A IN ('wmic os get osarchitecture /format:list ^| find "OSArchitecture"') DO SET ARCH=%%A

REM Print detected architecture
echo Detected Architecture: %ARCH%

REM Set the base image based on the architecture
IF "%ARCH%"=="64-bit" (
    SET BASE_IMAGE=tensorflow/tensorflow:2.8.0
) ELSE IF "%ARCH%"=="ARM64" (
    SET BASE_IMAGE=armswdev/tensorflow-arm-neoverse:r22.04-tf-2.8.0-eigen
) ELSE (
    ECHO Unsupported architecture: %ARCH%
    EXIT /B 1
)

REM Ensure the correct path format and run Docker commands
echo Using BASE_IMAGE: %BASE_IMAGE%

docker-compose build --build-arg BASE_IMAGE=%BASE_IMAGE%
docker-compose up
