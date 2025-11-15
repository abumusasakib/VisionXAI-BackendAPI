$ARCH = (Get-CimInstance Win32_ComputerSystem).SystemType
if (-not $ARCH) {
    $ARCH = (Get-WmiObject Win32_ComputerSystem).SystemType
}

# Determine base image based on architecture
if ($ARCH -eq "x64-based PC") {
    $env:BASE_IMAGE = "tensorflow/tensorflow:2.8.0"
}
elseif ($ARCH -eq "ARM64-based PC") {
    $env:BASE_IMAGE = "armswdev/tensorflow-arm-neoverse:r22.04-tf-2.8.0-eigen"
}
else {
    Write-Host "Unsupported architecture: $ARCH"
    exit 1
}

Write-Host "Using BASE_IMAGE: $env:BASE_IMAGE"

# Set the correct BASE_IMAGE path
docker-compose build --build-arg BASE_IMAGE=$env:BASE_IMAGE
docker-compose up
