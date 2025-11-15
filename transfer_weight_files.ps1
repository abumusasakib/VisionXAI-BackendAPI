# Load .env file
Get-Content ".env" | ForEach-Object {
    if ($_ -match "^\s*([^#=]+)\s*=\s*(.+)\s*$") {
        Set-Variable -Name $matches[1].Trim() -Value $matches[2].Trim()
    }
}

# Normalize path (optional if you're using SCP with Git Bash/WSL)
$localPathUnix = $LOCAL_PATH -replace '\\', '/'

# Construct and execute the SCP command
$scpCommand = "scp -r `"$localPathUnix`" ${REMOTE_USER}@${REMOTE_HOST}:`"$REMOTE_PATH`""
Write-Host "Running: $scpCommand"
Invoke-Expression $scpCommand
