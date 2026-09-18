$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$installation = & $vswhere -latest -version '[17.0,18.0)' -products '*' -property installationPath
if ($LASTEXITCODE -ne 0 -or !$installation) { throw 'Visual Studio 2022 is required' }
$toolset = Join-Path $installation 'VC/Tools/MSVC/14.37.32822'
if (!(Test-Path -LiteralPath $toolset)) {
    $installer = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/setup.exe'
    # Component ID from Microsoft's VS 2022 Build Tools component catalog.
    $installArguments = @('modify', '--installPath', ('"{0}"' -f $installation),
        '--add', 'Microsoft.VisualStudio.Component.VC.14.37.17.7.x86.x64',
        '--quiet', '--norestart')
    $result = Start-Process -FilePath $installer -ArgumentList $installArguments -Wait -PassThru -WindowStyle Hidden
    if ($result.ExitCode -notin @(0, 3010)) { throw "MSVC installation failed: $($result.ExitCode)" }
}
if (!(Test-Path -LiteralPath $toolset)) { throw 'Audited MSVC 14.37.32822 was not installed' }
$vcvars = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
$compilerEnvironment = & cmd.exe /s /c ('"{0}" -vcvars_ver=14.37 >nul && set' -f $vcvars)
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed' }
foreach ($entry in $compilerEnvironment) {
    if ($entry -match '^([^=]+)=(.*)$') {
        if ([Environment]::GetEnvironmentVariable($matches[1], 'Process') -ne $matches[2]) {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
            $entry | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
        }
    }
}
& cl.exe /Bv /? 2>&1 | Select-Object -First 8 | ForEach-Object { Write-Host $_ }
# bindgen needs libclang; C++ compilation still uses the selected MSVC toolset.
if (Test-Path 'C:/Program Files/LLVM/bin/libclang.dll') {
    'LIBCLANG_PATH=C:/Program Files/LLVM/bin' | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
