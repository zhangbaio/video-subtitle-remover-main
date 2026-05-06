param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

& (Join-Path $PSScriptRoot "build_installer_ascii.ps1") -BackendPreset rtx50 @Rest
