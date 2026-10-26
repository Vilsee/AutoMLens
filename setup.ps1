# Run from the project root in PowerShell:  .\setup.ps1
$ErrorActionPreference = 'Stop'
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host 'Core packages installed. Optional (large): pip install autogluon.tabular'
