# Execute no PowerShell do Windows. Usa o projeto do WSL sem copiar pesos ou dataset.
# 'Continue': no Windows PowerShell 5.1, com 'Stop', qualquer texto que o Python escreva no
# stderr (avisos do pip, logs do MediaPipe, o erro esperado de um import ausente) encerra o script.
# Cada comando externo abaixo confere $LASTEXITCODE explicitamente.
$ErrorActionPreference = 'Continue'

function Test-PythonImport([string]$modules) {
    $null = & $python -c "import $modules" 2>&1
    return $LASTEXITCODE -eq 0
}
$project = $PSScriptRoot
$environment = Join-Path $env:LOCALAPPDATA 'LibrasYOLO\venv'
$python = Join-Path $environment 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    py -3 -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw 'Instale Python 3.10 ou superior no Windows, com o launcher py.' }
}
# Uma instalação pronta não consulta o pip nem baixa pacotes em cada execução.
if (-not (Test-PythonImport 'cv2, ultralytics, pyttsx3, yaml, mediapipe')) {
    Write-Host 'Instalando dependências ausentes no ambiente do Windows...'
    if (-not (Test-PythonImport 'torch, torchvision')) {
        # A webcam não precisa de CUDA; evita baixar bibliotecas de GPU num ambiente novo.
        & $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar PyTorch para CPU.' }
    }
    & $python -m pip install -r (Join-Path $project 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar as dependências no Windows.' }
    if (-not (Test-PythonImport 'cv2, ultralytics, pyttsx3, yaml, mediapipe')) {
        & $python -c "import cv2, ultralytics, pyttsx3, yaml, mediapipe"
        throw 'As dependências foram instaladas, mas o import falhou (veja o erro acima). Se for o mediapipe, use Python 3.12.'
    }
}
# Primeiro argumento terminado em .py escolhe o script (ex.: coletar.py); o padrão é webcam.py.
$script = 'webcam.py'
$rest = @($args)
if ($rest.Count -gt 0 -and "$($rest[0])" -like '*.py') {
    $script = "$($rest[0])"
    $rest = @($rest | Select-Object -Skip 1)
}
& $python (Join-Path $project $script) @rest
exit $LASTEXITCODE
