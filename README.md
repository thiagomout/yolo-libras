# Reconhecimento de letras de Libras

Treinamento com YOLO, reconhecimento de sinais estáticos pela webcam e fala offline do nome da letra em português. Aceita imagens organizadas por letra ou um dataset YOLO de detecção.

## Reconhecimento por pontos da mão (recomendado)

Em vez de classificar pixels, o MediaPipe extrai **21 pontos 3D da mão** e os modelos classificam só a geometria dos dedos. Fundo, iluminação, cor da pele e distância deixam de importar, o que reduz o overfitting a um dataset de poucas pessoas com parede branca. Sem mão detectada, não há previsão.

- **Letras estáticas**: MLP sobre os pontos de um quadro. Treinado com os pontos extraídos do dataset de imagens e, opcionalmente, com gravações suas.
- **Letras com movimento (H, J, K, X, Z, Ç)**: o `MotionSegmenter` detecta quando a mão começa e para de se mover, e uma GRU classifica a trajetória do movimento. Enquanto a mão se move, a previsão estática fica suspensa.

### 1. Extrair os pontos do dataset (WSL, uma vez)

```bash
python extrair_landmarks.py --dataset workspace/extraido
```

Gera `workspace/landmarks_extraido.npz` e um relatório `.json` com a taxa de detecção por letra. Na primeira vez, baixa `model/hand_landmarker.task` (~8 MB).

### 2. Gravar exemplos pela webcam (Windows)

> **Coleta em grupo:** veja [COLETA.md](COLETA.md) (gravações enviadas pelo Git, metas por pessoa, `resumo_coleta.py` e `treinar_landmarks.py --por-pessoa`).

O dataset não tem nenhuma letra com movimento; elas precisam ser gravadas. Só os pontos são salvos (em `coleta/`, que vai para o Git), nunca a imagem.

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 coletar.py --pessoa ana --letras H J K X Z CC NADA
```

`CC` = Ç. `NADA` são movimentos que **não** são letras: transições entre letras, mão entrando no quadro, coçar o rosto. Sem essa classe, qualquer movimento vira uma letra. Em cada exemplo, aperte ESPAÇO, faça o sinal partindo da mão parada, pare e aperte ESPAÇO. Comece com 30 a 50 exemplos por letra; 2 ou 3 pessoas fazem muita diferença. A barra de movimento mostra o limiar; o sinal precisa passar dele.

Gravar letras estáticas com a sua webcam também ajuda (`--tipo estatico` grava 2 s; mova e gire levemente a mão):

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 coletar.py --pessoa ana --letras A B C D E --tipo estatico
```

### 3. Treinar (WSL)

```bash
python treinar_landmarks.py                     # divide gravações aleatoriamente
python treinar_landmarks.py --pessoa-teste bia  # bia só no teste: mede generalização real
```

Treina em segundos ou poucos minutos e gera `model/mao.pt` e `model/mao_metadata.json` com métricas por fonte (`dataset_test`, `coleta_val`, `coleta_test`). Com coleta e dataset juntos, 30% de cada batch vem da coleta (`--collected-share`). Para medir de verdade, olhe o `coleta_test` com uma pessoa de fora.

### 4. Usar

`webcam.py` usa `model/mao.pt` automaticamente quando ele existe (`--modo yolo` volta ao modelo de imagem). Não há quadrado de recorte: a mão pode estar em qualquer lugar do quadro. Ajustes: `--dyn-conf` (confiança das letras com movimento), `--motion` (limiar de movimento, em palmas por segundo; padrão 1,5), `--smooth` (quadros de suavização).

Comparação nas imagens de teste do dataset, com a parede branca trocada por um fundo colorido: YOLO 49%, pontos 90%. Com luz amarelada, o YOLO se saiu melhor (97% contra 77%) porque o MediaPipe perde mais mãos em imagens de 64×64; na webcam, em resolução normal, a detecção tende a ser melhor. Os dois modelos acertam quase 100% nas imagens originais, que se parecem com o treino.

O treino espelha as mãos aleatoriamente, então mão esquerda e direita funcionam. Duas poses que sejam só espelho uma da outra ficam indistinguíveis.

## Executar câmera e voz no Windows

No ambiente analisado, o WSL não tem nenhum `/dev/video*`: a webcam do Windows não está disponível ao OpenCV do Linux. Use o WSL para treinar e o Python do Windows para reconhecer. Não é necessário copiar o dataset nem os pesos.

Instale Python 3.10 ou superior no Windows com o launcher `py`. Abra o PowerShell na pasta deste projeto (ela pode estar em `\\wsl.localhost\NOME_DA_DISTRO\home\thiago\yolo-libras`) e execute:

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1
```

O script cria um ambiente Python **do Windows** em `%LOCALAPPDATA%\LibrasYOLO\venv`, instala as dependências e abre a câmera. A primeira execução precisa de internet. Não use a `.venv` do Linux pelo Windows.

Para trocar a câmera ou testar sem áudio:

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 --camera 1
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 --no-voice
```

Para testar somente a fala, depois de criar o ambiente:

```powershell
& "$env:LOCALAPPDATA\LibrasYOLO\venv\Scripts\python.exe" .\voz.py
```

Se a câmera falhar no Windows, confira a permissão de câmera para aplicativos de desktop e feche outros aplicativos que estejam usando a webcam.

## Treinar no WSL/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**O dataset deste projeto já está extraído. Reutilize-o:**

```bash
python treinar.py --dataset workspace/extraido --epochs 50
```

Para um novo ZIP:

```bash
python treinar.py --zip data/archive.zip --epochs 50
```

O ZIP é extraído uma vez por identidade de arquivo. Extrações concluídas são reutilizadas; o script não apaga todo o `workspace`. A divisão em treino/validação também é reutilizada quando os arquivos e parâmetros não mudam. Usa hardlinks quando possível, para evitar uma segunda cópia das imagens; por isso, não edite as imagens preparadas. Se uma preparação existente estiver incompleta ou diferente, o script informa isso em vez de apagá-la.

Outras opções:

```bash
python treinar.py --dataset workspace/extraido --prepare-only
python treinar.py --dataset workspace/extraido --device 0 --batch 8 --workers 0
python treinar.py --dataset workspace/extraido --seed 42 --val-fraction 0.15
```

`--model` permite escolher outros pesos pré-treinados compatíveis com a tarefa. O padrão mantém YOLO26n, que já está disponível no projeto. `--include-j-z` mantém essas classes quando existirem; uma imagem isolada não reconhece o movimento.

Arquivos gerados:

- `model/best.pt` e `model/metadata.json`: melhor modelo, classes, pré-processamento e métricas de teste disponíveis.
- `workspace/dataset_classificacao_*/split_manifest.json`: origem das imagens, contagens por classe e divisão, semente e duplicatas removidas.
- `runs/`: métricas, matriz de confusão e gráficos.

O conjunto `test` da origem é mantido separado, quando existir, e avaliado depois de selecionar os melhores pesos. Duplicatas exatas são removidas, inclusive entre treino e teste; imagens idênticas com rótulos diferentes geram erro. Imagens quase iguais, frames da mesma gravação e pessoas compartilhadas entre conjuntos ainda podem inflar as métricas. Para medir generalização real, use pessoas e sessões diferentes no teste.

## Reconhecimento e voz

No Linux com webcam disponível ou no ambiente Python do Windows:

```bash
python webcam.py
python webcam.py --conf 0.75 --margin 0.15 --hold 0.6
python voz.py
```

Coloque a mão inteira dentro do quadrado central e faça ela ocupar boa parte do espaço. Use fundo neutro e boa iluminação. O classificador recebe apenas esse recorte; modelos de detecção recebem o quadro inteiro. `--roi 0` desativa o recorte; `--roi 0.45` diminui o quadrado.

A entrada não é espelhada por padrão. Experimente `--mirror-input` se a orientação da mão no dataset exigir. A letra só é confirmada após uma duração contínua (`--hold`), confiança mínima (`--conf`) e, para classificação, diferença entre as duas maiores probabilidades (`--margin`). `Q`/`Esc` encerra; `R` repete a letra. Uma pausa de previsões rejeitadas (`--release`, padrão 0,5 s) permite falar a mesma letra novamente.

O classificador sempre escolhe uma classe, mesmo sem uma mão: os limiares não garantem detectar ausência. Uma classe de fundo/sem sinal com exemplos reais ou um detector de mão é necessária para rejeição mais confiável.

A fala usa os nomes das letras, como “Letra bê”, e uma fila limitada para não acumular falas antigas. No Windows, usa `pyttsx3`; no WSL, tenta o sintetizador do Windows via PowerShell; no Linux, prefere `espeak-ng`. Falhas aparecem no terminal e na janela.

```bash
python voz.py --backend windows  # WSL com PowerShell disponível
python voz.py --backend espeak  # Linux; requer espeak-ng e saída de áudio funcional
```

## O que mudou e limites do dataset

O treinamento anterior atingiu 100% de acerto na validação, mas usava imagens de 64×64 com a mão bem enquadrada. A inferência anterior classificava a cena inteira espelhada. Uma validação excelente nessas imagens não comprova desempenho na webcam.

O novo treinamento preserva a imagem inteira, com pequenas variações de posição, rotação e cor. Evita os recortes aleatórios, apagamento de regiões e transformações automáticas que podem esconder os dedos. O mesmo redimensionamento é usado na validação e na inferência dos novos modelos. Há parada antecipada, semente configurável e avaliação independente quando há conjunto de teste.

Essas mudanças precisam ser medidas em imagens reais da sua câmera; não há garantia de ganho de precisão sem novos dados. O dataset atual contém apenas A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W e Y. O modelo YOLO não reconhecerá letras ausentes nem sinais com movimento; para estas, use o modo por pontos da mão com gravações próprias.

Para aumentar a precisão ao vivo, acrescente imagens da sua webcam com diferentes pessoas, iluminação, distâncias e fundos, usando o mesmo enquadramento. Separe pessoas/sessões entre treino, validação e teste.

No WSL2, apagar arquivos libera espaço dentro do Linux, mas um VHD expansível pode não reduzir automaticamente seu tamanho no Windows. Reutilizar os dados evita extrações e cópias recorrentes. Referências: [disco do WSL](https://learn.microsoft.com/en-us/windows/wsl/disk-space), [compactação de VHD](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact-vdisk) e [USB no WSL](https://learn.microsoft.com/en-us/windows/wsl/connect-usb).

## Verificar o código

```bash
python -m unittest discover -s tests -v
```

## Investigar validação fácil e overfitting

Para procurar imagens visualmente próximas entre treino, validação e teste:

```bash
python auditar_dataset.py --dataset workspace/dataset_classificacao_bc96d0d8b965
```

Troque o caminho pela pasta preparada correspondente ao seu treino. A auditoria compara hashes perceptuais e registra pares para inspeção em `workspace/auditoria_dataset.json`. Distância baixa é uma triagem de similaridade, não uma prova de duplicação, de mesma pessoa ou de vazamento.

Neste dataset, a auditoria de 10 imagens por classe encontrou distância de até 4 bits (de 63) para 210/210 exemplos de validação e 126/210 exemplos de teste. Alguns pares inspecionados são quase iguais. Além disso, o novo treino já atingiu 100% na validação na primeira época. Avaliar imagens novas da webcam, separadas por pessoa/sessão, é mais informativo do que reduzir épocas ou trocar de modelo com base nessas métricas.

O launcher do Windows agora verifica se as dependências já estão instaladas antes de chamar o pip. Instalações novas usam primeiro o PyTorch para CPU para a webcam, evitando os pacotes de CUDA. Ambientes existentes são reutilizados.
