# Libras YOLO — versão inicial

Projeto inicial em Python para treinar um YOLO com imagens de letras estáticas em Libras, reconhecer os sinais pela webcam e pronunciar a letra em voz alta.

## O que esta versão faz

1. Extrai um ZIP com o dataset.
2. Tenta identificar automaticamente um destes formatos:
   - **Classificação:** uma pasta para cada letra, como `A/imagem1.jpg`, `B/imagem1.jpg` etc.
   - **Detecção YOLO:** um arquivo `data.yaml`/`data.yml` com `train`, `val` e `names`, mais imagens e rótulos no formato YOLO.
3. Treina um modelo pequeno da família YOLO26 com pesos pré-treinados.
4. Salva o modelo treinado em `model/best.pt`.
5. Abre a webcam e mostra a previsão. Quando a letra fica estável por alguns quadros, o computador fala a letra em voz alta.

Por padrão, as classes **J** e **Z** são removidas porque os sinais dessas letras envolvem movimento. Para manter todas as classes do dataset, use `--include-j-z` no treinamento.

> A estrutura do dataset varia. O script cobre os formatos mais comuns; se o ZIP tiver CSVs, imagens sem pastas por classe, anotações COCO ou outra organização, será necessário adaptar a etapa de preparação.

## 1. Preparar o ambiente (Windows)

Instale Python 3.10 ou 3.11 e marque a opção **Add Python to PATH** durante a instalação.

Abra o PowerShell nesta pasta e execute:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação do ambiente, use o Prompt de Comando:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A primeira utilização pode baixar os pesos iniciais do YOLO pela internet. O treinamento também pode ficar lento em computadores sem GPU compatível; nesse caso, ele pode usar a CPU.

## 2. Colocar o dataset ZIP na pasta

Copie seu arquivo ZIP para esta pasta e, se possível, renomeie-o para `dataset.zip`.

Exemplo:

```text
libras_yolo_inicial/
  treinar.py
  webcam.py
  requirements.txt
  dataset.zip
```

## 3. Treinar

```powershell
python treinar.py --zip dataset.zip
```

O padrão são 50 épocas. Para um teste rápido, use 5 ou 10 épocas:

```powershell
python treinar.py --zip dataset.zip --epochs 10
```

Para ajustar o batch, por exemplo em uma máquina com pouca memória:

```powershell
python treinar.py --zip dataset.zip --epochs 50 --batch 8
```

Para incluir J e Z, caso deseje experimentar mesmo sendo sinais com movimento:

```powershell
python treinar.py --zip dataset.zip --include-j-z
```

Arquivos importantes gerados:

- `model/best.pt`: pesos treinados.
- `model/metadata.json`: tarefa e classes do modelo.
- `runs/`: gráficos, métricas e histórico do treinamento.
- `workspace/`: dataset extraído e, para classificação, conjunto separado em treino e validação.

## 4. Usar a webcam e a voz

Depois do treinamento, execute:

```powershell
python webcam.py
```

Controles da janela:

- `Q` ou `Esc`: sair.
- `R`: repetir em voz alta a última letra estável.

Se houver mais de uma câmera, tente:

```powershell
python webcam.py --camera 1
```

Para testar sem voz:

```powershell
python webcam.py --no-voice
```

A confiança mínima padrão é 70%. Para experimentar um valor diferente:

```powershell
python webcam.py --conf 0.60
```

## 5. Dicas para o primeiro teste

- Deixe a mão grande o suficiente para aparecer claramente na imagem.
- Use iluminação uniforme e evite fundos muito parecidos com a mão.
- Experimente posicionar a mão no centro da câmera.
- As imagens de treino devem ser variadas: pessoas, fundos, distâncias e iluminação diferentes.
- A divisão treino/validação ajuda a avaliar o modelo, mas não garante o mesmo desempenho com uma câmera real. Teste várias letras e fundos.
- A síntese de voz usa as vozes instaladas no sistema. Para melhor pronúncia, instale/ative uma voz em português (Brasil) nas configurações de idioma e voz do Windows.

## Observação sobre o tipo de YOLO

Quando as imagens estão organizadas por letra, o script usa **YOLO para classificação de imagens**: ele escolhe a classe da imagem inteira. Se o ZIP já vier anotado com caixas delimitadoras no formato YOLO, usa **YOLO para detecção**, que também localiza a mão na imagem.
