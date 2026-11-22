# Guia de coleta para o grupo

Cada pessoa grava os sinais pela própria webcam. Só os **pontos da mão** são salvos (21 pontos por quadro, com posição e tamanho normalizados), nunca imagem ou vídeo. As gravações ficam em `coleta/` e vão para o Git junto com o código, então o repositório também é o dataset.

**Consentimento:** o repositório é público. Grave só se você concordar que seus pontos da mão fiquem públicos, e use um apelido, nunca o nome completo. Depois de enviado, um arquivo continua no histórico do Git mesmo se for apagado.

## 1. Preparar (uma vez)

1. Instale o Python 3.12 para Windows ([python.org](https://www.python.org/downloads/)), marcando *Add python.exe to PATH*, e o [Git](https://git-scm.com/download/win).
2. Peça para ser adicionado como colaborador no GitHub e aceite o convite que chega por e-mail. Sem isso, o `git push` é recusado.
3. No PowerShell:

```powershell
cd $HOME\Documents
git clone https://github.com/thiagomout/yolo-libras.git
cd yolo-libras
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1
```

O último comando abre o reconhecimento. Na primeira vez instala as dependências (precisa de internet, alguns minutos). Feche com Q.

## 2. Gravar

Antes de cada sessão, pegue as novidades (modelo novo, gravações dos outros):

```powershell
git pull
```

Use **sempre o mesmo apelido** em `--pessoa` (minúsculo, sem espaço). É ele que permite testar o modelo com uma pessoa que ficou de fora do treino.

Letras com movimento e a classe NADA (`CC` = Ç):

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 coletar.py --pessoa apelido --letras H J K X Z CC NADA
```

Letras estáticas:

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 coletar.py --pessoa apelido --tipo estatico --letras A B C D E F G I L M N O P Q R S T U V W Y
```

| Tecla | Ação |
|---|---|
| ESPAÇO | começa/para a gravação (estáticas param sozinhas em 2 s) |
| N / P | próxima / letra anterior |
| D | apaga a última gravação |
| Q / ESC | sai |

### Metas por pessoa

| O quê | Quantidade |
|---|---|
| Cada letra estática | 15 |
| Cada letra com movimento (H, J, K, X, Z, Ç) | 30 |
| NADA | 40 |

Dá para dividir em várias sessões; as gravações se somam. Para ver o que falta (de todo mundo que já enviou):

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 resumo_coleta.py
```

### Como gravar bem

- **Letras com movimento:** comece com a mão parada na forma inicial, faça o movimento, pare, e só então aperte ESPAÇO. O sinal precisa passar do traço branco da barra de movimento; o esqueleto fica vermelho enquanto o movimento é detectado.
- **NADA** é tudo que **não** é letra: trocar de uma letra para outra, mão entrando e saindo do quadro, coçar o rosto, ajeitar o cabelo, acenar. Sem isso, qualquer movimento vira letra.
- **Estáticas:** durante os 2 s, mova e gire **levemente** a mão e mude um pouco a distância, mantendo a letra correta.
- **Varie entre as sessões:** perto e longe da câmera, luz de dia e de lâmpada, fundos diferentes, velocidade normal e mais lenta. Se puder, grave parte com a outra mão.
- **Errou? Aperte D na hora.** Um exemplo errado atrapalha mais do que um exemplo a menos.
- Não renomeie nem edite arquivos de `coleta/`, nem os seus nem os dos outros.

## 3. Enviar

Ao fim de cada sessão:

```powershell
git add coleta
git commit -m "Coleta de apelido"
git pull --rebase
git push
```

O `git pull --rebase` junta o que os outros enviaram enquanto você gravava. Como cada arquivo tem o apelido e o horário no nome, não há conflito. Envie **só** a pasta `coleta`: se `git status` mostrar outros arquivos alterados, não os inclua.

## 4. Treinar (quem organiza)

No WSL, na pasta do projeto:

```bash
git pull
python resumo_coleta.py
python treinar_landmarks.py --por-pessoa
```

`--por-pessoa` treina uma vez para cada pessoa deixando-a de fora e mostra a acurácia dela, que é a melhor estimativa de como o modelo se sai com alguém novo. Depois treina o modelo final com todo mundo. Se uma pessoa vai bem pior que as outras, vale olhar como ela faz os sinais ou gravar mais com ela.

O modelo estático também usa os pontos do dataset original (`workspace/landmarks_extraido.npz`, gerado por `extrair_landmarks.py`), que não vai para o Git.

Publique o modelo para o grupo:

```bash
git add model/mao.pt model/mao_metadata.json
git commit -m "Atualiza modelo com coleta do grupo"
git push
```

Os outros só precisam de `git pull` para usar o modelo novo.
