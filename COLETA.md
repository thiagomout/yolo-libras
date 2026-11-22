# Guia de coleta para o grupo

Cada pessoa grava os sinais pela própria webcam. Só os **pontos da mão** são salvos (21 pontos por quadro), nunca imagem ou vídeo. As gravações vão para uma pasta compartilhada no Google Drive; o Git guarda só o código e o modelo treinado.

## 1. Preparar (uma vez)

**Pasta do Drive.** Quem organiza cria a pasta `Libras-coleta` no próprio Drive e compartilha com o grupo como **Editor**. Quem recebeu abre o Drive no navegador, vai em *Compartilhados comigo*, clica com o botão direito na pasta → *Organizar* → *Adicionar atalho* → *Meu Drive*. Sem o atalho, a pasta não aparece no computador.

**Google Drive para computador.** Instale ([google.com/drive/download](https://www.google.com/drive/download/)) e entre na conta. A pasta aparece no Explorador de Arquivos, normalmente em `G:\Meu Drive\Libras-coleta` (em inglês: `G:\My Drive\Libras-coleta`). Confira a letra e o nome no Explorador.

**Python e projeto.** Instale o Python 3.12 para Windows ([python.org](https://www.python.org/downloads/)), marcando *Add python.exe to PATH*, e o Git. Depois, no PowerShell:

```powershell
cd $HOME\Documents
git clone https://github.com/thiagomout/yolo-libras.git
cd yolo-libras
```

**Dizer onde fica a pasta**, uma vez só (troque pelo caminho que você conferiu acima):

```powershell
setx LIBRAS_COLETA "G:\Meu Drive\Libras-coleta"
```

Feche e abra o PowerShell de novo para a variável valer. Todos os scripts passam a gravar e ler nessa pasta.

**Testar.** `powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1` abre o reconhecimento. Na primeira vez instala as dependências (precisa de internet, alguns minutos).

## 2. Gravar

Use **sempre o mesmo apelido** em `--pessoa` (minúsculo, sem espaço, sem nome completo). É ele que permite testar o modelo com uma pessoa que ficou de fora do treino.

Letras com movimento e a classe NADA (`CC` = Ç):

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 coletar.py --pessoa apelido --letras H J K X Z CC NADA
```

Letras estáticas:

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 coletar.py --pessoa apelido --tipo estatico --letras A B C D E F G I L M N O P Q R S T U V W Y
```

O terminal mostra `Gravando em: ...`. Confira se é a pasta do Drive.

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

Dá para dividir em várias sessões; as gravações se somam. Para ver o que falta (de todo mundo):

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 resumo_coleta.py
```

### Como gravar bem

- **Letras com movimento:** comece com a mão parada na forma inicial, faça o movimento, pare, e só então aperte ESPAÇO. O sinal precisa passar do traço branco da barra de movimento; o esqueleto fica vermelho enquanto o movimento é detectado.
- **NADA** é tudo que **não** é letra: trocar de uma letra para outra, mão entrando e saindo do quadro, coçar o rosto, ajeitar o cabelo, acenar. Sem isso, qualquer movimento vira letra.
- **Estáticas:** durante os 2 s, mova e gire **levemente** a mão e mude um pouco a distância, mantendo a letra correta.
- **Varie entre as sessões:** perto e longe da câmera, luz de dia e de lâmpada, fundos diferentes, velocidade normal e mais lenta. Se puder, grave parte com a outra mão.
- **Errou? Aperte D na hora.** Um exemplo errado atrapalha mais do que um exemplo a menos.
- Não renomeie, mova ou edite arquivos na pasta do Drive. Os nomes já são únicos por pessoa e horário.

## 3. Treinar (quem organiza)

No Windows, usando a mesma pasta do Drive:

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar_windows.ps1 treinar_landmarks.py --por-pessoa
```

Ou no WSL, com GPU, montando o drive do Google (`G:`) uma vez por sessão:

```bash
sudo mkdir -p /mnt/g && sudo mount -t drvfs G: /mnt/g
LIBRAS_COLETA="/mnt/g/Meu Drive/Libras-coleta" python treinar_landmarks.py --por-pessoa
```

`--por-pessoa` treina uma vez para cada pessoa deixando-a de fora e mostra a acurácia dela, que é a melhor estimativa de como o modelo se sai com alguém novo. Depois treina o modelo final com todo mundo. Se uma pessoa vai bem pior que as outras, vale olhar como ela faz os sinais ou gravar mais com ela.

Para treinar no Windows, o modelo estático precisa dos pontos do dataset original (`workspace/landmarks_extraido.npz`), que não vão para o Git. Rode a partir da pasta do projeto no WSL (`\\wsl.localhost\...`) ou copie esse arquivo.

Por fim, publique o modelo para o grupo:

```bash
git add model/mao.pt model/mao_metadata.json
git commit -m "Atualiza modelo com coleta do grupo"
git push
```

Os outros só precisam de `git pull` para usar o modelo novo.
