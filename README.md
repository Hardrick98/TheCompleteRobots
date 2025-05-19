# Come condurre la sperimentazione

## 1. Prendere l'URDF dalla cartella
## 2. Promptare l'LLM con il prompt (da definire) sarà nel file prompt.txt
## 3. Creare il file .json
Inserire le pose con lo STESSO FORMATO del file GT. Rinominarlo {LLM}_{nome_robot}.json
## 4. Testarlo su webots
Lo posso fare io tranquillamente una volta ottenuti i .json


## Per visualizzare i robot

Dopo aver installato l'environment.yml:

```
python display_robot.py --urdf nao.urdf
```