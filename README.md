# TCC Formula 1 - Analise Preditiva de Tempo de Volta

## Titulo
Analise Multicircuito de Tempos de Volta na Formula 1: comparacao de abordagens de modelagem temporal para previsao e suporte estrategico.

## Objetivo
Este projeto desenvolve e compara modelos de Machine Learning para prever tempo de volta em corridas de Formula 1 usando dados historicos de 2022 a 2025.

O foco atual do repositorio esta em:
- comparacao entre Regressao Linear e XGBoost
- comparacao entre abordagens de validacao temporal
- avaliacao em holdout sequencial final

## Principais Atualizacoes Implementadas
As funcionalidades abaixo foram implementadas e consolidadas no codigo:

1. Execucao multi-pistas automatizada com um unico script.
2. Parametrizacao de pista via variavel de ambiente TARGET_GP_NAME em todos os modelos principais.
3. Padronizacao do split temporal por LapNumber:
   - 80% inicial para modelagem
   - 20% final para holdout sequencial
4. Organizacao dos resultados por familia e abordagem.
5. Remocao do acoplamento com batch_runs no orquestrador atual.
6. Inclusao de IC95% no holdout para cv, ew e sw (bootstrap) para:
   - RMSE
   - MAE
   - R2

## Estrutura do Projeto

```text
TCC/
|- Bibliografia/
|- Data/
|  |- Bahrain/
|  |- Hungary/
|  |- Italy/
|  |- Saudi Arabia/
|  |- United States/
|- Scripts/
|  |- Notebooks/
|  |- Source/
|  |  |- script_model_data.py
|  |  |- run_all_models_tracks.py
|  |  |- model_lr_cv.py
|  |  |- model_lr_ew.py
|  |  |- model_lr_sw.py
|  |  |- model_lr_wf.py
|  |  |- model_xgb_cv.py
|  |  |- model_xgb_ew.py
|  |  |- model_xgb_sw.py
|  |  |- model_xgb_wf.py
|  |- ModelData/
|  |- Results/
|  |  |- linear_regression/
|  |  |  |- cv/
|  |  |  |- ew/
|  |  |  |- sw/
|  |  |  |- wf/
|  |  |- xgboost/
|  |  |  |- cv/
|  |  |  |  |- params/
|  |  |  |- ew/
|  |  |  |  |- params/
|  |  |  |- sw/
|  |  |  |  |- params/
|  |  |- runs/
|  |- fastf1_cache/
|- Utils/
|  |- compounds.json
|  |- requirements.txt
|- README.md
```

Observacao:
- Ainda podem existir pastas legadas de rodadas antigas dentro de alguns subdiretorios de resultados.
- A estrutura ativa de execucao e salvamento atual e a descrita acima.

## Fluxo de Execucao

### 1) Gerar base limpa por pista

```bash
python Scripts/Source/script_model_data.py
```

Entrada esperada:
- Data/<Pista>/...

Saida esperada:
- Scripts/ModelData/<Grand Prix>/<safe_gp_name>_cleaned_data.csv

### 2) Rodar modelos individualmente

Exemplos:

```bash
python Scripts/Source/model_lr_cv.py
python Scripts/Source/model_lr_ew.py
python Scripts/Source/model_lr_sw.py
python Scripts/Source/model_xgb_cv.py
python Scripts/Source/model_xgb_ew.py
python Scripts/Source/model_xgb_sw.py
```

### 3) Rodar lote multi-pistas e multi-modelos

```bash
python Scripts/Source/run_all_models_tracks.py
```

O orquestrador permite selecionar pistas e modelos e gera logs por execucao.

## Parametrizacao por Pista
Todos os scripts principais usam:

- TARGET_GP_NAME (variavel de ambiente)

Se a variavel nao for informada, o script usa valor padrao definido no arquivo.

Exemplo no PowerShell:

```powershell
$env:TARGET_GP_NAME = "United States Grand Prix"
python Scripts/Source/model_xgb_sw.py
```

## Abordagens de Validacao

### CV
- validacao interna com K-Fold na base de modelagem (80%)

### EW
- Expanding Window na base de modelagem (80%)

### SW
- Sliding Window na base de modelagem (80%)

### WF
- Walk-Forward (scripts especificos de WF)

## Metricas e IC no Holdout
Nos metodos cv, ew e sw, a avaliacao final inclui:
- Holdout RMSE
- Holdout MAE
- Holdout R2
- Holdout IC95% de RMSE, MAE e R2 por bootstrap

Avaliacao interna continua sendo reportada por abordagem (CV, EW, SW).

## Onde os Artefatos Sao Salvos

### Parametros do XGBoost
- Scripts/Results/xgboost/cv/params/*_xgb_params_cv.json
- Scripts/Results/xgboost/ew/params/*_xgb_params_ew.json
- Scripts/Results/xgboost/sw/params/*_xgb_params_sw.json

### Logs de execucao
Padrao atual por familia/abordagem:
- Scripts/Results/<familia>/<abordagem>/runs/<timestamp>/logs/*.log

### Sumarios do orquestrador
- Scripts/Results/runs/<timestamp>/summary.json
- Scripts/Results/runs/<timestamp>/summary.csv

## Dependencias
As dependencias estao em:
- Utils/requirements.txt

Instalacao:

```bash
pip install -r Utils/requirements.txt
```

## Reproducao Rapida

1. Instalar dependencias.
2. Garantir dados em Data/.
3. Gerar dados limpos em Scripts/ModelData/.
4. Rodar run_all_models_tracks.py para comparacao em lote.
5. Consolidar resultados pelos logs e sumarios em Scripts/Results/.

## Status Atual
- Pipeline funcional para 5 GPs: Bahrain, Hungary, Italy, Saudi Arabia, United States.
- Comparacao ativa entre familias de modelos e abordagens temporais.
- Estrutura de resultados reorganizada e padronizada.
- IC95% no holdout implementado para cv, ew e sw.

## Autores
- Marcos Paulo de Oliveira Pereira
- Alexandre Magno de Sousa

UFOP - Engenharia de Computacao

## Licenca
Uso academico.
