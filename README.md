# 🏁 TCC Fórmula 1 - Análise Preditiva e Otimização de Estratégias

## 📋 Título da Pesquisa

**Análise Multicircuito de Tempos de Volta na Fórmula 1: Comparação de Modelos Preditivos e Extração de Insights Estratégicos**

---

## 🎯 Objetivo

Este projeto de Trabalho de Conclusão de Curso (TCC) visa desenvolver modelos de **Machine Learning para prever tempos de volta** em corridas de Fórmula 1 baseado em dados históricos (2022-2025). Os modelos incorporam informações de telemetria, condições climáticas, características dos pneus e estratégias de corrida para otimizar o desempenho dos pilotos.

### Objetivos Específicos

1. **Coleta e normalização de dados** multitemporais de corridas F1 via API FastF1
2. **Análise exploratória** de padrões de degradação de pneus, tempos de volta e estratégias
3. **Desenvolvimento de modelos preditivos** usando diferentes algoritmos (Linear Regression, XGBoost, Random Forest)
4. **Otimização de hiperparâmetros** com Optuna
5. **Análise de importância de variáveis** usando SHAP
6. **Comparação de desempenho** entre modelos para identificar a melhor abordagem

---

## 📊 Dados do Projeto

### Grande Prêmios Analisados

Atualmente, o projeto analisa **5 grandes prêmios** em 4 temporadas (2022-2025):

- 🇧🇭 **Bahrein Grand Prix** (Deserto, alta temperatura)
- 🇭🇺 **Hungarian Grand Prix** (Circuito entre-muros, mudanças de tempo)
- 🇮🇹 **Italian Grand Prix** (Circuito de alta velocidade)
- 🇸🇦 **Saudi Arabian Grand Prix** (Rua, noturno)
- 🇺🇸 **United States Grand Prix** (Circuito de uso misto)

### Tipos de Dados Coletados

Para cada Grande Prêmio, em cada ano coletamos:

#### 📈 Dados de Corridas (Race)
- **Laps:** Tempo de volta, composto de pneu, número do stint, telemetria GPS
- **Weather:** Temperatura da pista/ar, umidade, pressão, velocidade do vento
- **Results:** Posição final, status do piloto (finished/DNF), pontos

#### 🏁 Dados de Treinos Livres (Free Practice - opcional)
- Mesmos dados de laps e weather para análise comparativa
- Útil para prever desempenho do fim de semana

### Estrutura do Diretório de Dados

```
Data/
├── Bahrain/
│   ├── Race/
│   │   ├── Laps/
│   │   │   ├── bahrain_grand_prix_laps_2022.csv
│   │   │   ├── bahrain_grand_prix_laps_2023.csv
│   │   │   ├── bahrain_grand_prix_laps_2024.csv
│   │   │   └── bahrain_grand_prix_laps_2025.csv
│   │   └── Weather/
│   │       └── [arquivos de weather correspondentes]
│   ├── Free Practice/ (opcional)
│   └── Results/
├── Hungary/
├── Italy/
├── Saudi Arabia/
└── United States/
```

### Mapeamento de Compostos de Pneus

Um desafio fundamental na análise multitemporal de F1 é que os **rótulos relativos de pneus** (Soft, Medium, Hard) mudam conforme o circuit e o ano. Para garantir que o modelo aprenda a **física real da degradação de borracha**, utilizamos um arquivo de configuração:

📄 **`compounds.json`** - Mapeia rótulos genéricos (Soft/Medium/Hard) para **compostos Pirelli específicos (C1-C6)**

Exemplo:
```json
{
  "data": {
    "2024": {
      "Bahrain Grand Prix": {
        "Hard": "C2",
        "Medium": "C3",
        "Soft": "C4"
      }
    }
  }
}
```

Isso permite que o modelo generalize: *"pneus C3 degradam de forma X"* independente de serem chamados "Medium" ou "Soft" em uma determinada corrida.

---

## 🏗️ Estrutura do Projeto

A estrutura de diretórios foi cuidadosamente reorganizada para separar a obtenção de dados via API, transformações em notebooks (EDA) e a execução produtiva dos modelos:

```
TCC/
├── Bibliografia/               # Artigos e referências do TCC
├── Data/                       # Dados brutos obtidos do FastF1
│   ├── Bahrain/                # Organizado por país -> Sessão -> Variável
│   │   ├── Race/               
│   │   │   ├── Laps/           # bahrain_grand_prix_laps_2022.csv ...
│   │   │   └── Weather/        # bahrain_grand_prix_weather_2022.csv ...
│   │   ├── Free Practice/
│   │   └── Results/            
│   ├── Hungary/                
│   ├── Italy/
│   ├── Saudi Arabia/
│   └── United States/
├── Scripts/                    # Todo o código fonte e desenvolvimento
│   ├── Notebooks/              # Exploração de dados e visualização (EDA)
│   │   ├── Notebook_Bahrain.ipynb
│   │   └── Notebook_USA.ipynb ...
│   ├── Source/                 # Scripts Python modulares para pipeline final
│   │   ├── script_model_data.py       # Extração de outliers, transformações RBF e merge do CSV Limpo
│   │   ├── model_lr_baseline.py       # Regressão Linear Simples
│   │   ├── model_lr_crossvalidation.py# Regressão Linear com K-Fold robusto
│   │   ├── model_lr_wf.py             # Regressão Linear Walk-Forward (por voltas)
│   │   ├── model_xgb_cv.py            # XGBoost com K-Fold e Tuning via Optuna
│   │   └── model_xgb_wf.py            # XGBoost via Walk-Forward usando hiperparâmetros tunados
│   ├── ModelData/              # [NOVO] CSVs totalmente limpos gerados pelo script_model_data.py
│   │   └── Bahrain Grand Prix/
│   │       └── bahrain_grand_prix_cleaned_data.csv
│   ├── Utils/                  # [NOVO] Artefatos salvos pelos modelos (ex: hyperparams em JSON)
│   │   └── bahrain_grand_prix_xgb_params.json
│   └── compounds.json          # Mapeamento oficial dos compostos C1-C5
├── .gitignore                  
├── README.md                   
└── requirements.txt            
```

---

## 🚀 Como Executar o Pipeline

O novo fluxo de trabalho é totalmente modular. Siga esta ordem:

1. **Pré-Processamento dos Dados:**
   Abra e edite a variável `target_gp_name` (ex: `'Bahrain Grand Prix'`) no arquivo de tratamento, em seguida execute a limpeza. Ele criará o `.csv` definitivo na pasta `ModelData/`.
   ```bash
   python Scripts/Source/script_model_data.py
   ```

2. **Modelos Baseline e Walk-Forward (Linear Regression):**
   Com os dados limpos disponíveis, teste a regressão linear:
   ```bash
   python Scripts/Source/model_lr_baseline.py
   python Scripts/Source/model_lr_crossvalidation.py
   python Scripts/Source/model_lr_wf.py
   ```

3. **Otimização Avançada (XGBoost + Optuna):**
   Execute o treinamento XGBoost. Ele fará o _tuning_ do Optuna e **salvará** automaticamente os melhores parâmetros num `.json` dentro da pasta `Scripts/Utils/`, além de rodar os splits da validação cruzada.
   ```bash
   python Scripts/Source/model_xgb_cv.py
   ```

4. **Validação Walk-Forward (XGBoost):**
   Por fim, teste a robustez cronológica do XGBoost. Ele lerá os hiperparâmetros salvos previamente no JSON para treinar árvores crescentes (Walk Forward Validation).
   ```bash
   python Scripts/Source/model_xgb_wf.py
   ```
├── README.md                                    # Este arquivo
├── requirements.txt                             # Dependências Python
├── scripts/                                     # Coleta de dados
│   ├── script.py                               # Script principal de coleta (raças)
│   ├── script_treino                           # Script para coleta de treinos
│   ├── compounds.json                          # Mapeamento de compostos Pirelli
│   ├── Notebook_Bahrain.ipynb                 # Análise completa: Bahrein
│   ├── Notebook_Hungria.ipynb                 # Análise: Hungria
│   ├── Notebook_Italia.ipynb                  # Análise: Itália
│   ├── Notebook_Saudi.ipynb                   # Análise: Arábia Saudita
│   ├── Notebook_USA.ipynb                     # Análise: EUA
│   ├── Notebook_Bahrain-Impuro.ipynb          # Análise com dados incompletos
│   ├── f1_plots/                              # Diretório com visualizações geradas
│   └── fastf1_cache/                          # Cache da API FastF1
├── Data/                                        # Dados brutos baixados
│   ├── Bahrain/
│   ├── Hungary/
│   ├── Italy/
│   ├── Saudi Arabia/
│   └── United States/
├── Bibliografia/                                # Referências acadêmicas
│   ├── Deep_Neural_Network-based_lap_time_forecasting_of_.pdf
│   ├── Optimization-of-pit-stop-strategies-in-Formula-1-racing.pdf
│   └── [outros papers]
└── Proposta_TCC___Marcos_Paulo-4.pdf          # Proposta original do TCC
```

---

## 🔧 Dependências

### Requisitos do Sistema
- **Python 3.8+**
- 1GB de RAM mínimo (2GB+ recomendado para processamento de múltiplos anos)
- Conexão com internet (para download via FastF1)

### Pacotes Python

```
fastf1              # API de dados da Fórmula 1
pandas              # Manipulação e análise de dados
numpy               # Computação numérica
matplotlib          # Visualização estática
seaborn             # Visualização estatística
scikit-learn        # Modelos de ML e pré-processamento
plotly              # Visualização interativa
scipy               # Computação científica
statsmodels         # Análise estatística
xgboost             # Modelo XGBoost (gradient boosting)
optuna              # Otimização de hiperparâmetros
shap                # Análise de importância de features (SHAP values)
```

---

## ⚙️ Instalação e Configuração

### 1. Clone o Repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd TCC-Formula1
```

### 2. Crie um Ambiente Virtual (Python)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Replicar a Pesquisa

### Fase 1: Coleta de Dados

#### Opção A: Coletar Dados Automaticamente (Recomendado)

Execute o script principal para baixar dados de corridas:

```bash
cd Scripts
python script.py
```

**O que ele faz:**
1. Conecta à API FastF1
2. Baixa dados de cada Grande Prêmio (2022-2025)
3. Normaliza compostos de pneus via `compounds.json`
4. Salva dados em `/Data/[Pais]/Race/Laps/` e `/Weather/`
5. Cache automático em `fastf1_cache/`

**Tempo estimado:** 5-15 minutos (dependendo da conexão)

#### Opção B: Coletar Dados de Treinos Livres (Optional)

```bash
python script_treino
```

Coleta dados de FP1, FP2, FP3 para análise adicional.

#### Opção C: Usar Dados Já Fornecidos

Se o diretório `/Data/` já contiver CSVs, os notebooks carregarão automaticamente os dados locais.

### Fase 2: Análise Exploratória (EDA)

Abra e execute os notebooks Jupyter correspondentes a cada pista:

```bash
# Exemplo: Análise do Grand Prix do Bahrein
jupyter notebook Scripts/Notebook_Bahrain.ipynb
```

Cada notebook realiza:

1. **Carregamento de Dados**
   - Integração de dados de laps, weather e results
   - Conversão de tipos de dados (timedelta, números)
   - Tratamento de valores faltantes

2. **Análise Exploratória (EDA)**
   - Distribuição de tempos de volta
   - Tendências temporais ao longo da prova
   - Impacto da meteorologia no desempenho
   - Degradação de pneus por stint

3. **Visualizações**
   - Gráficos de tempos vs volta
   - Heatmaps de temperatura/umidade
   - Box plots de pneus vs tempos
   - Curvas de degradação

4. **Modelagem Preditiva**

   **Modelos testados:**
   - Linear Regression (baseline)
   - XGBoost (gradient boosting)
   - Random Forest (ensemble)

   **Pipeline:**
   ```
   Dados brutos → Tratamento → Features → Scaling → Modelo → Previsões
   ```

5. **Otimização de Hiperparâmetros**
   - Usa Optuna para buscar automatically melhores parâmetros
   - Cross-validation com KFold (k=5)
   - Métrica: MAE (Mean Absolute Error)

6. **Análise de Importância (SHAP)**
   - Identifica quais variáveis mais influenciam as previsões
   - Gera gráficos SHAP force plots e dependence plots
   - Explainability: por que cada previsão foi feita

7. **Persistência**
   - Salva modelos em arquivos .pkl
   - Exporta análises em PNG/HTML

### Fase 3: Comparação e Interpretação

Após rodar todos os notebooks:

1. **Compare resultados** entre Grande Prêmios (quais são mais previsíveis?)
2. **Identifique padrões globais** (variáveis consistentemente importantes)
3. **Analise outliers** (voltas inesperadas, mudanças climáticas abruptas)

---

## 📈 Fluxo de Análise Detalhado (Entre Notebooks)

```
Coleta de Dados (FastF1)
    ↓
Carregamento em Pandas
    ↓
Limpeza e Normalização (compostos, tipos de dados)
    ↓
Análise Exploratória (EDA)
    ├─ Visualizações
    ├─ Correlações
    └─ Testes estatísticos
    ↓
Feature Engineering
    ├─ Lap number, stint number
    ├─ Condições climáticas
    ├─ Histórico de pneus
    └─ Variáveis derivadas
    ↓
Modelagem
    ├─ Linear Regression (baseline)
    ├─ XGBoost (otimizado via Optuna)
    └─ Random Forest
    ↓
Avaliação
    ├─ MSE, MAE, R²
    ├─ Análise de resíduos
    └─ Teste estatístico (Shapiro-Wilk)
    ↓
Explainability (SHAP)
    ├─ Feature importance
    ├─ Force plots
    └─ Dependence plots
    ↓
Relatórios e Visualizações
```

---

## 🔍 Descrição dos Notebooks

### Notebook_Bahrain.ipynb
**Completo, detalhado, sem problemas de dados**
- Modelo preditivo multi-season (2022-2025)
- Análise de degradação por stint
- Otimização com Optuna
- SHAP explainability total
- Previsões vs realidade

### Notebook_Hungria.ipynb, Notebook_Italia.ipynb, etc.
Mesmo fluxo, adaptar para cada pista (diferentes compostos de pneu, layouts).

### Notebook_Bahrain-Impuro.ipynb
**Versão com dados incompletos** - para estudar robustez do modelo com missing values.

---

## 📊 Variáveis Principais Utilizadas

### Features de Entrada (X)

| Variável | Tipo | Descrição | Fonte |
|----------|------|-----------|--------|
| `LapNumber` | int | Número da volta na prova | Laps data |
| `StintNumber` | int | Seu número do stint atual | Laps data |
| `Compound` | category | Pneu: Soft, Medium, Hard | Laps data |
| `pirelliCompound` | category | Composto Pirelli (C1-C6) normalizado | compounds.json |
| `AirTemp` | float | Temperatura do ar (°C) | Weather data |
| `TrackTemp` | float | Temperatura da pista (°C) | Weather data |
| `Humidity` | float | Umidade relativa (%) | Weather data |
| `Pressure` | float | Pressão atmosférica (mb) | Weather data |
| `WindSpeed` | float | Velocidade do vento (m/s) | Weather data |

### Alvo (y)

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `LapTime` | timedelta → float | Tempo da volta em segundos |

---

## 🎯 Métricas de Desempenho

Os modelos são avaliados com:

- **MAE (Mean Absolute Error):** Erro médio absoluto em segundos
- **RMSE (Root Mean Squared Error):** Raiz do erro quadrático médio
- **R² Score:** Proporção da variância explicada pelo modelo
- **Residuals Test (Shapiro-Wilk):** Normalidade dos resíduos

### Exemplo de Resultado Esperado

```
XGBoost Model Performance:
  MAE:  0.45 segundos
  RMSE: 0.63 segundos
  R²:   0.87

Interpretação: O modelo prevê tempos de volta com erro médio de ~0.45 seg,
explicando 87% da variância nos tempos.
```

---

## 🧬 Técnicas de Machine Learning Utilizadas

### 1. **Regressão Linear**
- Modelo baseline simples
- Interpretável
- Rápido

### 2. **XGBoost (eXtreme Gradient Boosting)**
- Ensemble iterativo de árvores de decisão
- Otimizado para regressão
- Melhor desempenho geral
- Suporta importância de features nativa

### 3. **Random Forest**
- Ensemble de árvores independentes
- Mais robusto a outliers
- Menos overfitting que XGBoost

### 4. **Otimização de Hiperparâmetros (Optuna)**

Busca automática dos melhores parâmetros:
- `max_depth`: profundidade da árvore
- `learning_rate`: taxa de aprendizado
- `n_estimators`: número de árvores
- Métrica otimizada: MAE (erro absoluto médio)

### 5. **SHAP (SHapley Additive exPlanations)**

Explica cada previsão:
- **Feature Importance:** Quais features mais importam?
- **Force Plot:** Como cada feature influencia cada previsão
- **Dependence Plot:** Relação não-linear entre feature e output

---

## 📁 Como Organizar seus Próprios Dados

Se quiser **adicionar mais pistas** ou **anos**:

### 1. Adicione ao Script de Coleta

```python
# Em script.py
target_gp_name = 'New Grand Prix Name'  # Nome exato do calendário FastF1

# Adicione os compostos em compounds.json
{
  "data": {
    "2024": {
      "New Grand Prix Name": {
        "Hard": "C1",
        "Medium": "C2",
        "Soft": "C3"
      }
    }
  }
}
```

### 2. Crie a Estrutura de Diretórios

```bash
mkdir -p Data/YourCountry/Race/Laps
mkdir -p Data/YourCountry/Race/Weather
mkdir -p Data/YourCountry/Results
```

### 3. Execute o Script Modificado

```bash
python script.py
```

Os dados serão automaticamente salvos nas pastas corretas.

---

## 🐛 Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'fastf1'"

**Solução:**
```bash
pip install fastf1
# ou atualizar
pip install --upgrade fastf1
```

### Problema: Erro de permissão ao criar cache

**Solução:**
```bash
# Certifique-se de ter permissão de escrita no diretório
chmod 755 Scripts/
# Ou delete o cache e deixe criar novo
rm -rf Scripts/fastf1_cache/
```

### Problema: A API FastF1 retorna erro 429 (Rate Limit)

**Solução:** Os scripts já incluem tratamento automático, mas se persistir:
```python
import time
time.sleep(5)  # Aguarde 5 segundos entre requisições
```

### Problema: Dados faltando para um determinado ano/pista

**Solução:**
1. Verifique se a corrida existiu naquele ano (Calendário F1)
2. Verifique a nomenclatura exata em `schedule[schedule.columns]`
3. Consulte fastf1 documentation: https://docs.fastf1.dev/

---

## 📚 Referências Bibliográficas

Consulte a pasta `/Bibliografia/` para papers relacionados:

- **Deep Neural Network-based lap time forecasting**
- **Optimization of pit-stop strategies in Formula 1**
- **Machine Learning applications in motorsport**
- E mais artigos acadêmicos

---

## 🤝 Como Contribuir

Se encontrar bugs ou quiser melhorar o projeto:

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/sua-feature`
3. Commit suas mudanças: `git commit -m 'Adiciona nova funcionalidade'`
4. Push: `git push origin feature/sua-feature`
5. Abra um Pull Request

---

## 📝 Autor

**Marcos Paulo de Oliveira Pereira**
**Alexandre Magno de Sousa** 
Trabalho de Conclusão de Curso - UFOP  
Engenharia de Computação

---

## 📄 Licença

Este projeto é fornecido como material acadêmico. Todos os dados públicos de F1 provêm da API FastF1.

---

## ✨ Próximos Passos / Melhorias Futuras

- [ ] Validar a abordgem ideal para construção do modelo, através de comparações e justificativas na literatura
- [ ] Integrar dados de treino e classificação como uma informação importante para o modelo
- [ ] Integrar o modelo ideal à uma simulação para cada corrida
- [ ] Execução de cenários e análise de estratégias
- [ ] Criar API web para fazer previsões em tempo real

---

**Última atualização:** Abril 2026  
**Status:** Em desenvolvimento ✅
