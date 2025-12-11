# 🏎️ Análise Preditiva e Otimização de Estratégias na Fórmula 1 Utilizando Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento-yellow)
![Scope](https://img.shields.io/badge/Escopo-5_Pistas-blueviolet)

Este repositório contém o código fonte e as análises desenvolvidas para a Monografia **"Análise Preditiva e Otimização de Estratégias na Fórmula 1 Utilizando Aprendizado de Máquina"**.

O projeto propõe uma abordagem generalista de Ciência de Dados para prever tempos de volta (*lap times*) e simular cenários estratégicos. Para garantir a robustez e a capacidade de generalização do modelo, o estudo abrange **5 circuitos distintos** do calendário da Fórmula 1, com características aerodinâmicas e de degradação variadas.

---

## 🚧 Status do Projeto

Atualmente, o pipeline de análise e modelagem foi **completamente implementado e validado para o Grande Prêmio do Bahrein (Sakhir)**. A expansão para os demais 4 circuitos está em andamento.

| Circuito | Características | Status |
| :--- | :--- | :--- |
| **🇧🇭 GP do Bahrein (Sakhir)** | Alta degradação (abrasivo), foco em tração. | ✅ **Concluído** |
| **Circuitos 2-5** | Variedade de *downforce* e clima. | 🔄 *Em Breve* |

---

## 📋 Sobre o Projeto

A estratégia na Fórmula 1 é um problema de otimização sob incerteza. Este projeto visa isolar as variáveis físicas (degradação de pneus, consumo de combustível, clima) das variáveis de contexto para criar modelos preditivos que funcionem em diferentes pistas.

**Destaques Técnicos:**
* **Metodologia Escalável:** O código foi estruturado para ser replicado em qualquer pista com ajustes mínimos.
* **Coleta de Dados:** Extração automatizada via API [FastF1](https://github.com/theOehrly/Fast-F1).
* **Engenharia de Features:** Transformação RBF (Radial Basis Function) para dados climáticos e Clusterização K-Means.
* **Seleção de Modelos:** Algoritmo de *Backward Elimination* otimizado via critério de informação **Mallows' $C_p$**.

---

## 📊 Estudo de Caso I: GP do Bahrein

Os resultados abaixo referem-se à validação inicial no circuito de Sakhir, servindo como prova de conceito da metodologia.

### Análise Exploratória (EDA)
* Identificação de *outliers* estratégicos (Safety Car, VSC) via **Intervalo Interquartil (IQR)**.
* Correlação robusta detectada entre a idade do pneu (`TyreLife`) e o aumento do tempo de volta, validando a física do modelo.

### Modelagem Preditiva
Foram desenvolvidos dois modelos de regressão para prever o tempo da próxima volta (`LapTime_next`):

1.  **Modelo Autoregressivo (Baseline):**
    * Utiliza o tempo anterior (`LapTime_prev`).
    * **$R^2 \approx 0.93$**.
2.  **Modelo Físico/Estratégico (Otimizado):**
    * Utiliza apenas estado do carro e clima (sem histórico imediato).
    * Seleção de variáveis via **Mallows' $C_p$** (redução de 55 para ~24 features).
    * **$R^2 \approx 0.75$**.
    * *Insight:* Permite simular cenários de longo prazo ("undercut", "overcut") sem depender do tempo da volta anterior.

---

## 📈 Resultados Preliminares (Bahrein)

### Seleção de Variáveis
A técnica de eliminação retroativa provou ser eficaz para limpar ruídos estatísticos do dataset.

![Seleção de Variáveis](imagens/mallows_cp_plot.png)
*O gráfico demonstra o ponto ótimo de complexidade do modelo (Mínimo $C_p$).*

### Interpretação dos Coeficientes
O modelo quantificou a física da corrida em Sakhir:
* **Degradação:** +0.19s por volta de desgaste.
* **Combustível:** -0.33s por volta devido à perda de peso.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Bibliotecas:** `pandas`, `numpy`, `fastf1`, `statsmodels` (Inferência), `scikit-learn` (Machine Learning), `matplotlib`, `seaborn`, `scipy`.

---

## 🚀 Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/f1-strategy-ml.git](https://github.com/seu-usuario/f1-strategy-ml.git)
    cd f1-strategy-ml
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Notebooks:**
    * Os notebooks na pasta `/notebooks` seguem a ordem lógica: Coleta -> EDA -> Modelagem. Atualmente focados no dataset do Bahrein.

---

## 👨‍💻 Autores

* **Marcos P. O. Pereira** - *Desenvolvimento e Pesquisa*
* **Alexandre M. Souza** - *Orientador*

---

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
