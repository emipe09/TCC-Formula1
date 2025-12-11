# 🏎️ Análise Preditiva e Otimização de Estratégias na Fórmula 1 Utilizando Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento-yellow)
![Scope](https://img.shields.io/badge/Escopo-5_Pistas-blueviolet)

Este repositório contém o código fonte e as análises desenvolvidas para a Monografia **"Análise Preditiva e Otimização de Estratégias na Fórmula 1 Utilizando Aprendizado de Máquina"**.

O projeto propõe uma abordagem generalista de Ciência de Dados para prever tempos de volta (*lap times*) para auxiliar na simulação de corridas e otimização de estratégias. Para garantir a robustez e a capacidade de generalização do modelo, o estudo abrange **5 circuitos distintos** do calendário da Fórmula 1, com características aerodinâmicas e de degradação variadas.

---

## 🚧 Status do Projeto

Atualmente, o pipeline de análise e modelagem foi **completamente implementado e validado para o Grande Prêmio do Bahrein (Sakhir)**. A expansão para os demais 4 circuitos está em andamento.

| Circuito  Status |
| :--- | :--- |
| **🇧🇭 GP do Bahrein (Sakhir)** |✅ **Concluído** |
| **Circuitos 2-5** |🔄 *Em Breve* |

---

## 📋 Sobre o Projeto

A estratégia na Fórmula 1 é um problema de otimização sob incerteza. Este projeto visa isolar as variáveis físicas (degradação de pneus, consumo de combustível, clima) das variáveis de contexto para criar modelos preditivos que funcionem em diferentes pistas.

**Destaques Técnicos:**
* **Metodologia Escalável:** O código foi estruturado para ser replicado em qualquer pista com ajustes mínimos.
* **Coleta de Dados:** Extração automatizada via API [FastF1](https://github.com/theOehrly/Fast-F1).
* **Engenharia de Features:** Transformação RBF (Radial Basis Function) para dados climáticos (multimodais), PCA, Clusterização K-Means, remoção de outliers.
* **Seleção de Modelos:** Algoritmo de *Backward Elimination* otimizado via critério de informação **Mallows' $C_p$**.

---

## 📊 Estudo de Caso I: GP do Bahrein

Ainda explorando.
---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Bibliotecas:** `pandas`, `numpy`, `fastf1`, `statsmodels` (Inferência), `scikit-learn` (Machine Learning), `matplotlib`, `seaborn`, `scipy`.

---

## 🚀 Como Executar

Ainda explorando.
---

## 👨‍💻 Autores

* **Marcos P. O. Pereira** - *Desenvolvimento e Pesquisa*
* **Alexandre M. Souza** - *Orientador*

---

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
