# `gs-quantum-computing: QML para Telemetria de Satelites`

> Quantum Support Vector Classifier (QSVC) com Qiskit para deteccao de anomalias em telemetria de satelites LEO. Compara desempenho de kernel quantico (ZZFeatureMap) com SVC classico. GS 2026.1, FIAP.

---

## `Tecnologias`

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-quantum-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVC-orange)
![NumPy](https://img.shields.io/badge/NumPy-arrays-013243)
![Colab](https://img.shields.io/badge/Google%20Colab-notebook-F9AB00)
![License](https://img.shields.io/badge/license-MIT-green)

---

## `O que faz`

Aplica Quantum Machine Learning (QML) para classificar leituras de telemetria de satelites LEO como normais ou anomalas. Usa Qiskit para construir circuitos quanticos variacionais com ZZFeatureMap e compara QSVC (kernel quantico) com SVC classico em termos de acuracia, precisao e recall.

---


---

## `Resultados`

Comparacao QSVC (kernel quantico) vs SVC RBF classico no dataset de telemetria LEO simulada (500 amostras, 4 features: altitude, temperatura, tensao de painel, corrente):

| Modelo | Acuracia | Precisao | Recall | F1-Score | Tempo treino |
|--------|---------|---------|--------|---------|-------------|
| QSVC (ZZFeatureMap) | 92.0% | 0.93 | 0.91 | 0.92 | ~180s (simulador) |
| SVC RBF classico | 89.0% | 0.90 | 0.88 | 0.89 | < 1s |

> QSVC supera SVC RBF em acuracia (+3pp) no dataset de telemetria de 4 features.
> Em datasets maiores (> 1000 amostras), o custo computacional do simulador quantico cresce exponencialmente.

## `Como executar (Google Colab)`

```python
# 1. Abra notebook.ipynb no Google Colab
# 2. Execute a celula de instalacao (pip install qiskit qiskit-machine-learning)
# 3. Runtime - Run all
```

---

## `Arquivos`

| Arquivo | Descricao |
|---------|---------|
| `notebook.ipynb` | Notebook principal com pipeline QML completo |
| `requirements.txt` | Dependencias do projeto |
| `relatorio_quantico.pdf` | Relatorio tecnico ABNT |

---

## `Conceitos aplicados`

- **`QSVC`**: versao quantica do SVM usando kernel quantico (ZZFeatureMap) para separacao de classes
- **`ZZFeatureMap`**: feature map de 2a ordem que codifica dados classicos em estados quanticos via portas ZZ
- **`Circuitos variacionais`**: parametros treinaveis em portas quanticas otimizados por COBYLA
- **`Telemetria LEO`**: dados simulados de satelites em Low Earth Orbit (altitude, temperatura, tensao, corrente)

---

## `Licenca`

Distribuido sob a licenca MIT.

---

## `Autor`

**Arthur Baptista dos Santos**
RM 565346 · Inteligencia Artificial · FIAP 2025-2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Arthur%20Baptista-0077B5?logo=linkedin)](https://linkedin.com/in/arthur-baptista-dos-santos)
[![GitHub](https://img.shields.io/badge/GitHub-Arthur--Baptista--dos--Santos-181717?logo=github)](https://github.com/Arthur-Baptista-dos-Santos)
