#  GS 2026.1 — Quantum Machine Learning para Telemetria de Satélites

**FIAP | Turma 2TIAP | Global Solution 2026.1**
**Disciplina:** Computação Quântica e IA
**Grupo:** NeuroQuantum
**ODS:** 9 · 11 · 13

---

##  Vídeo Demonstrativo

[![SpaceRAG Demo](https://img.youtube.com/vi/v78XVNwJDBg/0.jpg)](https://youtu.be/v78XVNwJDBg)

> Clique na imagem para assistir à demonstração completa no YouTube.

---

##  Sobre o Projeto

Aplicação de **Quantum Machine Learning (QML)** para detecção de anomalias em dados de telemetria de satélites LEO (Low Earth Orbit), utilizando o algoritmo **QSVC** (Quantum Support Vector Classifier) com `ZZFeatureMap` no Qiskit 1.x.

O modelo quântico é comparado com baselines clássicos (SVM-RBF e Random Forest) usando métricas obrigatórias: AUC-ROC, F1-Score, acurácia, precisão, recall, tempo de treino e tempo de inferência.

---

##  Estrutura do Projeto

```
GS2026_CQ_2TIAP_NeuroQuantum/
├── notebook.ipynb       # Notebook Jupyter 100% reproduzível (random_state=42)
├── relatorio.docx       # Relatório técnico completo (6 seções, ABNT)
├── requirements.txt     # Dependências com versões exatas
└── README.md
```

---

##  O que foi implementado

| Requisito | Implementação |
|---|---|
| Dataset espacial | Telemetria sintética NASA MSAD-style (600 registros, 8 sensores) |
| Pré-processamento | MinMaxScaler [-π, π] + PCA (4 qubits) sem data leakage |
| Modelo QML | QSVC com ZZFeatureMap (Qiskit 1.x, 4 qubits, Qiskit Aer) |
| Visualizações | Circuito quântico + Kernel Matrix + Curvas ROC + Matrizes de confusão |
| Baselines clássicos | SVM-RBF e Random Forest |
| Métricas | AUC-ROC, F1, acurácia, precisão, recall, tempo de treino e inferência |
| Reproducibilidade | `random_state=42` em todas as operações |

---

##  Como Executar

### Google Colab
1. Abra o `notebook.ipynb` no [Google Colab](https://colab.research.google.com/)
2. Substitua a célula 1 por:
```python
!pip install qiskit==1.3.2 qiskit-machine-learning==0.8.2 qiskit-aer==0.15.1 qiskit-algorithms==0.3.1 scikit-learn imbalanced-learn pandas numpy matplotlib seaborn scipy
```
3. Reinicie o runtime: **Runtime → Restart session**
4. Execute tudo: **Runtime → Run all**

### Local (VS Code)
```bash
# Criar ambiente virtual
python -m venv venv
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Registrar kernel e abrir notebook
python -m ipykernel install --user --name venv-gs-qml --display-name "GS QML"
jupyter notebook
```

---

##  Resultados Esperados

| Modelo | AUC-ROC | F1-Score | T. Treino |
|---|---|---|---|
| QSVC (QML) | ~0.92 | ~0.87 | ~60–120s |
| SVM-RBF | ~0.95 | ~0.91 | ~0.01s |
| Random Forest | ~0.97 | ~0.93 | ~0.05s |

> Valores exatos gerados pelo notebook com `random_state=42`.

---

##  Referências

- [NASA MSAD](https://nasa.github.io/MSAD/)
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
- Biamonte et al. (2017). *Quantum machine learning*. Nature, 549, 195–202.
- Havlíček et al. (2019). *Supervised learning with quantum-enhanced feature spaces*. Nature, 567, 209–212.
