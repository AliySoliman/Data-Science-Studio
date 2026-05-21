<div align="center">

# 🧠 Stream — AI-Powered Data Science Studio

**An end-to-end, no-code data science platform built for students, researchers, and analysts.**  
Upload your data. Let AI guide you. Build ML pipelines. Generate reports. All in one place.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLMs-black?logo=ollama)](https://ollama.com/)
[![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Feature Walkthrough](#feature-walkthrough)
  - [🚀 Boot & Setup Wizard](#-boot--setup-wizard)
  - [🤖 AI Recommendations](#-ai-recommendations)
  - [⚙️ Data Transformation Pipeline](#️-data-transformation-pipeline)
  - [📊 Analysis & EDA](#-analysis--eda)
  - [🧠 Machine Learning Studio](#-machine-learning-studio)
  - [📈 Statistical Analysis](#-statistical-analysis)
  - [📝 Reporting](#-reporting)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Option A: Docker (Recommended)](#option-a-docker-recommended)
  - [Option B: Run Locally](#option-b-run-locally)
- [Project Structure](#project-structure)
- [How It Works — AI Engine](#how-it-works--ai-engine)
- [Supported Models & Algorithms](#supported-models--algorithms)
- [Pipeline Save & Resume](#pipeline-save--resume)

---

## Overview

**Stream** is a full-featured, interactive data science platform built with [Streamlit](https://streamlit.io/). It allows anyone — regardless of coding experience — to go from raw tabular data to trained machine learning models and professional reports through a guided, visual workflow.

At its core, Stream uses a **dual-LLM AI engine** (powered by locally-run [Ollama](https://ollama.com/) models) to automatically profile your dataset, reason about its structure, and generate actionable, hallucination-guarded advisory reports. The entire pipeline — transformations, models, statistics, and visualizations — can be **saved and resumed** across sessions.

```
Upload CSV/Excel  →  AI Analysis  →  Transform  →  EDA  →  ML Models  →  Report
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Streamlit Frontend                         │
│  ┌─────────────┐  ┌──────────┐  ┌────────┐  ┌──────┐  ┌────────┐  │
│  │ AI Advisor  │  │Transform │  │  EDA   │  │  ML  │  │Reports │  │
│  └──────┬──────┘  └────┬─────┘  └───┬────┘  └──┬───┘  └───┬────┘  │
│         │              │             │           │           │        │
│  ┌──────▼──────────────▼─────────────▼───────────▼───────────▼────┐ │
│  │                  Session State Pipeline (in-memory)             │ │
│  │         Transformations  |  ML Models  |  Stats  |  Visuals     │ │
│  └─────────────────────────────┬───────────────────────────────────┘ │
│                                │ Save / Load (.enc)                  │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
          ┌──────────────────────┴──────────────────────┐
          │               AI Engine                     │
          │  ┌─────────────────────────────────────┐    │
          │  │  PLANNER MODEL (deepseek-r1:1.5b)   │    │
          │  │  Strategy & reasoning                │    │
          │  └─────────────────┬───────────────────┘    │
          │                    │ structured strategy     │
          │  ┌─────────────────▼───────────────────┐    │
          │  │  WRITER MODEL (llama3.2:latest)      │    │
          │  │  Polished Markdown report generation │    │
          │  └─────────────────────────────────────┘    │
          │         Runs locally via Ollama              │
          └─────────────────────────────────────────────┘
```

---

## Feature Walkthrough

### 🚀 Boot & Setup Wizard

Before the app launches, Stream runs an **intelligent boot sequence**:

1. **Ollama Health Check** — Verifies the Ollama service is reachable. If not, displays a live retry screen with instructions.
2. **RAM Detection** — Reads your system's available memory and recommends the best LLM configuration for your hardware.
3. **Auto-Provisioning** — If default models are already installed, setup is skipped entirely (zero-click). Otherwise, a guided wizard allows 1-click recommended setup or advanced customization.
4. **Model Pulling** — Missing models are streamed and pulled directly within the app with a live progress bar.

| RAM Available | Planner Model        | Writer Model          |
|---------------|---------------------|-----------------------|
| < 8 GB        | deepseek-r1:1.5b    | llama3.2:1b           |
| 8–16 GB       | deepseek-r1:1.5b    | llama3.2:latest       |
| 16 GB+        | deepseek-r1:8b      | llama3.2:latest       |

---

### 🤖 AI Recommendations

The AI Advisor is the flagship feature of Stream. It uses a **two-stage LLM pipeline** to analyze your dataset and generate a professional ML advisory report:

**Stage 1 — Planning (DeepSeek-R1)**  
The reasoning model receives a structured dataset profile (shape, dtypes, missing values, class distribution, etc.) and produces a data science *strategy*: which features to engineer, which models to consider, what preprocessing is critical.

**Stage 2 — Writing (LLaMA 3.2)**  
The writer model transforms the strategy into a polished, client-ready Markdown report following strict hallucination-guard rules — it can only reference columns and facts actually present in the dataset.

The report covers:
- 📋 Executive Summary
- 🔍 Dataset Health Report (missingness, duplicates, constants)
- 🔧 Feature Engineering Recommendations
- 🤖 Model Selection Table (with justifications)
- ⚠️ Deployment Risk Assessment
- 📥 PDF Export (via ReportLab)

> **Hallucination-guarded**: The system prompt enforces 21 non-negotiable rules ensuring the AI never invents columns, fabricates statistics, or recommends models not suited to the actual data.

---

### ⚙️ Data Transformation Pipeline

A visual, drag-and-drop-style pipeline builder for preprocessing your data. Each step is named, typed, and stored in the session pipeline. Steps can be **edited**, **deleted**, and **reordered**.

| Category               | Available Operations                                                                 |
|------------------------|--------------------------------------------------------------------------------------|
| **Cleaning**           | Duplicate Handling, Null Handling, Outlier Handling, Rename Columns, Date Validation |
| **Transformation**     | Delete Columns, Computed Columns, Filter Rows, Group By                              |
| **Standardization**    | MinMax, Z-Score, Robust Scaler, Mean Normalization                                   |
| **Encoding**           | Label Encoding, Ordinal Encoding, One-Hot Encoding, Target Encoding, Binary Encoding  |
| **Feature Selection**  | Chi-Squared, RFE (Recursive Feature Elimination), Correlation, Variance, ANOVA       |
| **Dimensionality Reduction** | PCA, t-SNE, UMAP, LDA                                                         |

- Computationally expensive steps (like UMAP/t-SNE) are **cached in-session** to avoid re-execution.
- You can preview the transformed dataframe at any intermediate pipeline step.
- The full pipeline is saved as an **encrypted `.enc` file** and can be reloaded later.

---

### 📊 Analysis & EDA

An interactive visualization dashboard powered by Plotly. Features include:

- Create multiple chart types and arrange them into a dashboard layout.
- Apply **global transformations** to all charts simultaneously (select which pipeline steps to apply).
- Preview the globally transformed data shape and columns before plotting.
- Save, export, and reload visualization configurations from the project pipeline.

---

### 🧠 Machine Learning Studio

A full-featured ML experimentation environment with a card-based model browser. Models are organized by category and a "Show only performed models" filter lets you focus on what you've configured.

**Regression:**
| Standard                     | Ensemble                        |
|------------------------------|---------------------------------|
| Linear Regression            | Random Forest Regression        |
| KNN Regressor                | Gradient Boosting Regressor     |
| Support Vector Regression    | XGBoost Regressor               |

**Classification:**
| Linear & Statistical          | Tree-Based                      | Neighbors     |
|------------------------------|---------------------------------|---------------|
| Logistic Regression          | Decision Tree Classifier        | KNN Classifier|
| Naive Bayes Classifier       | Random Forest Classifier        |               |
| Support Vector Machine       | Gradient Boosting Classifier    |               |
|                              | XGBoost Classifier              |               |

**Unsupervised:**
- K-Means Clustering
- Hierarchical Clustering
- Unsupervised KNN Clustering
- DBSCAN

Each model page supports configuring hyperparameters, running training, and viewing evaluation metrics — all of which are stored in the session pipeline and can be included in reports.

---

### 📈 Statistical Analysis

A flexible statistical analysis module allowing you to build a pipeline of statistical tests:

| Analysis Type              | Description                                          |
|----------------------------|------------------------------------------------------|
| **Descriptive Statistics** | Mean, median, std, skewness, kurtosis, quartiles     |
| **Two-Sample T-Test**      | Compare means across two groups                      |
| **Confidence Interval**    | Bootstrap or parametric interval estimation          |
| **Correlation & Regression** | Pearson/Spearman correlation, simple linear regression |

Statistical steps can be added, edited, deleted, and previewed in a collapsible results dashboard.

---

### 📝 Reporting

A composable report builder that aggregates results from across the app:

- Pull in ML model metrics, statistical analyses, and visualizations into a single structured report.
- **Reorder report blocks** with move-up/move-down controls.
- Each report asset is **cached in-session** to avoid re-computation on re-renders.
- Supports text, dataframe, Matplotlib, Plotly, and composite (figure + per-class breakdown) asset types.
- PDF export via **ReportLab** and **fpdf2**.

---

## Technology Stack

| Layer                | Technologies                                                              |
|----------------------|---------------------------------------------------------------------------|
| **Frontend**         | Streamlit 1.44, streamlit-option-menu, streamlit-elements                 |
| **AI / LLMs**        | Ollama (DeepSeek-R1, LLaMA 3.2, Mistral, Phi-3), Google Gemini           |
| **Data**             | Pandas 2.2, NumPy 1.26, PyArrow                                           |
| **Machine Learning** | Scikit-Learn 1.7, XGBoost 3.2, TensorFlow 2.19, Keras 3.9               |
| **Visualization**    | Plotly 6, Matplotlib 3.8, Seaborn 0.13, Altair 5.5                       |
| **Statistics**       | SciPy 1.13, Statsmodels 0.14, Lifelines 0.30                             |
| **Dimensionality**   | UMAP-Learn, Scikit-Learn (PCA, t-SNE, LDA)                               |
| **Reporting**        | ReportLab, fpdf2, PyMuPDF                                                 |
| **Containerization** | Docker, Docker Compose                                                    |
| **Security**         | Cryptography (Fernet) for encrypted pipeline saves                        |

---

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) & Docker Compose  **OR** Python 3.11+
- [Ollama](https://ollama.com/download) installed and running (`ollama serve`)

---

### Option A: Docker (Recommended)

Docker Compose handles everything — the app container and routing to your host Ollama instance.

**1. Clone the repository**
```bash
git clone https://github.com/lWAHBAl/Grad-project.git
cd Grad-project
```

**2. Make sure Ollama is running on your host machine**
```bash
ollama serve
```

**3. Build and start the app**
```bash
docker-compose up -d --build
```

**4. Open the app**
```
http://localhost:8501
```

> **How it works:** The `docker-compose.yml` sets `OLLAMA_HOST=http://host.docker.internal:11434` and adds the `host.docker.internal` extra host, so the container can reach the Ollama service running natively on your machine.

**Stop the app:**
```bash
docker-compose down
```

---

### Option B: Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/lWAHBAl/Grad-project.git
cd Grad-project/Stream
```

**2. Create and activate a virtual environment** *(optional but recommended)*
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start Ollama** *(in a separate terminal)*
```bash
ollama serve
```

**5. Run the app**
```bash
streamlit run scripts/main.py
```

**6. Open the app**
```
http://localhost:8501
```

> **First run:** The setup wizard will detect your RAM and guide you through downloading the recommended LLM models. This is a one-time step.

---

## Project Structure

```
Grad-project/
├── docker-compose.yml               # Compose file: app container + Ollama host routing
├── README.md
└── Stream/
    ├── Dockerfile                   # Python 3.11-slim, exposes port 8501
    ├── requirements.txt             # All Python dependencies (pinned)
    ├── .dockerignore
    ├── save files/                  # Persisted pipelines, datasets, user config
    │   ├── save.enc                 # Encrypted pipeline save file
    │   ├── user_config.json         # LLM model role configuration
    │   ├── diabetes.csv             # Example dataset
    │   └── study_performance.csv    # Example dataset
    └── scripts/                     # Application source code
        ├── main.py                  # 🚪 Entry point — navigation bar, page routing
        ├── setup_lib.py             # 🔌 Boot sequence: Ollama check, model provisioning
        ├── user_config.py           # 💾 Load/save user model role config
        ├── constants.py             # 🎨 DataManager, shared CSS, UI constants
        ├── transform.py             # ⚙️ Data transformation pipeline builder
        ├── visualizations.py        # 📊 EDA dashboard & global transformations
        ├── machine.py               # 🧠 ML model browser & experiment runner
        ├── stats.py                 # 📈 Statistical analysis pipeline
        ├── reporting.py             # 📝 Composable report builder
        ├── save.py                  # 🔐 Encrypted pipeline save/load (Fernet)
        ├── pipeline_executor.py     # ▶️ Core pipeline execution engine
        ├── mapping_tables.py        # 🗺️ Column/type mapping utilities
        ├── ai_recommendations_lib/  # 🤖 AI advisor system
        │   ├── __init__.py          #    Module entry point & main run() function
        │   ├── config.py            #    LLM model config & system prompts
        │   ├── llm_integration.py   #    Dual-LLM client (planner + writer)
        │   ├── data_profiling.py    #    Dataset health & statistics profiler
        │   ├── report_generator.py  #    Report section orchestration
        │   ├── pdf_generator.py     #    PDF export with ReportLab
        │   ├── visuals.py           #    EDA charts for the advisory report
        │   ├── state_managers.py    #    Streamlit session state helpers
        │   └── AI_gemini_advisor.py #    Google Gemini integration
        ├── transformations/         # Transformation execution logic per category
        ├── models/                  # ML model definitions & training logic
        │   ├── model.py             #    Model page renderer & router
        │   ├── models_execution.py  #    Training, prediction, metrics
        │   └── model_components.py  #    Reusable UI components for model pages
        ├── viz_lib/                 # Visualization helpers
        ├── stats_lib/               # Statistical computation functions
        └── reporting_lib/           # Report asset execution & rendering
```

---

## How It Works — AI Engine

The AI advisor uses a carefully engineered **dual-model architecture** to separate *reasoning* from *writing*:

```
Dataset Upload
     │
     ▼
Data Profiler ──────────────────────────────────────────────────────┐
(data_profiling.py)                                                  │
Computes: shape, dtypes, nulls, duplicates, cardinality,             │
          skewness, class balance, constant cols, outliers           │
     │                                                               │
     ▼                                                           GROUND_TRUTH
PLANNER (deepseek-r1:1.5b) ◄── System prompt + dataset facts block  │
     │   <think>...</think> reasoning stripped automatically         │
     │   Output: structured ML strategy text                         │
     ▼
WRITER (llama3.2:latest) ◄── Planner strategy + dataset facts
     │   Enforces: 21 hallucination-guard rules
     │   Output: polished Markdown report sections
     ▼
Report Assembly → PDF Export
```

**Background warmup:** On startup, both models receive a minimal "hi" prompt in a daemon thread so the first real inference request has no cold-start delay.

---

## Supported Models & Algorithms

<details>
<summary><b>🔵 Regression Models</b></summary>

- Linear Regression
- K-Nearest Neighbors Regressor
- Support Vector Regression (SVR)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

</details>

<details>
<summary><b>🟢 Classification Models</b></summary>

- Logistic Regression
- Naive Bayes Classifier
- Support Vector Machine (SVC)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- K-Nearest Neighbors Classifier

</details>

<details>
<summary><b>🟡 Unsupervised / Clustering</b></summary>

- K-Means Clustering
- Hierarchical Clustering (Agglomerative)
- DBSCAN
- Unsupervised KNN

</details>

<details>
<summary><b>🟣 Dimensionality Reduction</b></summary>

- PCA (Principal Component Analysis)
- t-SNE
- UMAP
- LDA (Linear Discriminant Analysis)

</details>

---

## Pipeline Save & Resume

Every transformation, model configuration, statistical analysis, and report block is stored in a unified **pipeline dictionary** in Streamlit's session state. At any point you can:

- Click **"Save Progress"** in the top-right corner to persist the pipeline to disk as an encrypted `.enc` file (AES-based via `cryptography.fernet`).
- Reload it in a future session via the file path input on the Transform page — all transformations, model configs, and report items will be restored instantly.
- Docker volumes ensure `save files/` persists across container restarts.

---

<div align="center">

*Graduation Project — Built with ❤️ using Streamlit, Ollama, and open-source ML.*

</div>
