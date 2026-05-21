# Data Science Studio (Graduation Project)

A comprehensive, end-to-end Data Science and Machine Learning platform built with Streamlit. This application empowers users to seamlessly transition from raw data to actionable insights and predictive models without writing code. 

It provides an intuitive UI for data transformation, exploratory data analysis (EDA), statistical analysis, machine learning modeling, AI-powered recommendations, and automated report generation.

## 🌟 Key Features

1. **🤖 AI Recommendations**
   - Integrates with Large Language Models (Google Gemini & local Ollama models).
   - Automated data profiling and intelligent preprocessing recommendations.
   - AI-driven insights generated directly from dataset statistics.

2. **⚙️ Data Transformation Pipeline**
   - **Cleaning**: Handle missing values, outliers, duplicate records, and validate data types.
   - **Standardization**: Apply Mean Normalization, MinMax Scaling, Robust Scaler, and Z-score standardization.
   - **Encoding**: Ordinal, Label, One-hot, Target, and Binary encoding for categorical features.
   - **Feature Selection**: Chi-Squared, Correlation, Variance-based selection, and ANOVA.
   - **Dimensionality Reduction**: PCA, t-SNE, UMAP, and LDA.
   - *Pipelines can be saved, edited, and reloaded to maintain reproducible workflows.*

3. **📊 Analysis & EDA (Interactive Visualizations)**
   - Create custom dynamic dashboards using Plotly, Altair, and Seaborn.
   - Apply global data transformations directly to visualization inputs.
   - Save and export custom visual reports.

4. **🧠 Machine Learning Studio**
   - **Regression**: Linear Regression, KNN Regressor, SVR, Random Forest, Gradient Boosting, and XGBoost.
   - **Classification**: Logistic Regression, Naive Bayes, SVC, Decision Trees, Random Forest, and XGBoost.
   - **Unsupervised Learning**: K-Means, Hierarchical Clustering, DBSCAN.
   - *Includes interactive model creation, evaluation metrics, and hyperparameter tuning interfaces.*

5. **📈 Statistical Analysis (`Stats`)**
   - Deep statistical evaluations of datasets to uncover patterns and relationships.

6. **📝 Reporting**
   - Generate comprehensive PDF and visual reports summarizing EDA, transformations, and model performances.

## 🛠️ Technology Stack

- **Frontend / Framework**: Streamlit, Streamlit Option Menu, Streamlit Elements
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, UMAP-Learn
- **Visualization**: Plotly, Altair, Matplotlib, Seaborn
- **AI & LLMs**: Google Generative AI (Gemini), Ollama
- **Containerization**: Docker, Docker Compose

## 🚀 Getting Started

### Prerequisites

You need to have Python 3.11+, [Docker](https://www.docker.com/), and [Docker Compose](https://docs.docker.com/compose/) installed on your machine.

### Method 1: Using Docker (Recommended)

The easiest way to run the application along with the required environment configurations (especially for the Ollama integration) is via Docker Compose.

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd Grad-project
   ```

2. **Start the application**:
   ```bash
   docker-compose up -d --build
   ```

3. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`.

*Note: The `docker-compose.yml` is pre-configured to route the `OLLAMA_HOST` to your host machine if you are running a local instance of Ollama.*

### Method 2: Running Locally

1. **Navigate to the project directory**:
   ```bash
   cd Grad-project/Stream
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run scripts/main.py
   ```

## 📁 Project Structure

```
Grad-project/
├── docker-compose.yml        # Docker Compose configuration for the full stack
├── README.md                 # Project documentation
└── Stream/                   # Main Application Directory
    ├── Dockerfile            # Dockerfile for the Streamlit app
    ├── requirements.txt      # Python dependencies
    ├── save files/           # Directory for saved pipelines, user configs, and datasets
    └── scripts/              # Application Source Code
        ├── main.py           # Entry point of the Streamlit application
        ├── transform.py      # Data transformation and preprocessing logic
        ├── visualizations.py # Interactive EDA and dashboarding
        ├── machine.py        # Machine Learning studio and model evaluation
        ├── stats.py          # Statistical analysis module
        ├── reporting.py      # Automated report generation
        ├── constants.py      # Global constants and state managers
        ├── ai_recommendations_lib/ # AI advising, profiling, and LLM integration
        ├── transformations/  # Transformation execution libraries
        ├── models/           # ML model definitions and logic
        ├── viz_lib/          # Visualization helper libraries
        └── stats_lib/        # Statistical computation functions
```

## 💡 Usage Workflow

1. **Upload Data**: Navigate to the **Transform** tab to upload your `.csv` or `.xlsx` dataset.
2. **Consult AI Advisor**: Go to **AI Recommendations** for an automated health check and suggested preprocessing steps.
3. **Preprocess**: Apply necessary cleaning, encoding, or scaling transformations in the **Transform** tab. Save your progress.
4. **Explore**: Use **Analysis and EDA** and **Stats** to visually and statistically understand your data.
5. **Model**: Build predictive models in the **Machine Learning** tab, comparing different algorithms.
6. **Report**: Generate a summary report of your entire workflow in the **Reporting** tab.

---
*Developed as a Graduation Project.*
