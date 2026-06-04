# Workflow-CI - Sentiment Analysis ML Project

Project machine learning untuk analisis sentimen dengan CI/CD menggunakan GitHub Actions.

## Struktur Project

```
Workflow-CI
├── .github
│   └── workflows
│       └── train.yml          # GitHub Actions workflow
└── MLProject
    ├── MLproject              # MLflow project config
    ├── conda.yaml             # Conda environment config
    ├── requirements.txt       # Python dependencies
    ├── modelling.py           # Training script
    └── dataset_preprocessing.csv  # Dataset
```

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/firzadillah0192-source/Workflow-CI-Maheza.git
cd Workflow-CI-Maheza
```

### 2. Install Dependencies

```bash
pip install -r MLProject/requirements.txt
```

### 3. Run Training Locally

```bash
cd MLProject
python modelling.py
```

## GitHub Actions CI/CD

Workflow akan otomatis berjalan setiap kali ada push ke branch `main`:

1. Checkout repository
2. Setup Python 3.12
3. Install dependencies
4. Run training script

## Model Details

- **Model**: Multinomial Naive Bayes
- **Vectorizer**: TF-IDF (max 1000 features)
- **Task**: Binary sentiment classification (positive/negative)
- **MLflow**: Tracking experiments dan model versioning

## Output

Setelah training, akan dihasilkan:
- `sentiment_model.pkl` - Trained model
- `vectorizer.pkl` - TF-IDF vectorizer
- MLflow tracking data

## Requirements

- Python 3.12
- pandas
- numpy
- scikit-learn
- mlflow==2.19.0
- nltk
- joblib
