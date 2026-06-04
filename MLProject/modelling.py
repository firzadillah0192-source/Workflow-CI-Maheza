import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: Could not download NLTK stopwords")

def preprocess_text(text):
    """Preprocess text data"""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def main():
    print("Starting ML Training Pipeline...")
    
    # Start MLflow run
    mlflow.set_experiment("sentiment-analysis")
    
    with mlflow.start_run():
        # Load dataset
        try:
            df = pd.read_csv('dataset_preprocessing.csv')
            print(f"Dataset loaded successfully with {len(df)} records")
        except FileNotFoundError:
            print("Warning: dataset_preprocessing.csv not found. Creating sample data...")
            # Create sample dataset
            df = pd.DataFrame({
                'text': [
                    'I love this product', 'This is terrible', 'Great experience',
                    'Not good at all', 'Absolutely fantastic', 'Very disappointed',
                    'Highly recommend', 'Waste of money', 'Best purchase ever',
                    'Would not buy again'
                ],
                'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive',
                             'negative', 'positive', 'negative', 'positive', 'negative']
            })
        
        # Preprocess text
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
        )
        
        # Vectorization
        vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Training Complete!")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Log parameters and metrics to MLflow
        mlflow.log_param("model_type", "MultinomialNB")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("max_features", 1000)
        mlflow.log_metric("accuracy", accuracy)
        
        # Save model and vectorizer
        joblib.dump(model, 'sentiment_model.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact('sentiment_model.pkl')
        mlflow.log_artifact('vectorizer.pkl')
        
        print("\nModel and artifacts saved successfully!")
        print("MLflow tracking completed.")

if __name__ == "__main__":
    main()
