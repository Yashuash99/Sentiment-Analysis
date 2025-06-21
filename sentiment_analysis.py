import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    tokens = text.split()  # ✅ More robust tokenizer for tweets
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def main():
    # Load Dataset
    df = pd.read_csv("Tweets.csv")[['text', 'airline_sentiment']]
    df = df[df['airline_sentiment'].isin(['positive', 'neutral', 'negative'])]
    
    # Preprocess Text
    print("Cleaning tweets...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Feature Extraction
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = LabelEncoder().fit_transform(df['airline_sentiment'])

    # Train-test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Save Model & Vectorizer
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("\nModel and vectorizer saved!")

if __name__ == "__main__":
    main()
