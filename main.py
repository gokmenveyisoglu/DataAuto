import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Load dataset
def load_dataset(base_path):
    data = []
    labels = []
    for author in os.listdir(base_path):
        author_path = os.path.join(base_path, author)
        if os.path.isdir(author_path):
            for file in os.listdir(author_path):
                file_path = os.path.join(author_path, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    data.append(content)
                    labels.append(author)
    return pd.DataFrame({'text': data, 'author': labels})

# Feature extraction functions
def extract_tfidf_features(corpus, ngram_range):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range)
    return vectorizer.fit_transform(corpus), vectorizer

def extract_char_ngrams(corpus, ngram_range):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)
    return vectorizer.fit_transform(corpus), vectorizer

def extract_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting BERT embeddings"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_embedding)
    return np.array(embeddings)

# Model training and evaluation
def evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    return acc, p, r, f

# Main pipeline
def main():
    base_path = "dataset_authorship/"  # Adjust path if needed
    df = load_dataset(base_path)

    # Encode labels
    label_encoder = LabelEncoder()
    df['author_encoded'] = label_encoder.fit_transform(df['author'])

    X_train, X_test, y_train_text, y_test_text = train_test_split(df['text'], df['author'], test_size=0.2, random_state=42)
    y_train = label_encoder.transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)

    feature_sets = [
        ('Word TF-IDF 1-gram', (1, 1), extract_tfidf_features),
        ('Word TF-IDF 2 gram', (2, 2), extract_tfidf_features),
        ('Word TF-IDF 3 gram', (3, 3), extract_tfidf_features),
        ('Char TF-IDF 2 gram', (2, 2), extract_char_ngrams),
        ('Char TF-IDF 3 gram', (3, 3), extract_char_ngrams)
    ]

    models = [
        ('Random Forest', RandomForestClassifier()),
        ('SVM', SVC()),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),
        ('Naive Bayes', MultinomialNB()),
        ('MLP', MLPClassifier(max_iter=300)),
        ('Decision Tree', DecisionTreeClassifier())
    ]

    results = []
    for name, ngram_range, extractor in feature_sets:
        print(f"\nFeature Set: {name}")
        X_train_feat, vectorizer = extractor(X_train, ngram_range)
        X_test_feat = vectorizer.transform(X_test)

        for model_name, model in models:
            print(f" Training {model_name}...")
            try:
                acc, p, r, f = evaluate_model(X_train_feat, X_test_feat, y_train, y_test, model)
                results.append((name, model_name, acc, p, r, f))
            except ValueError as e:
                print(f"  Skipping {model_name} due to error: {e}")

    # BERT embeddings (separately due to different format)
    print("\nExtracting BERT features...")
    X_train_bert = extract_bert_embeddings(X_train.tolist())
    X_test_bert = extract_bert_embeddings(X_test.tolist())

    for model_name, model in models:
        print(f" Training {model_name} on BERT features...")
        if isinstance(model, MultinomialNB): #GausianNb ye geç
            print("  Skipping Naive Bayes on BERT due to negative values in data.")
            continue
        try:
            acc, p, r, f = evaluate_model(X_train_bert, X_test_bert, y_train, y_test, model)
            results.append(("BERT", model_name, acc, p, r, f))
        except ValueError as e:
            print(f"  Skipping {model_name} on BERT due to error: {e}")

    # Save sorted results to CSV
    df_results = pd.DataFrame(results, columns=["Feature Set", "Model", "Accuracy", "Precision", "Recall", "F1-score"])
    df_results.sort_values(by="F1-score", ascending=False, inplace=True)
    df_results.to_csv("classification_results.csv", index=False)
    print("\nResults saved to classification_results.csv")

if __name__ == '__main__':
    main()
