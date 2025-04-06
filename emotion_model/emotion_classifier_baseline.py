from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import numpy as np

# 1. Load dataset
dataset = load_dataset("dair-ai/emotion")
train_data = dataset['train']
label_names = dataset['train'].features['label'].names

# 2. Extract text and labels
texts = [example['text'] for example in train_data]
labels = [example['label'] for example in train_data]

# 3. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Train classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_names))
