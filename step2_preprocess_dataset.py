from clearml import Task
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Initialize ClearML Task
task = Task.init(project_name="Fake News Detection v1", task_name="Pipeline Step 2 - Preprocess Dataset")

# Task arguments
args = {
    'dataset_task_id': '4dbb68b1f7814280a066303850bf7428',  # Will be overridden by pipeline
    'test_size': 0.3,
    'random_state': 1,
}

task.connect(args)

# Get dataset from Step 1
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
dataset_path = dataset_task.artifacts['raw_dataset'].get_local_copy()
df = pd.read_csv(dataset_path)
print(f"Loaded dataset with shape: {df.shape}")

# Drop unused columns if exist
for col in ['subject', 'date']:
    if col in df.columns:
        df = df.drop(col, axis=1)

# Preserve labels
label_train = df['label']

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text preprocessing
corpus = []
for x in range(len(df)):
    review = df['text'][x]
    review = re.sub(r'[^a-zA-Z\s]', '', str(review))
    review = review.lower()
    tokens = nltk.word_tokenize(review)
    tokens = [lemmatizer.lemmatize(y) for y in tokens if y not in stop_words and len(y) > 2]
    clean_review = ' '.join(tokens)
    corpus.append(clean_review)

# Replace cleaned text and labels
df['text'] = corpus
df['label'] = label_train

# Drop rows with empty or invalid values
df = df[df['text'].str.strip().astype(bool)]
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)


# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

print("Label distribution:", y.value_counts())

#Save vectorizer
import joblib
vectorizer_path = "tfidf_vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_path)
task.upload_artifact("tfidf_vectorizer", artifact_object=vectorizer_path)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args['test_size'], random_state=args['random_state']
)

# Upload artifacts to ClearML
task.upload_artifact("X_train", X_train)
task.upload_artifact("X_test", X_test)
task.upload_artifact("y_train", y_train)
task.upload_artifact("y_test", y_test)

task.set_parameters({"General/processed_dataset_id": task.id})

print("Uploaded artifacts: X_train, X_test, y_train, y_test")
print(f"Task link: {task.get_output_log_web_page()}")
print(f"Task ID: {task.id}")

# Close the task
task.close()
