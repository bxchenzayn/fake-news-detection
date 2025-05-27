import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained XGBoost model and the TF-IDF vectorizer
xgb_model = joblib.load("XGBoost_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define a preprocessing function to clean and tokenize input text
def preprocess(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Remove non-letter characters
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# Example news article input
news_text = "The president announced a new economic policy today."

# Preprocess and vectorize the text
cleaned_text = preprocess(news_text)
X_input = vectorizer.transform([cleaned_text])

# Make prediction using the XGBoost model
prediction = xgb_model.predict(X_input)[0]
label = "Real News" if prediction == 1 else "Fake News"

# Output the result
print(f"Prediction: {label}")
