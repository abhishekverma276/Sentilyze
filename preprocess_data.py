import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from joblib import Parallel, delayed
import nltk
from tqdm import tqdm

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load NLTK resources
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Define preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r"@[^\s]+", "", text)
    # Remove special characters and numbers
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

# Load the Sentiment140 dataset
data_path = "data/sentiment140.csv"  # Path to your dataset
columns = ["target", "id", "date", "flag", "user", "text"]
data = pd.read_csv(data_path, encoding="ISO-8859-1", names=columns)

# Display the first few rows of the dataset
print("Original Data:")
print(data.head())

# Apply preprocessing using parallel processing and tqdm for progress bar
n_jobs = -1  # Use all available CPU cores
data['clean_text'] = Parallel(n_jobs=n_jobs)(delayed(preprocess_text)(text) for text in tqdm(data['text']))

# Display the preprocessed text
print("Preprocessed Data:")
print(data[['text', 'clean_text']].head())

# Optionally, save the cleaned data for future use
cleaned_data_path = "data/cleaned_sentiment140.csv"
data.to_csv(cleaned_data_path, index=False)
print(f"Cleaned data saved to {cleaned_data_path}")
