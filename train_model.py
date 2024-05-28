import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load the cleaned dataset
data_path = "data/cleaned_sentiment140.csv"
data = pd.read_csv(data_path)

# Drop rows with missing values
data.dropna(inplace=True)


# Extract features and labels
X = data['clean_text']
y = data['target']

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Use TF-IDF Vectorizer to convert text data into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Define a list of alpha values to try
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_accuracy = 0
best_alpha = None
best_model = None

# Train a model for each alpha value and select the best one
for alpha in alpha_values:
    # Initialize the Naive Bayes classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    
    # Train the classifier
    nb_classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate the classifier on the validation set
    y_val_pred = nb_classifier.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    # Check if this model has the highest accuracy so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_alpha = alpha
        best_model = nb_classifier

# Train the best model on the full training set
X_full_train_tfidf = vectorizer.transform(X)
best_model.fit(X_full_train_tfidf, y)

# Evaluate the best model on the test set
X_test_tfidf = vectorizer.transform(X_test)
y_test_pred = best_model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best Alpha:", best_alpha)
print("Test Accuracy:", test_accuracy)

# Save the best model and vectorizer
joblib.dump(best_model, "server/best_model.pkl")
joblib.dump(vectorizer, "server/vectorizer.pkl")
print("Best model and vectorizer saved successfully!")
