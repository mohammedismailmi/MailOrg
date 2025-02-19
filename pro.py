import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# --- DATA LOADING AND PREPROCESSING ---
def load_data(filepath):
    """Loads the dataset from csv"""
    data = pd.read_csv(filepath)
    return data

def preprocess_text(text):
    """Cleans the emails"""
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)
# --- FEATURE EXTRACTION ---
def tfidf_representation(text, max_features=1000):
    """Converts emails into features using tfidf"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(text)
    return features, vectorizer

# --- MODEL TRAINING ---
def train_classifier(features, labels):
    """Trains a model based on input"""
    model = MultinomialNB()
    model.fit(features, labels)
    return model

# --- CLASSIFICATION ---
def classify_email(email, vectorizer, classifier):
    """Predict the label of a given email"""
    preprocessed_email = preprocess_text(email)
    feature = vectorizer.transform([preprocessed_email])
    return classifier.predict(feature)[0]

# --- USER INTERACTION ---
def user_interaction(emails, labels, classifier, vectorizer):
    """Prompts user to take action for different emails"""
    new_labels = []
    deleted_email_ids = []
    for idx, (email, label) in enumerate(zip(emails, labels)):
        print(f"\nEmail {idx+1}:")
        print(f"   Email: {email[:100]}...")
        print(f"   Current Label: {label}")
        predicted_label = classify_email(email, vectorizer, classifier)
        print(f"   Predicted Label: {predicted_label}")

        action = input("Keep(k), Delete(d) or provide new label: ").lower()
        if action == "d":
            deleted_email_ids.append(idx)
        elif action != "k":
             new_labels.append((idx,action))

    return new_labels, deleted_email_ids

def main():
    # Load data
    filepath = 'path/to/your/emails.csv'  #replace
    data = load_data(filepath)
    # Ensure your DataFrame has columns 'email' and 'label'
    data.rename(columns={"email_column_name":"email", "label_column_name":"label"}, inplace=True) #replace
    data["preprocessed_email"] = data["email"].apply(preprocess_text)

    # Split Data
    features, vectorizer = tfidf_representation(data['preprocessed_email'])
    labels = data['label']
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train Model
    classifier = train_classifier(features_train, labels_train)
    predicted_labels = classifier.predict(features_test)

    accuracy = accuracy_score(labels_test, predicted_labels)
    print(f"Accuracy of model on test set: {accuracy}")
    # User interaction for some emails
    emails_for_user_interaction = data["email"].to_list()[:5] # take first five to display
    labels_for_user_interaction = data["label"].to_list()[:5]
    new_labels, deleted_email_ids = user_interaction(emails_for_user_interaction, labels_for_user_interaction, classifier, vectorizer)
    print(f"New Labels: {new_labels}")
    print(f"Deleted Email Ids: {deleted_email_ids}")
    # Do some operations based on user input, like storing the results
if __name__ == "__main__":
    main()