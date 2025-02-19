# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models import Word2Vec
 
# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# # Step 1: Load CSV File
# def load_csv(file_path):
#     df = pd.read_csv(file_path)
#     return df['email_text'].astype(str)  # Assuming the email content is in 'email_text' column

# # Step 2: Preprocess Text
# def preprocess_text(text):
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\W', ' ', text)  # Remove special characters
#     text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
#     words = word_tokenize(text)  # Tokenize words
#     stop_words = set(stopwords.words('english'))
#     return ' '.join([word for word in words if word not in stop_words])  # Remove stopwords

# # Step 3: Convert Text to Vectors using TF-IDF
# def tfidf_vectorize(corpus):
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform(corpus)
#     return vectors, vectorizer

# # Step 4: Convert Text to Vectors using Word2Vec
# def word2vec_vectorize(corpus):
#     tokenized_corpus = [word_tokenize(email) for email in corpus]
#     model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
#     return model  # Model can be used to get vectors of words

# # Main Execution
# if __name__ == "__main__":
#     file_path = "/Users/mi/Docs/RVU/IML/project/Data In CSV/New.csv"  # Replace with your actual file path
#     email_texts = load_csv(file_path)
#     preprocessed_texts = [preprocess_text(text) for text in email_texts]

#     # TF-IDF Vectorization
#     tfidf_vectors, tfidf_model = tfidf_vectorize(preprocessed_texts)
#     print("TF-IDF Vector Shape:", tfidf_vectors.shape)

#     # Word2Vec Embedding
#     word2vec_model = word2vec_vectorize(preprocessed_texts)
#     print("Word2Vec Model Trained. Vocabulary Size:", len(word2vec_model.wv))

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load CSV File
def load_csv(file_path):
    df = pd.read_csv(file_path)
    df['email_text'] = df['Subject'].astype(str) + " " + df['Body'].astype(str)  # Combine subject and body
    return df[['email_text']]

# Step 2: Preprocess Text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = word_tokenize(text)  # Tokenize words
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word not in stop_words])  # Remove stopwords

# Step 3: Convert Text to Vectors using TF-IDF
def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    return vectors, vectorizer

# Step 4: Convert Text to Vectors using Word2Vec
def word2vec_vectorize(corpus):
    tokenized_corpus = [word_tokenize(email) for email in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    return model  # Model can be used to get vectors of words

# Main Execution
if __name__ == "__main__":
    file_path = "/Users/mi/Docs/RVU/IML/project/Data In CSV/New.csv"  # Replace with your actual file path
    df = load_csv(file_path)
    preprocessed_texts = [preprocess_text(text) for text in df['email_text']]

    # TF-IDF Vectorization
    tfidf_vectors, tfidf_model = tfidf_vectorize(preprocessed_texts)
    print("TF-IDF Vector Shape:", tfidf_vectors.shape)

    # Word2Vec Embedding
    word2vec_model = word2vec_vectorize(preprocessed_texts)
    print("Word2Vec Model Trained. Vocabulary Size:", len(word2vec_model.wv))