import re

# Single-text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# Batch preprocessing for FunctionTransformer

def preprocess_texts_func(texts):
    return [preprocess_text(t) for t in texts]

# Convert sparse matrix to dense

def to_dense_func(X):
    return X.toarray()