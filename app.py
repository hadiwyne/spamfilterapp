import streamlit as st
import joblib
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load saved artifacts
model = joblib.load('spam_classifier_pipeline.joblib')

# Preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Streamlit app
st.title("📧 Spam Email Classifier")
st.write("Using Logistic Regression")

# Sidebar for EDA
with st.sidebar:
    st.header("Dataset Overview")
    df = pd.read_csv("spam_ham_dataset.csv")
    st.write(f"Total emails: {len(df)}")
    spam_count = df['label'].value_counts()['spam']
    ham_count = df['label'].value_counts()['ham']
    st.write(f"Spam emails: {spam_count} ({spam_count/len(df)*100:.1f}%)")
    st.write(f"Ham emails: {ham_count} ({ham_count/len(df)*100:.1f}%)")
    
    # Pie chart
    fig, ax = plt.subplots()
    ax.pie([ham_count, spam_count], labels=['Ham', 'Spam'], autopct='%1.1f%%',
           colors=['#66b3ff','#ff9999'])
    st.pyplot(fig)

# Main interface
email_input = st.text_area("Enter email text:", height=200)

# Preprocessing 
if st.button("Check Spam"):
    if email_input:
        # Preprocess
        cleaned_text = preprocess_text(email_input)
        st.write("Preprocessed text:", cleaned_text)  # Debug output
        
        # Predict directly using pipeline
        prediction = model.predict([cleaned_text])[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is HAM (not spam)")
        
        st.write("Model confidence:")
        proba = model.predict_proba([cleaned_text])[0]
        st.write(f"Ham probability: {proba[0]*100:.2f}%")
        st.write(f"Spam probability: {proba[1]*100:.2f}%")
    else:
        st.warning("Please enter some text to analyze")

# Documentation
with st.expander("How it works"):
    st.markdown("""
    This spam classifier uses:
    - Logistic Regression model
    - Text preprocessing (stemming, stopword removal)
    - Bag-of-words feature extraction
    - Trained on 5171 emails (Spam/Ham dataset)
    """)