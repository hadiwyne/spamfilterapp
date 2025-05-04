import joblib, streamlit as st
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load pipelines for each model
pipelines = {
    'Logistic Regression': joblib.load('spam_classifier_pipeline_lr.joblib'),
    'Naive Bayes': joblib.load('spam_classifier_pipeline_nb.joblib'),
    'MLP': joblib.load('spam_classifier_pipeline_mlp.joblib')
}

# Initialize session state for message history
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# Streamlit app
st.title("📧 Ensemble Spam Detector")
st.write("This app uses Logistic Regression, Naive Bayes, and MLP in a majority-vote ensemble.")

# Sidebar for EDA
with st.sidebar:
    st.header("Dataset Overview")
    df = pd.read_csv("spam_ham_dataset.csv")
    total = len(df)
    spam_count = (df['label'] == 'spam').sum()
    ham_count = (df['label'] == 'ham').sum()
    st.write(f"Total emails: {total}")
    st.write(f"Spam: {spam_count} ({spam_count/total*100:.1f}%)")
    st.write(f"Ham: {ham_count} ({ham_count/total*100:.1f}%)")
    
    # Pie chart
    fig, ax = plt.subplots()
    ax.pie([ham_count, spam_count], labels=['Ham', 'Spam'], autopct='%1.1f%%',
           colors=['#66b3ff','#ff9999'])
    st.pyplot(fig)

# Main interface
if 'email_input' in st.session_state:
    email_input = st.text_area("Enter email text:", st.session_state.email_input, height=200, key="email_text")
    # Clear session state if user types something new
    if email_input != st.session_state.email_input:
        del st.session_state.email_input
else:
    email_input = st.text_area("Enter email text:", height=200, key="email_text")

# Add example messages
with st.expander("Example messages to test"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ham examples")
        if st.button("Load Ham Example 1"):
            st.session_state.email_input = "Hi there, just checking in about our meeting tomorrow. Looking forward to seeing you!"
            st.rerun()
        if st.button("Load Ham Example 2"):
            st.session_state.email_input = "Please find attached the report you requested. Let me know if you need anything else."
            st.rerun()
    with col2:
        st.subheader("Spam examples")
        if st.button("Load Spam Example 1"):
            st.session_state.email_input = "CONGRATULATIONS! You've WON $5,000,000! Click here to claim your PRIZE now!!!"
            st.rerun()
        if st.button("Load Spam Example 2"):
            st.session_state.email_input = "Buy Viagra online! 90% discount! Limited time offer! Act now!"
            st.rerun()

# Classification
if st.button("Check Spam"):
    if email_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        try:
            verdicts = {}
            spam_votes = 0
            probs = {}
            
            # Run all pipelines
            for name, pipeline in pipelines.items():
                pred = pipeline.predict([email_input])[0]
                proba = pipeline.predict_proba([email_input])[0][1]
                label = 'SPAM' if pred == 1 else 'HAM'
                verdicts[name] = label
                probs[name] = proba
                if pred == 1:
                    spam_votes += 1
                    
            # Ensemble decision
            final_label = 'SPAM' if spam_votes >= 2 else 'HAM'
            avg_spam_proba = sum(probs.values()) / len(probs)
            
            # Display individual results
            st.subheader("Model Predictions")
            for name in pipelines:
                st.write(f"**{name}:** {verdicts[name]} ({probs[name]*100:.1f}% spam)")
            
            st.markdown("---")
            
            # Display ensemble verdict
            if final_label == 'SPAM':
                st.error(f"🚨 **Ensemble Verdict:** {final_label} (avg {avg_spam_proba*100:.1f}% spam)")
            else:
                st.success(f"✅ **Ensemble Verdict:** {final_label} (avg {avg_spam_proba*100:.1f}% spam)")
            
            # Add to message history
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.message_history.append({
                'timestamp': timestamp,
                'message': email_input[:50] + ("..." if len(email_input) > 50 else ""),
                'full_message': email_input,
                'result': final_label,
                'avg_prob': f"{avg_spam_proba*100:.1f}%",
                'model_verdicts': verdicts,
                'model_probs': {k: f"{v*100:.1f}%" for k, v in probs.items()}
            })

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")

# Show message history
if st.session_state.message_history:
    st.header("Message History")
    
    for i, entry in enumerate(st.session_state.message_history):
        with st.expander(f"{entry['timestamp']} - {entry['message']} ({entry['result']} - {entry['avg_prob']})"):
            st.text_area("Full message", entry['full_message'], height=100, key=f"history_{i}", disabled=True)
            
            # Show individual model predictions
            st.subheader("Model Predictions")
            for model_name in entry['model_verdicts']:
                st.write(f"**{model_name}:** {entry['model_verdicts'][model_name]} ({entry['model_probs'][model_name]} spam)")
    
    # Add a clear history button
    if st.button("Clear History"):
        st.session_state.message_history = []
        st.rerun()

# Documentation
with st.expander("How it works"):
    st.markdown("""
    This spam classifier uses an ensemble of three models:
    - Logistic Regression
    - Naive Bayes
    - Multi-Layer Perceptron (MLP)
    
    The final prediction is made using majority voting among the three models.
    
    ### Potential issues with spam classification:
    - Short messages may lack enough features for accurate classification
    - The model might be biased if the training data wasn't balanced
    - Some legitimate messages might contain words commonly found in spam
    - Try using longer, more detailed messages for better results
    """)