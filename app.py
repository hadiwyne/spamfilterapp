import joblib, streamlit as st
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from text_preprocessing import preprocess_texts



# Initialize session state for message history
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
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

# Add model debugging information
with st.expander("Model Information"):
    # Display model details if available
    try:
        pipeline_steps = model.named_steps
        st.write("Pipeline steps:", list(pipeline_steps.keys()))
        
        # If the model has a classifier with feature importances, show them
        if hasattr(model[-1], 'coef_'):
            # For logistic regression, get feature names if available
            if hasattr(model[0], 'get_feature_names_out'):
                feature_names = model[0].get_feature_names_out()
                coefficients = model[-1].coef_[0]
                
                # Show top positive and negative coefficients
                coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                
                st.write("Top 10 spam indicators:")
                st.dataframe(coef_df.head(10))
                
                st.write("Top 10 ham indicators:")
                st.dataframe(coef_df.tail(10))
    except Exception as e:
        st.write("Could not extract model details:", str(e))

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

# Remove the duplicate update of email_input
# if 'email_input' in st.session_state:
#     email_input = st.session_state.email_input

# Preprocessing 
if st.button("Check Spam"):
    if email_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        try:
            prediction = model.predict([email_input])[0]
            proba = model.predict_proba([email_input])[0]
            result_label = "Spam" if prediction == 1 else "Ham"

            if prediction == 1:
                st.error("🚨 This email is classified as **SPAM**.")
            else:
                st.success("✅ This email is classified as **NOT SPAM**.")

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ham probability", f"{proba[0]*100:.2f}%")
            with col2:
                st.metric("Spam probability", f"{proba[1]*100:.2f}%")
            # Progress bar (simple visualization)
            st.progress(proba[1])
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")
        
         # Add to message history
            timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
            st.session_state.message_history.append({
                "timestamp": timestamp,
                "message": email_input[:50] + ("..." if len(email_input) > 50 else ""),
                "full_message": email_input,
                "result": result_label,
                "ham_prob": f"{proba[0]*100:.2f}%",
                "spam_prob": f"{proba[1]*100:.2f}%"
            })
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


# Documentation
with st.expander("How it works"):
    st.markdown("""
    This spam classifier uses:
    - Logistic Regression model
    - Text preprocessing (stemming, stopword removal)
    - Bag-of-words feature extraction
    - Trained on 5171 emails (Spam/Ham dataset)
    
    ### Potential issues with spam classification:
    - Short messages may lack enough features for accurate classification
    - The model might be biased if the training data wasn't balanced
    - Some legitimate messages might contain words commonly found in spam
    - Try using longer, more detailed messages for better results
    """)

# Add a section for model retraining or threshold adjustment
with st.expander("Advanced Settings"):
    st.write("Adjust classification threshold (default is 0.5)")
    threshold = st.slider("Spam probability threshold", 0.0, 1.0, 0.5, 0.05)
    st.write(f"Messages with spam probability > {threshold} will be classified as spam")
    
    if email_input and 'proba' in locals():
        adjusted_prediction = "SPAM" if proba[1] > threshold else "HAM"
        
        # Add visual indicators for threshold-based prediction
        if adjusted_prediction == "SPAM":
            st.error(f"With threshold {threshold}: This message is {adjusted_prediction}")
        else:
            st.success(f"With threshold {threshold}: This message is {adjusted_prediction}")
            
        # Show the difference from default threshold
        if (proba[1] > 0.5) != (proba[1] > threshold):
            st.warning("Note: This prediction differs from the default threshold (0.5)")
            
        # Add visualization of threshold
        st.write("Probability distribution:")
        prob_chart = st.progress(proba[1])
        st.markdown(f"""
        - Current spam probability: {proba[1]*100:.2f}%
        - Current threshold: {threshold*100:.2f}%
        - Default threshold: 50%
        """)

if st.session_state.message_history:
    st.header("Message History")
    
    # Create a table for the message history
    history_df = pd.DataFrame(st.session_state.message_history)
    
    # Display the table with custom formatting
    for i, entry in enumerate(st.session_state.message_history):
        with st.expander(f"{entry['timestamp']} - {entry['message']} ({entry['result']})"):
            st.text_area("Full message", entry['full_message'], height=100, key=f"history_{i}", disabled=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Ham probability: {entry['ham_prob']}")
            with col2:
                st.write(f"Spam probability: {entry['spam_prob']}")
    
    # Add a clear history button
    if st.button("Clear History"):
        st.session_state.message_history = []
        st.rerun()