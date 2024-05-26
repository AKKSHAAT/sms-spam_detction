import streamlit as st
import pickle
import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Function for batch prediction
def batch_predict(file):
    df = pd.read_csv(file)
    if 'message' not in df.columns:
        st.error('The uploaded CSV file must contain a "message" column.')
        return None, 0, 0  # Return 0 for spam_count and ham_count if there's an error
    
    df['language'] = df['message'].apply(lambda x: detect(x))
    df['transformed_message'] = df['message'].apply(transform_text)
    vector_input = tfidf.transform(df['transformed_message'])
    predictions = model.predict(vector_input)
    df['prediction'] = ['Spam ⛔' if label == 1 else 'Not Spam 👍' for label in predictions]
    
    # Count of spam and ham messages
    spam_count = df[df['prediction'] == 'Spam ⛔'].shape[0]
    ham_count = df[df['prediction'] == 'Not Spam 👍'].shape[0]
    
    return df[['message', 'prediction', 'language']], spam_count, ham_count

# Streamlit app
st.title("SMS Spam Detection ✉️")

# Sidebar for selecting batch or Sms Prediction
mode = st.sidebar.selectbox("Select Mode", ["Sms Prediction", "Batch Prediction"])

if mode == "Sms Prediction":
    # Sms Prediction
    input_sms = st.text_area("Enter your message here")
    if st.button('Predict'):
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        language = detect(input_sms)
        if result == 1:
            st.header("Spam ⛔")
        elif result == 0:
            st.header("Not Spam 👍")
        st.write(f"Detected Language: {language}")

elif mode == "Batch Prediction":
    # Batch prediction
    st.subheader('Batch Prediction')
    st.caption('Upload a CSV file with multiple messages and get predictions for all of them at once.')
    uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

    if uploaded_file is not None:
        predictions_df, spam_count, ham_count = batch_predict(uploaded_file)
        if predictions_df is not None:
            st.subheader('Predictions:')
            st.write(predictions_df)
            # Option to download the results
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions data ⬇️⬇️",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )

            # Visualization
            st.subheader('Spam vs Ham Distribution')
            fig = plt.figure(facecolor='black')  # Create a figure with a black background
            ax = fig.add_subplot()
            ax.pie([spam_count, ham_count], labels=['Spam', 'Ham'], autopct='%1.1f%%', startangle=90)
            ax.set_facecolor('black')  # Set the background color of the plot to black
            st.pyplot(fig)
