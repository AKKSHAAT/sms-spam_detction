import streamlit as st
import pickle
import string
import nltk
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

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

def batch_predict(file):
    df = pd.read_csv(file)
    if 'message' not in df.columns:
        st.error('The uploaded CSV file must contain a "message" column.')
        return None
    
    df['transformed_message'] = df['message'].apply(transform_text)
    vector_input = tfidf.transform(df['transformed_message'])
    predictions = model.predict(vector_input)
    df['prediction'] = ['Spam ⚠️' if label == 1 else 'Not Spam 👍' for label in predictions]
    
    return df[['message', 'prediction']]


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS spam Detction ✉️")
input_sms = st.text_area("Enter your message here")
st.caption("100% Precision Score")
st.caption("97% Accuracy Score")

# 0.9709864603481625
# 1.0

if st.button('Predict'):
    #1. preprocess
    transformed_sms = transform_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)[0]
    #  Display
    if result == 1:
        st.header("Spam ⚠️")
    elif result == 0:
        st.header("Not Spam 👍")

# -----------------------Batch Prediction------------------------------
st.subheader('Batch Prediction')
st.caption('you can upload a CSV file with multiple messages and get predictions for all of them at once.')
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])


if uploaded_file is not None:
    predictions_df = batch_predict(uploaded_file)
    if predictions_df is not None:
        st.subheader('Predictions:')
        st.write(predictions_df)

        # Option to download the results
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )
