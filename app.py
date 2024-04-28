import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import requests

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

model_url = 'https://filetransfer.io/data-package/dKezepPr/download'  
vectorizer_url = 'https://filetransfer.io/data-package/ucMBTvgA/download' 
model_path = './models'
os.makedirs(model_path, exist_ok=True)
model_filename = os.path.join(model_path, 'multinomial_nb_model_bow.pkl')
vectorizer_filename = os.path.join(model_path, 'bow_vectorizer.pkl')

with st.spinner('Preparing Data...'):
    if not os.path.exists(model_filename):
        download_file(model_url, model_filename)
    if not os.path.exists(vectorizer_filename):
        download_file(vectorizer_url, vectorizer_filename)
st.success('Done Preparing data!')


models = {
    'multinomial_nb': {
        'bag_of_words': {
            'model': pickle.load(open(model_filename, 'rb')),
            'vectorizer': pickle.load(open(vectorizer_filename, 'rb'))
        },
    }
}

st.title('Sentiment Analysis')
model_choice = st.selectbox('Choose a model:', list(models.keys()))
feature_choice = st.selectbox('Choose a feature type:', ['bag_of_words'])
review_text = st.text_area("Enter the review text:", height=150)

if st.button('Predict Sentiment'):
    model = models[model_choice][feature_choice]['model']
    vectorizer = models[model_choice][feature_choice]['vectorizer']
    
    vec_text = vectorizer.transform([review_text])
    
    prediction = model.predict(vec_text)
    confidence = np.max(model.predict_proba(vec_text)) * 100  # confidence score
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    st.success(f'The predicted sentiment is {result} with {confidence:.2f}% confidence.')

