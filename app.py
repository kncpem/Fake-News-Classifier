import streamlit as st
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Load your trained model and CountVectorizer
model = joblib.load('fake_news_classifier.pkl')
cv = joblib.load('count_vectorizer.pkl')

nltk.download('stopwords')

st.title('Fake News Classifier')

# Define the icon URL (this example uses a Font Awesome icon)
icon_url = "https://upload.wikimedia.org/wikipedia/commons/4/4a/News_Icon.png"

# Create a string with HTML for the icon and the text
custom_text2 = f"""
<p style="font-size:12px; color:grey;">
    <img src="{icon_url}" alt="News Icon" style="width:16px; vertical-align:middle; margin-right:4px;">
    by kncpem
</p>
"""

# Display the custom text with the icon using st.markdown
st.markdown(custom_text2, unsafe_allow_html=True)

input_text = st.text_area("Enter the news text:")

def preprocess_title(title):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', title)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

if st.button("Classify"):
    preprocessed_title = preprocess_title(input_text)
    
    # Transform the input text using the fitted CountVectorizer
    title_vector = cv.transform([preprocessed_title]).toarray()
    
    # Perform prediction
    prediction = model.predict(title_vector)
    
    try:
        prediction_proba = model._predict_proba_lr(title_vector)
        confidence = prediction_proba[0][prediction[0]] * 100
    except AttributeError:
        confidence = "N/A"

    # Output with better-decorated font
    st.markdown(f"""
        <div style="text-align: center; font-size: 24px; font-weight: bold; color: {'green' if prediction[0] == 1 else 'red'};">
        Prediction: {'Real News' if prediction[0] == 1 else 'Fake News'}
        </div>
        <div style="text-align: center; font-size: 20px; margin-top: 10px;">
            Confidence: {confidence if isinstance(confidence, str) else f'{confidence:.2f}%'}
        </div>
    """, unsafe_allow_html=True)

# Create a string with HTML and CSS for the text
custom_text = """
<p style="font-size:12px; color:grey;">by kncpem</p>
"""

# Display the custom text using st.markdown
st.markdown(custom_text, unsafe_allow_html=True)
