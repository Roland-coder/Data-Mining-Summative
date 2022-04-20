import streamlit as st
import pandas as pd
import pickle
import numpy as np
import nltk



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import string
from PIL import Image

model = pickle.load(open('news_model', 'rb'))

nltk.download('stopwords')

st.title("Fake News Predict App")
st.header("Fake News Prediction")
st.write("This web app predicts the fake news based on the content of the news article")
st.write("Please be patient, app may take a while to do predictions due to some internal processes that need to be met")

image = Image.open("fake-news-image.png")
st.image(image, use_column_width=True)
text_pred = st.text_input("Please enter news article in the text below")
df = pd.read_csv("https://jeanbucket001.s3.us-west-2.amazonaws.com/train.csv.zip" )



if st.button('Predict News article  text'):
	if len(text_pred) <= 2:
		st.write("Please enter a valid news article")
	else:
		pred = model.predict([text_pred])
# 		pred = np.argmax(pred, axis = 0)
		if pred == 0:
			st.write("News Article is Reliable")
		else:
			st.write("News article is not Unreliable")
  	
# 	st.write("The overall predicted score for the above player is", clubs.index(club))
else:
	st.write('Thank You For Trusting Us')
