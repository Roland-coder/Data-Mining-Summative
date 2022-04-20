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

model = pickle.load(open('https://cloud-lab-20.s3.amazonaws.com/fake_news_model', 'rb'))

nltk.download('stopwords')

st.title("Fake News Predict App")
st.header("Fake News Prediction")
st.write("This web app predicts the fake news based on the content of the news article")
st.write("Please be patient, app may take a while to do predictions due to some internal processes that need to be met")

image = Image.open("fake-news-image.png")
st.image(image, use_column_width=True)
text_pred = st.text_input("Please enter news article in the text below")
df = pd.read_csv("https://jeanbucket001.s3.us-west-2.amazonaws.com/train.csv.zip" )


def cleaning_text(text):
    stop_words = stopwords.words("english")

    # removing urls from tweets
    text = re.sub(r'http\S+', " ", text)    
    # remove mentions
    text = re.sub(r'@\w+',' ',text)         
    # removing hastags
    text = re.sub(r'#\w+', ' ', str(text))       
    # removing html tags
    text = re.sub('r<.*?>',' ', text)       
    
    # removing stopwords stopwords 
    text = text.split()
    text = " ".join([word for word in text if not word in stop_words])

    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    
    return text

df['text'] = df['text'].replace(np.nan, '').apply(lambda x: cleaning_text(x)) 
new_text = cleaning_text(text_pred)

  
def tokenizer(x_train, y_train, newv, max_len_word):
    # because the data distribution is imbalanced, "stratify" is used
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                      test_size=.2, shuffle=True, 
                                                      stratify=y_train, random_state=0)

    # Tokenizer
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    sequence_dict = tokenizer.word_index
    word_dict = dict((num, val) for (val, num) in sequence_dict.items())

    # Sequence data
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences,
                                 maxlen=max_len_word,
                                 truncating='post',
                                 padding='post')
    X_val[len(X_val)] = newv
    val_sequences = tokenizer.texts_to_sequences(X_val)
    val_padded = pad_sequences(val_sequences,
                                maxlen=max_len_word,
                                truncating='post',
                                padding='post', )
   

    print(train_padded.shape)
    print(val_padded.shape)
    print('Total words: {}'.format(len(word_dict)))
    return train_padded, val_padded, y_train, y_val, word_dict

X_train, X_val, y_train, y_val, word_dict = tokenizer(df.text, df.label, new_text, 300)


if st.button('Predict News article  text'):
	if len(new_text) <= 2:
		st.write("Please enter a valid news article")
	else:
		pred = model.predict(X_val[-1])
		pred = np.argmax(pred, axis = 0)
		if pred == 0:
			st.write("News Article is fake")
		else:
			st.write("News article is not fake")
  	
# 	st.write("The overall predicted score for the above player is", clubs.index(club))
else:
	st.write('Thank You For Trusting Us')
