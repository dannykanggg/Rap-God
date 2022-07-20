import streamlit as st
import keras
import pickle
from tensorflow.keras.utils import pad_sequences
import numpy as np
from random import randint
from better_profanity import profanity

# title and description
st.write("""
# Rap god
""")

# search bar
query = st.text_input("Give me a song title", "")

## load model
model = keras.models.load_model("./model/model110.h5")
with open('./model/tokenizer110.pickle','rb') as pfile:
    tokenizer = pickle.load(pfile)

def rearrange_text(rap_lyrics):
    title_len = 2
    title = ""
    for i in range(title_len):
        title += rap_lyrics.pop(0) + " "
    title = title.strip()
    verses = []
    while len(rap_lyrics)>0:
        verse = ''
        for i in range(randint(8,12)):
            if len(rap_lyrics)==0:
                break
            verse += rap_lyrics.pop(0) + " "
        verse = verse.strip() + "\n"
        verse = profanity.censor(verse)  # censor out profanity
        verses.append(verse) 
    return title,verses

def generate_text(seed_text, next_words, max_sequence_len, model):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen= 
                             max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
  
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(predicted[0]):
                output_word = word
                break
        seed_text += " " + output_word
    
    title,verses = rearrange_text(seed_text.split(" "))
    return title,verses


max_sequence_len = 67



if query:
    title,verses = generate_text(query, 100, max_sequence_len,model)


    st.write(f"## Song title: {title}")

    st.write('Lyrics:')
    st.write(*verses, sep = "\n")

    #st.write(f"Query = '{query}'")
    