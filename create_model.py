## create model with top 10 songs of 21 artists, 210 songs total

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import numpy as np
from lyricsgenius import Genius
import re
import time
import pickle

from random import randint
from better_profanity import profanity
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences,to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dropout,Dense


import sys
sys.path.insert(1, '../../')
import api

## authenticate credentials from spotify
cid = api.SPOTIFY_CID
secret = api.SPOTIFY_SKEY
#Authentication - without user
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)



## start with eminem's page
eminem = 'https://open.spotify.com/artist/7dGJo4pcD2V6oG8kP0tJRR?si=cbkdp4ZLQl-iQ-FLN1-1Qg'
# get eminem's info
eminem_info = sp.artist(eminem)


# returns top 10 tracks of every artist
def artist_top_tracks(artist_id): 
    top_tracks_info = sp.artist_top_tracks(artist_id)
    song_names = [n['name'] for n in top_tracks_info['tracks']]
    return song_names


## get artists related to eminem & add to list
related_artists = sp.artist_related_artists(eminem)
all_artists = [eminem_info] + related_artists['artists']



# initialize token for getting lyric data
genius = Genius(api.GENIUS_TOKEN)
lyrics_list = []


## get top 10 tracks per artist & save with lyrics
#all_tracks = []
for i,artist in enumerate(all_artists):
    artist_name = artist['name']
    artist_id = artist['id']
    songs = artist_top_tracks(artist_id)

    # get lyrics for each song
    for song in songs:
        """try:
            genius_song = genius.search_song(song,artist_name)
        except:
            time.sleep(2.4) # adding some time if failed to retrieve on 1 try
            genius_song = genius.search_song(song,artist_name)"""
        
        while True:
            try:
                genius_song = genius.search_song(song,artist_name)
                break
            except:
                pass

        if genius_song==None:
            continue

        lyrics_list.append([artist_name,song,genius_song.lyrics])


## process lyric data
def preprocess_lyrics(lyrics):
    lyrics = re.sub('^[^\[]*','',lyrics)  #get rid of everything before first verse
    lyrics = re.sub('\[(.*?)\]','[VERSE]',lyrics) # sub all [] with verse
    lyrics = re.sub('\((.*?)\)','',lyrics) # remove all parenthesis
    lyrics = re.sub('[0-9]+Embed','',lyrics) # remove the last Embed thingy
    lyrics = re.sub('  *\n','\n',lyrics) # remove space before newline
    lyrics = re.sub('[,"?.\']','',lyrics) #remove comma, quotes,questionmark,period
    lyrics = re.sub('\n+','\n',lyrics) # replace double newlines with just newline
    return lyrics

processed_lyrics = []
for track in lyrics_list:
    artist = track[0]
    song = track[1]
    lyric = track[2]
    processed_lyric = preprocess_lyrics(lyric)
    #processed_lyric = re.sub('\n\n','\n',processed_lyric) # replace double newlines with just newline
    processed_lyrics.append(processed_lyric)



## build lyric corpus & tokenizer
corpus = []
for lyric_ in processed_lyrics:
    corpus += lyric_.lower().split("\n")
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
with open('./model/tokenizer.pickle','wb') as pfile:
    pickle.dump(tokenizer,pfile)
#print("tokenizer saved")
#exit()

total_words = len(tokenizer.word_index) + 1


# break down each line into multiple training data
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)     


# pad sequences to match length
max_sequence_len = max([len(x) for x in input_sequences])
#print(max_sequence_len)
#exit()
input_sequences = np.array(pad_sequences(input_sequences,   
                      maxlen=max_sequence_len, padding='pre'))


# use the last word as label for model to predict
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
label = to_categorical(label, num_classes=total_words)


### Create Model & Train
def create_model(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=1)
    return model

model = create_model(predictors, label, max_sequence_len, total_words)


model.save('./model/model.h5')

print("Model saved to model/model.h5")


