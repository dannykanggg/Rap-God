# Rap-God
The birth of a new AI Rapper

Small data science project using lyric data from Spotify and Genius.com API. 
Starting from Eminem, use the Spotify API to get 20 similar artists and get the lyric data of their top 10 tracks. 

With this lyric data, I trained an LSTM model to learn the words to the lyrics and generate its own lyrics. 
I figure the model would perform better if it was trained on a bigger variety of lyrics, but due to limited resources, I kept it small.

The model was wrapped with Docker and deployed to Heroku. 
You can try it yourself through the link below!
https://rap-god.herokuapp.com/

As expected, the AI Rapper uses a lot of foul languages, so I decided to censor them out.
