# -*- coding: utf-8 -*-
"""TuneTalk.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1if4gM7dNuZF-_nrvGd6y8LaDlQkacr3x
"""

#from google.colab import drive
#drive.mount('/home/keerthana')

#from google.colab import drive
#drive.mount('/content/drive')

#pip install anvil-uplink

#mport anvil.server
#nvil.server.connect("client_3XYBD7A4CFPLMW7PG4FYDCRD-PYPYNXXNZ27MZOWR")

#pip install tensorflow keras pickle nltk
#from google.colab import drive
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
intents_file = open('/home/keerthana/mini/intents.json').read()
intents = json.loads(intents_file)

import nltk
nltk.download('punkt')
words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)

import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

# Assuming you have defined 'documents', 'classes', 'words', and 'lemmatizer' somewhere in your code

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

# Separate patterns and intents
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

print("Training data is created")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
#rom keras.models import Sequential
#rom keras.layers import Dense, Dropout
#rom tf_keras.optimizers import legacy_SGD as SGD
#mport tensorflow as tf
#mport os
os.environ['TF_USE_LEGACY_KERAS']='TRUE'

# Define the deep neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model with the appropriate optimizer and loss function
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5', hist)
print("Model is created and saved.")

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('/home/keerthana/mini/chatbot_model.h5')
import json
import random
intents = json.loads(open('/home/keerthana/mini/intents.json').read())
words = pickle.load(open('/home/keerthana/mini/words.pkl','rb'))
classes = pickle.load(open('/home/keerthana/mini/classes.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))
def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
#Creating tkinter GUI
# import tkinter
# from tkinter import *
# def send():

import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import json
import pickle

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
intents_file = open('/home/keerthana/mini/intents.json').read()
intents = json.loads(intents_file)

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Preprocess intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Add documents in the corpus
        documents.append((word, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save preprocessed data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    bag = np.array(bag)
    output_row = np.array(output_row)

    training.append([bag, output_row])

# Shuffle and convert training data to numpy array
random.shuffle(training)
train_x = np.array([x for x, _ in training])
train_y = np.array([y for _, y in training])
#training = np.array(training)

#train_x = list(training[:, 0])
#train_y = list(training[:, 1])


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
# Define model architecture
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=172, batch_size=5, verbose=1)

# Save model
model.save('chatbot_model.h5', hist)
print("Model created")

#pip install ibm_watson

#mport anvil.server

#nvil.server.connect("server_BM7RB3FBPOMTF5HOZRIKF26Q-PYPYNXXNZ27MZOWR")

#pip install ibm_watson
#pip install requests

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
#import anvil.server
import requests

msg = list()
text = str()

#@anvil.server.callable
def responsed(msg1):
    msg.append(msg1)
    ints = predict_class(msg1)
    res = getResponse(ints, intents)
    return res

api_key = 'xVFFlrweN-yYMf4fesSkAihZDa68VNE-jrULqjUDnJn6'
nlu_service_url = 'https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/a3d00ba5-af6f-40cf-b853-3e0aaef2edf6'
lastfm_api_key = '5de6f4d0db4715f4e9f836d5ec8f31ec'
#@anvil.server.callable
def song_emotion():
    authenticator = IAMAuthenticator(api_key)
    nlu = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )
    nlu.set_service_url(nlu_service_url)

    # Combine the last few messages for analysis
    text_to_analyze = " ".join(msg[-5:])

    response = nlu.analyze(
        text=text_to_analyze,
        features=Features(emotion=EmotionOptions())
    ).get_result()

    dic1 = dict()
    emotion = response["emotion"]["document"]["emotion"]
    dominant_emotion = max(emotion, key=emotion.get)
    dic1['emotion'] = dominant_emotion

    url = f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={dominant_emotion}&api_key={lastfm_api_key}&format=json&limit=10"
    response = requests.get(url)
    payload = response.json()

    for i in range(10):
        r = payload['tracks']['track'][i]
        dic1[r['name']] = r['url']

    return dic1

import requests

def get_song_recommendation(user_input):
    dominant_emotion = 'happy'
    lastfm_api_key = '5de6f4d0db4715f4e9f836d5ec8f31ec'
    url = f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={dominant_emotion}&api_key={lastfm_api_key}&format=json&limit=10"
    response = requests.get(url)
    payload = response.json()

    song_recommendations = []
    for i in range(10):
        track = payload['tracks']['track'][i]
        song_name = track['name']
        song_url = track['url']
        song_recommendations.append(f"{song_name} ({song_url})")

    return song_recommendations

if __name__ == "__main__":
    print("Chatbot: Hey there, Wassup?")
    for i in range(5):
        m = input("User: ")
        res = f"Response to {m}"
        print("Chatbot: "+res)

    ans = {'emotion': 'happy'}
    dominant_emotion = ans['emotion']
    print("Emotion: "+dominant_emotion)

    recommendations = get_song_recommendation(dominant_emotion)
    print("Song Recommendations:")
    for song in recommendations:
        print(song)
"""

import requests

dominant_emotion = 'happy'
lastfm_api_key = '5de6f4d0db4715f4e9f836d5ec8f31ec'
url=f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={dominant_emotion}&api_key={lastfm_api_key}&format=json&limit=10"
response = requests.get(url)
payload = response.json()
# for i in range(4):
r=payload['tracks']['track'][0]
# print(r['url'])
print(payload)

print("Chatbot: Hey there, Wassup?")
for i in range(5):
    m = input("User: ")
    res = anvil.server.call('responsed', m)
    print("Chatbot: " + res)
    #res = responsed(m)
    #print("Chatbot: " + res)
ans = anvil.server.call('song_emotion')
dominant_emotion = ans.pop('emotion')

print("Emotion: " + dominant_emotion)

#ans = song_emotion()
#print("Emotion: " + ans['emotion'])

#ans = song_emotion()
#print("Emotion: " + ans['emotion'])

# Song recommendations
#ans.pop('emotion')
#lst = list(ans.keys())
print("Song Recommendations:")
for i in range(10):
    print("Song_name: " + lst[i])
    print("Song_URL: " + ans[lst[i]])"""
