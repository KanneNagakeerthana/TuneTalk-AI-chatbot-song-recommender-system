import streamlit as st
import numpy as np
from keras.models import load_model
from transformers import pipeline
import requests
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pickle

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

#emotion_classifier = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

model = load_model('chatbot_model.h5')

with open('/home/keerthana/mini/intents.json') as f:
    intents = json.load(f)

words = pickle.load(open('/home/keerthana/mini/words.pkl','rb'))
classes = pickle.load(open('/home/keerthana/mini/classes.pkl','rb'))

api_key = 'eBmzLr604XidjDWs5p4VI_QFm-io6Nx6Nptvs&GRZGzv'
nlu_service_url = 'https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/ddf94ac1-9ea4-48a5-8e31-e6b3beee68fa'
lastfm_api_key = '5de6f4d0db4715f4e9f836d5ec8f31ec'

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent":classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break
    return result


def analyze_emotion(messages):
    text_to_analyze = " ".join(messages[-5:])
    authenticator = IAMAuthenticator(api_key)
    nlu = NaturalLanguageUnderstandingV1(
            version='2021-08-01',
            authenticator=authenticator
    )
    nlu.set_service_url(nlu_service_url)
    response = nlu.analyze(
            text=text_to_analyze,
            features=Features(emotion=EmotionOptions())
    ).get_result()
    emotion = response["emotion"]["document"]["emotion"]
    dominant_emotion = max(emotion, key=emotion.get)
    return dominant_emotion

def get_song_recommendation(emotion):
    url = f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={emotion}&api_key={lastfm_api_key}&format=json&limit=10"
    response = requests.get(url)
    payload = response.json()

    song_recommendations = []
    for track in payload['tracks']['track']:
    #for i in range(10):
        #track = payload['tracks']['track'][i]
        song_name = track['name']
        song_url = track['url']
        song_recommendations.append(f"{song_name} ({song_url})")

    return song_recommendations

"""def analyze_emotion(messages):
    text_to_analyze = " ".join(messages[-5:])
    authenticator = IAMAuthenticator(api_key)
    nlu = NaturalLanguageUnderstandingV1(
            version='2021-08-01',
            authenticator=authenticator
    )
    nlu.set_service_url(nlu_service_url)
    response = nlu.analyze(
            text=text_to_analyze,
            features=Features(emotion=EmotionOptions())
    ).get_result()
    emotion = response["emotion"]["document"]["emotion"]
    dominant_emotion = max(emotion, key=emotion.get)
    return dominant_emotion
"""
def get_recommendations():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.chat_history.append(f"You: {user_input}")

        try:
            ints = predict_class(user_input)
            res = get_response(ints, intents)
            st.session_state.chat_history.append(f"TuneTalk: {res}")

            dominant_emotion = analyze_emotion(st.session_state.chat_history)
            st.session_state.chat_history.append(f"TuneTalk: Detected emotion - {dominant_emotion}")

            response = get_song_recommendation(dominant_emotion)
            if isinstance(response, list):
                recommendations = ', '.join([f"[{song.split('(')[0].strip()}]({song.split('(')[1].strip()[:1]})" for song in response])
                st.session_state.chat_history.append(f"TuneTalk: {recommendations}")
            else:
                st.session_state.chat_history.append("TuneTalk: Sorry, I couldn't find any recommendations.")
            #st.text_input("Your message:", key="user_input")

            st.session_state.user_input = ""

        except Exception as e:
            st.session_state.chat_history.append(f"TuneTalk: An error occured: {str(e)}")
        #st.session_state.user_input = ""

    if st.session_state.get("clear_input_field"):
        st.text_input("Your message:", key="user_input", value="")
        st.session_state.clear_input_field = False

st.title("TuneTalk Chatbot")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

for message in st.session_state.chat_history:
    st.markdown(message)

user_input = st.text_input("Your message:", key="user_input")
if st.button("Send"):
    get_recommendations()
    #user_input.empty()
#st.text_input("Your message:", key="user_input", on_change=get_recommendations)
