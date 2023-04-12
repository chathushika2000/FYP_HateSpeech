from flask import (
    Flask,
    render_template,
    request
)
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import joblib
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer



directory = os.getcwd()

app = Flask(__name__ , template_folder = 'template' )

 
model = load_model("model.h5",compile=False)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# model = joblib.load('finalized_model.joblib')
# model  = pickle.load(open('model.pkl', 'rb'))

# with open('finalized_model.sav', 'rb') as inputfile:
#     model = pickle.load(inputfile)


@app.route('/') 
def home():
	return render_template (
		'index.html'
	)


@app.route('/hatespeech') 
def hatesppech():
	return render_template (
		'demo.html'
	)

@app.route('/hatespeech2') 
def hatesppechEnglish():
	return render_template (
		'demo2.html'
	)

@app.route('/select') 
def hatespeechview():
	return render_template (
		'demo-1.html'
	)

@app.route('/submit' , methods=['POST','GET']) 
def submit():
	tokenizer = Tokenizer(num_words=5000, lower=True)
	df = pd.read_csv('mydataset.csv')
	tokenizer.fit_on_texts(df['full_text_with_emoji'])
	sequences = tokenizer.texts_to_sequences(df['full_text_with_emoji'])
	print(request.form)
	# if request.form == "POST":
	hatesppech = request.form['sentence']
	text = [hatesppech]
	sequences = tokenizer.texts_to_sequences(text)
	X_pred = pad_sequences(sequences, maxlen=200)
	text_val = ""
	print(hatesppech)
	# make predictions
	y_pred = model.predict(X_pred)
	print(y_pred)
	actual_val = y_pred[0][0]
	if y_pred[0][0] < 0.6:
		text_val = 'Not hate speech'
	else:
		text_val = 'Hate speech'

	return render_template (
	'demo.html' , val = actual_val , prediction = text_val
	)


@app.route('/submitEnglish' , methods=['POST','GET']) 
def submitEnglish():
	f = open('my_classifier.pickle', 'rb')
	classifier = pickle.load(f)
	f.close()
	print(request.form)
	# if request.form == "POST":
	hatesppech = request.form['sentenceEnglish']
	text = [hatesppech]
	# cv = CountVectorizer()
	# tfidftrans = TfidfVectorizer()
	# val = tfidftrans.transform(hatesppech)
	tf_idf_converter = joblib.load("tf-idf.joblib")

	val = classifier.predict(tf_idf_converter.transform(text))
	actual_val = val[0]
	if val[0] < 0.6:
		text_val = 'Hate speech'
	else:
		text_val = 'Not Hate speech'

	return render_template (
	'demo2.html' , val = actual_val , prediction = text_val
	)


# app.run(port=5000)

if _name_ == "_main_":
    app.run(debug = True)