import re

import nltk
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from flask_cors import cross_origin
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("stopwords")

app = Flask(__name__)

model = tf.keras.models.load_model('SpamClassifier.h5')

lemmatizer = WordNetLemmatizer()


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # text
        text = request.form['text']
        corpus = []
        review = re.sub('[^a-zA-Z0-9]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if word
                  not in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

        voc_size = 10000
        onehot_repr = [tf.keras.preprocessing.text.one_hot(
            words, voc_size) for words in corpus]
        max_sentence_length = 78
        embedded_docs = pad_sequences(onehot_repr, padding='pre',
                                      maxlen=max_sentence_length)
        x_final = np.array(embedded_docs)

        prediction = model.predict([x_final])

        spam_message = "The message is a spam message. Be safe."
        ham_message = "The message is not a spam. Don't worry. . "
        output = prediction[0][0]
        if output < 0.6:
            return render_template('home.html',
                                   prediction_text=spam_message)
        else:
            return render_template('home.html',
                                   prediction_text=ham_message)

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True, port=7777)
