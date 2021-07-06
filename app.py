from flask import Flask, render_template, request
from pickle import load
import numpy as np
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

from numpy import array
from pickle import load
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# do your ML task
tf.compat.v1.keras.backend.get_session().close();






app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

def define_model(vocab_size, max_length):
	# feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.load_weights('model_19_weights.h5')
    return model
	# summarize model
	# model.summary()
	# plot_model(model, to_file='model.png', show_shapes=True)



# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    file = request.files["file"]
    file.save('static/file.jpg')
    # load the tokenizer
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 34
    vocab_size = 7577
    #model = define_model(vocab_size, max_length)
    
    # # load the model
    model = load_model('model_19.h5')
    
    photo = extract_features('static/file.jpg')
    
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    print(description)
    query = description
    stopwords = ['startseq','endseq']
    querywords = query.split()
    print("good")

    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)

    print(result)
    return render_template("predict.html", output=result)


if __name__ == "__main__":
    app.run(debug=True)
