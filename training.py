import random
import json
import pickle
import numpy as np

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words=[]
classes=[]
documents=[]
ignore_letters=['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes=sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
# print(words)

training = []

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Split into X (patterns) and y (intents)
train_X = np.array([pattern for pattern, _ in training])
train_y = np.array([intent for _, intent in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(train_X, train_y, epochs=200, batch_size=5, verbose=1)

model.save('chatbot_model.model')
print("done")
