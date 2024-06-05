import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Pad the sequences to make them all the same length
data_padded = pad_sequences(data_dict['data'], padding='post', dtype='float32')

# Convert the padded data to a NumPy array
data = np.asarray(data_padded)
labels = np.asarray(data_dict['labels'])

# Flatten the 3D array to a 2D array
data_flat = np.reshape(data, (data.shape[0], -1))

x_train, x_test, y_train, y_test = train_test_split(data_flat, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
