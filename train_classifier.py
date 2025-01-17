import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

imgs_dict = pickle.load(open('./imgs.pickle', 'rb'))
imgs = np.asarray(imgs_dict['imgs'])
labels = np.asarray(imgs_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% classification done.'.format(score * 100))

file = open('model.p', 'wb')
pickle.dump({'model': model}, file)
file.close()
