import numpy as np 
import random
import keras
import matplotlib.pyplot as plt
import tensorflow 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

#loading dataset
X_train = np.loadtxt('input.csv',delimiter=',')
y_train = np.loadtxt('labels.csv', delimiter=',')

X_test = np.loadtxt('input_test.csv',delimiter=',')
y_test = np.loadtxt('labels_test.csv', delimiter=',')
'''''
# lets prints the shape of model
print(" shape of x_train : ", X_train.shape)
print(" shape of y_train :", y_train.shape)
print(" shape of x_test : ", X_test.shape)
print(" shape of y_test :", y_test.shape)
'''
# reshape the size of img 
X_train = X_train.reshape(len(X_train),100, 100, 3)
y_train = y_train.reshape(len(y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
y_test = y_test.reshape(len(y_test),1)
# to train our model properly we need to rescale this values b/w 0 to 1
X_train = X_train/255.0
X_test = X_test/255.0
 
 #lets shows thge random train img
idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx,:])
#plt.show()

# create the model classic the img 

model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)),
    MaxPooling2D((2,2,)),
    
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64,activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])
model = Sequential ()
model.add(Conv2D(32, (3,3), activation= 'relu', input_shape = (100, 100, 3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64,activation ='relu'))
model.add(Dense(1, activation = 'sigmoid'))
# we are going to compile this model by adding the loss and the back propagation algorithm 
#https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class
#https://keras.io/api/optimizers/
model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 5, batch_size = 64)
#lets evaluate the performance of train dataset
model.evaluate(X_train, y_train)
#lest evaluate the performance of test dataset
model.evaluate(X_test, y_test)
# lets predictions the img from  dataset
idx2 = random.randint(0, len(X_test))
plt.imshow(X_test[idx2, :]) 
y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

if(y_pred == 0):
    pred = 'dog'
else:
    pred = 'cat'

plt.print(" In given animals is  :",pred)    
plt.show()