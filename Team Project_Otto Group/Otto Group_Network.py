import keras
import pandas as pd
import numpy as np
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout, MaxPooling2D,BatchNormalization
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import Sequential
import tensorflow as tf
from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def to_one_hot(y):
    y_temp = np.zeros([y.shape[0], 10])  # 转换为one-hot向量
    for i in range(y.shape[0]):
        y_temp[i, y[i]] = 1
    return y_temp

# 读取数据
train_data = pd.read_csv(open("train.csv"))
test_data = pd.read_csv(open("test.csv"))
print(train_data.info())
train_y_raw = train_data["target"]
x_label = []
for i in range(1,94):
    x_label.append("feat_%s"%(i))
train_x = np.array(train_data[x_label])
test_x = np.array(test_data[x_label])

# 将train_y中形如Class_1的数据转换成one_hot向量，9维
train_y = np.zeros([len(train_y_raw),9])
for i in range(len(train_y_raw)):
    lable_data = int(train_y_raw[i][-1])  # 取最后一个字就行
    train_y[i,lable_data-1] = 1
print(train_x.shape,train_y.shape,test_x.shape);  # (49502, 93) (49502, 9) (12376, 93)

dim = train_x.shape[1]
print(dim, 'dims')
print('Building model')

nb_classes = train_y.shape[1]

model = Sequential()

model.add(Dense(256, input_shape=(dim, )))
model.add(Activation('relu'))
model.add(Dropout(0.005))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.005))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.005))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.005))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#93 dims
#Building model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 1024
epochs = 100


fBestModel = 'best_model7.h5'
early_stop = EarlyStopping(monitor='val_acc', patience=5, verbose=1)
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1, callbacks=[best_model, early_stop])


prediction = model.predict(test_x)

num_pre = prediction.shape[0]
columns = ['Class_'+str(post+1) for post in range(9)]

df2 = pd.DataFrame({'id' : range(1,num_pre+1)})
df3 = pd.DataFrame(prediction, columns=columns)

df_pre = pd.concat([df2, df3], axis=1)

df_pre.to_csv('predition.csv', index=False)
