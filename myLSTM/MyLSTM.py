#prediction that uses interval data stateful

import numpy as np
import os, sys
import csv
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

##params
#timesteps
maxlen = 15
#hidden_states
n_hidden = 32
#epochs
n_epochs = 100


#load data
print("Loading...")
with open('output/csv/ohkawa_interval_labels.csv') as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    y = [row for row in reader]

y = np.array(y, dtype=np.float)
y = y.reshape(-1,)
y = y[:1000]
#もともと1000
print("shape of y:{}".format(y.shape))

#total length of sequence
length_of_sequences = len(y)


y_onehot = np.zeros((1175, 4))
x = []
t = []

'''
#ラベルをone-hot vector化(ライブラリを使う手法)
y_onehot = np_utils.to_categorical(y)
print("Shape of y_onehot:{}".format(y_onehot.shape))
print("contents of y_onehot:{}" .format(y_onehot))
'''

print("Making datasets...")
#convert targetdata to one-hot vector
for i in tqdm(range(length_of_sequences)):
    if y[i] == 0:
        y_onehot[i, 0] = 1
    if y[i] == 1:
        y_onehot[i, 1] = 1
    if y[i] == 2:
        y_onehot[i, 2] = 1
        '''
        if i < 615:
            y_onehot[i, 2] = 1
        else:
            y_onehot[i, 3] = 1
            '''
    if y[i] == 3:
        y_onehot[i, 3] = 1

'''
explanation
label[0] → [1 0 0 0]
label[1] → [0 1 0 0]
label[2] → [0 0 1 0]
label[3] → [0 0 0 1]
'''

for i in tqdm(range(length_of_sequences - maxlen)):
    x.append(y[i:i + maxlen])
    t.append(y_onehot[i + maxlen])

x = np.array(x).reshape(-1, maxlen, 1)
t = np.array(t)

'''
print("Shape of x:{}".format(x.shape))
print("Contents of x:\n{}".format(x))
print("Shape of t:{}".format(t.shape))
print("Contents of t:\n{}".format(t))
'''

x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=0.2, shuffle=False)
print("Shape of x_val:{}".format(x_val.shape))
print("Shape of x_val[:1]:{}".format(x_val[:1].shape))
print("Shape of t_val:{}".format(t_val.shape))
print("Succeeded in making datasets.\n")


#build model
model = Sequential()
model.add(LSTM(n_hidden, stateful=True, input_shape=(maxlen, 1), batch_size=1))
model.add(Dense(t.shape[1], activation='softmax'))
optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

#learning
print("Start to fit...")
class_weight = {
    0: 1.30,
    1: 1.23,
    2: 0.49,
    3: 2.59
}
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
for i in tqdm(range(n_epochs)):
    history = model.fit(x_train, t_train, epochs=1, verbose=2, batch_size=1, validation_data=(x_val, t_val), callbacks=[es], shuffle=False, class_weight=class_weight)
    model.reset_states()
print("Succeeded in fitting.\n")
print(model.summary())
plot_model(
    model,
    show_shapes=True,
)



#Print confusion_matrix
predict_classes = np.zeros(len(t_val))
true_classes = np.zeros(len(t_val))

for i in tqdm(range(len(t_val))):
    predict_classes[i] = model.predict(x_val, batch_size=1)[i:i+1].argmax()
    true_classes[i] = t_val[i].argmax()
names = ['label:0', 'label:1', 'label:2', 'label:3']
print("=====Params=====")
print("timesteps={}, n_hidden={}, n_epochs={}" .format(maxlen, n_hidden, n_epochs))
print("\n=====Confusion_matrix:=====")
print(confusion_matrix(true_classes, predict_classes))
print("\n=====Classification_report:=====")
print(classification_report(true_classes, predict_classes, target_names=names))



#Show graphs
metrics = ['loss', 'accuracy']
fig = plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意
for i in range(len(metrics)):
    metric = metrics[i]
    plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title(metric)  # グラフのタイトルを表示
    
    plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
    plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
    
    plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
    plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()  # ラベルの表示

plt.show()  # グラフの表示


#save model
#model.save('LSTM_ohkawa_gen4.h5')

'''
#######ここに…
#One step ahed forecast 
gen = [None for i in range(maxlen)]
z = x[:1]

print("Predicting...")
preds = model.predict(z[-1:], batch_size=1)[0].argmax()
print("Shape of preds : {}".format(preds.shape))
print("contents of preds :\n{}".format(preds))
print("answer(t[0])is {}".format(t[0]))
print("contents of z:\n{}" .format(z))
z = np.append(z, preds)[1:]
z = z.reshape(-1, maxlen, 1)
print("contents of z:\n{}" .format(z))

for i in tqdm(range(length_of_sequences - maxlen)):
    preds = model.predict(z[-1:], batch_size=1)[0].argmax()
    z = np.append(z, preds)[1:]
    z = z.reshape(-1, maxlen, 1)
    gen.append(preds)


#output
with open('output/csv/ohkawa_interval_predicted_ful.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(gen)
'''


print("Complete.\n")