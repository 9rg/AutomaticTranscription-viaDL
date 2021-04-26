#CRNNを使って開始地点予測

#import libraries
import numpy as np
import csv
import random
import librosa
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

#################################
#           Parameters          #
#################################
audio_path = "../musics/shinobue.wav"
csv_path = "output/csv/ohkawa_beats_start2.csv"
weight0 = 0.50
weight1 = 107.96
batch_size = 3
n_epochs = 50


#################################
#           Load Datas          #
#################################
print("\n=====Loading shinobue data...=====")
y1, sr = librosa.load(audio_path)
y1_stft = np.abs(librosa.stft(y1))**2
y1_logstft = librosa.power_to_db(y1_stft)
y1_mel = librosa.feature.melspectrogram(S=y1_logstft, n_mels=80)
print("Shape of y1_mel:{}".format(y1_mel.shape))

print("\n=====Loading ohkawa data...=====")
with open(csv_path) as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    y2 = [row for row in reader]

y2 = np.array(y2, dtype=np.float)
y2 = y2.reshape(-1,)
print("Shape of y2:{}".format(y2.shape)) 


#################################
#         Making Dataset        #
#################################
print("\n=====Making Dataset...=====")
x = []
t = []

#最初の左端7フレームと最後の右端7フレーム分を除いた区間
for i in tqdm(range(y1_mel.shape[1] - 14)):
    x.append(y1_mel[:, i:i + 15])
    t.append(y2[i + 7])

x = np.array(x)
t = np.array(t)
x = np.array(x).reshape(x.shape[0], 80, 15, 1)
t = np.array(t).reshape(t.shape[0], 1)

x_T = [] #sequence_lengthでまとまったデータ(系列データ))
t_T = [] #上の正解データ(まとまりの最後の列をtargetとしている)
sequence_length = 20
for i in tqdm(range(x.shape[0] - sequence_length - 13)):
    x_T.append(x[i:i + sequence_length,:,:,:])
    t_T.append(t[i + sequence_length + 13])
    
x_T = np.array(x_T)
t_T = np.array(t_T)
print("Shape of x_T:{}".format(x_T.shape))
print("Shape of t_T:{}".format(t_T.shape))

x_train, x_val, t_train, t_val = train_test_split(x_T, t_T, test_size=0.2, shuffle=False)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= np.amax(np.abs(x))
x_val /= np.amax(np.abs(x))


#################################
#         organize model        #
#################################
print("Organize faze")
#CNN
cnn = Sequential()
cnn.add(Conv2D(10, kernel_size=(3, 7), activation='relu', input_shape=(80, 15, 1)))
cnn.add(MaxPooling2D(pool_size=(3, 1)))
cnn.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(3, 1)))
cnn.add(Flatten())

#LSTM
rnn = Sequential()
rnn = LSTM(64, return_sequences=False, stateful=True, input_shape=(sequence_length, 1120), batch_size=batch_size)

#Fully connected
dense = Sequential()
dense.add(Dense(256, activation='relu'))
dense.add(Dense(128, activation='relu'))
dense.add(Dense(1, activation='sigmoid'))

main_input = Input(batch_shape=(batch_size, sequence_length, 80, 15, 1))
#data has been reshaped to (800, 5, 120, 60, 1)
#In my case, this will be maybe (sequence_len, 80, 15, 1)

model = TimeDistributed(cnn)(main_input)    #add cnn with timedistributed()
model = rnn(model)                          #add rnn
model = dense(model)                        #add dense
final_model = Model(inputs=main_input, outputs=model)


#######################################
#               Learning              #
#######################################
print("Start compiles")
start = time.time()
class_weight = {
    0: weight0,
    1: weight1
}
#original metric function
def precision(y_true, y_pred):
    pred_labels = tf.round(y_pred)
    true_pred = K.sum(y_true * pred_labels)
    total_pred = K.sum(y_pred)
    return true_pred / (total_pred + K.epsilon())
#compile & learning
final_model.compile(loss='binary_crossentropy', optimizer=Adagrad(), metrics=['accuracy', precision])
for i in tqdm(range(n_epochs)):
    history = final_model.fit(x_train, t_train, batch_size=batch_size, epochs=1, verbose=1, validation_data=(x_val, t_val), class_weight=class_weight, shuffle=False)
    final_model.reset_states()
print(final_model.summary())


#######################################
#             Evaluation              #
#######################################
predict_classes = np.zeros(len(t_train))
true_classes = np.zeros(len(t_train))

preds = final_model.predict(x_train, batch_size=batch_size)

for i in tqdm(range(len(t_train))):
    if (preds[i:i + 1] > 0.5):
        predict_classes[i] = 1
    else:
        predict_classes[i] = 0
    true_classes[i] = t_train[i]
print("\n===== ('ω `)<Params =====")
print("w0:{}, w1:{}, batch_size:{}, n_epochs:{}".format(weight0, weight1, batch_size, n_epochs))
print("\n===== ('ω `)<Confusion_matrix =====")
print(confusion_matrix(true_classes, predict_classes))
names = ['label:0', 'label:1']
print("\n===== ('ω `)<Classification_report =====")
print(classification_report(true_classes, predict_classes, target_names=names, digits=4))
t = (time.time() - start)/60
print("Time:{}".format(t))

#Show graphs
metrics = ['loss', 'accuracy', 'precision']
fig = plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意
for i in range(len(metrics)):
    metric = metrics[i]
    plt.subplot(1, 3, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title('batch_size:'+str(batch_size), loc='left')
    plt.title(metric, loc='right')  # グラフのタイトルを表示
    
    plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
    plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
    
    plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
    plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()  # ラベルの表示

plt.show()  # グラフの表示

final_model.save('CRNN_start_detection.h5')

print("Complete.")