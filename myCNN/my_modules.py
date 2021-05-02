import numpy as np
import csv
import librosa
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, TimeDistributed
from tensorflow.keras import backend as K

def load_file(audio_path, csv_path):
    y1, sr = librosa.load(audio_path)
    y1_stft = np.abs(librosa.stft(y1))** 2
    y1_logstft = librosa.power_to_db(y1_stft)
    y1_mel = librosa.feature.melspectrogram(S=y1_logstft, n_mels=80)

    with open(csv_path) as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        y2 = [row for row in reader]
    y2 = np.array(y2, dtype=np.float)
    y2 = y2.reshape(-1,)
    return y1_mel, y2

def make_datasets(audio_path, csv_path, seq_length, flag):
    y1, y2 = load_file(audio_path, csv_path)
    x = []
    t = []
    for i in tqdm(range(y1.shape[1] - 15)):
        x.append(y1[:, (i + 1):(i + 16)] - y1[:, i:(i + 15)])
        t.append(y2[i + 7])
    x = np.array(x)
    t = np.array(t)
    x_seq = []
    t_seq = []
    for i in tqdm(range(x.shape[0] - 5)):
        x_seq.append(x[i:i + seq_length])
        t_seq.append(t[i + seq_length - 1])
    x_seq = np.array(x_seq)
    t_seq = np.array(t_seq)
    x_seq = np.array(x_seq).reshape(-1, seq_length, 80, 15, 1)
    t_seq = np.array(t_seq).reshape(t_seq.shape[0], 1)
    x_seq = x_seq.astype('float32')
    x_seq /= np.amax(np.abs(x))
    x_train, x_val, t_train, t_val = train_test_split(x_seq, t_seq, test_size=0.2, shuffle=False)
    if flag == 0: return x_train, x_val
    else: return t_train, t_val

def build_model(seq_length):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(10, kernel_size=(3, 7), activation='relu'), input_shape=(seq_length, 80, 15, 1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 1))))
    model.add(TimeDistributed(Conv2D(20, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 1))))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model

def precision(y_true, y_pred):
    pred_labels = tf.round(y_pred)
    true_pred = K.sum(y_true * pred_labels)
    total_pred = K.sum(y_pred)
    return true_pred / (total_pred + K.epsilon())

def recall(y_true, y_pred):
    true_positives = K.sum(y_true * y_pred)
    total_positives = K.sum(y_true)
    return true_positives / (total_positives + K.epsilon())

def prediction(model, input_timesteps, batch_size, x_train):
    gen = [None for i in range(input_timesteps)]
    z = x_train[:1]
    for i in tqdm(range(len(x_train)-input_timesteps)):
        preds = model.predict(z[-1:], batch_size=batch_size)[0].argmax()
        z = np.append(z, preds)[1:]
        z = z.reshape(-1, input_timesteps, 1)
        gen.append(preds)
    return gen

def save_file(y, output_path):
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(y)
