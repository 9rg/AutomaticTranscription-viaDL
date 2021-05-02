import numpy as np
import csv
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

def build_model(hidden_states, input_timesteps, batch_size, t_train):
    model = Sequential()
    model.add(LSTM(hidden_states, stateful=True, input_shape=(input_timesteps, 1), batch_size=batch_size))
    model.add(Dense(t_train.shape[1], activation='softmax'))
    return model

def optiFunc():
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    return optimizer

def train(model, x_train, x_val, t_train, t_val, class_weight, n_epochs):
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    for i in tqdm(range(n_epochs)):
        history = model.fit(x_train, t_train, epochs=1, verbose=2, batch_size=1, validation_data=(x_val, t_val), callbacks=[es], shuffle=False, class_weight=class_weight)
        model.reset_states()
    print(model.summary())

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
