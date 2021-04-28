import numpy as np
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

def make_datasets(file_path, input_timesteps, flag):
    with open(file_path) as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        y = [row for row in reader]
    y = np.array(y, dtype=np.float)
    y = y.reshape(-1,)
    length_of_sequences = len(y)

    y_onehot = np.zeros((length_of_sequences, 4))
    for i in tqdm(range(length_of_sequences)):
        if y[i] == 0:
            y_onehot[i, 0] = 1
        if y[i] == 1:
            y_onehot[i, 1] = 1
        if y[i] == 2:
            y_onehot[i, 2] = 1
        if y[i] == 3:
            y_onehot[i, 3] = 1

    x = []
    t = []
    for i in tqdm(range(length_of_sequences - input_timesteps)):
        x.append(y[i:i + input_timesteps])
        t.append(y_onehot[i + input_timesteps])
    x = np.array(x).reshape(-1, input_timesteps, 1)
    t = np.array(t)
    x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=0.2, shuffle=False)
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
