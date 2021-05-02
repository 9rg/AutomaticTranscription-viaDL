from argparse import ArgumentParser
import yaml
from tensorflow.keras.optimizers import Adagrad
from my_modules import (
    make_datasets, build_model, precision, recall
)

def main():
    args = parse_args()
    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)
    
    class_weight = config["class_weight"]
    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]
    seq_length = config["seq_length"]
    audio_path = args.audio_path
    csv_path = args.csv_path

    x_train, x_val = make_datasets(audio_path, csv_path, seq_length, 0)
    t_train, t_val = make_datasets(audio_path, csv_path, seq_length, 1)

    model = build_model(seq_length)
    model.compile(loss='binary_crossentropy', optimizer=Adagrad(), metrics=['accuracy', precision, recall])
    history = model.fit(x_train, t_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(x_val, t_val), class_weight=class_weight)

    print(model.summary())

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config_file', '-c', default='./config.yml',
        help='configuration file')
    parser.add_argument(
        '--audio_path', '-a', default='../../musics/shinobue.wav',
        help='path of shinobue file for input.')
    parser.add_argument(
        '--csv_path', '-v', default='../../products/output/csv/ohkawa_beats_start2.csv',
        help='path of drum file for input.')
    return parser.parse_args()

if __name__ == '__main__':
    main() 