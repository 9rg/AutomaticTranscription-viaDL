from argparse import ArgumentParser
import yaml
from my_modules import (
    make_datasets, build_model, optiFunc, train
)

def main():
    args = parse_args()
    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)
    
    weight0 = config["weight0"]
    weight1 = config["weight1"]
    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]
    seq_length = config["seq_length"]
    audio_path = args.audio_path
    csv_path = args.csv_path

    x_train, x_val = make_datasets(audio_path, csv_path, seq_length, 0)
    t_train, t_val = make_datasets(audio_path, csv_path, seq_length, 1)

    model = build_model(hidden_states, input_timesteps, batch_size, t_train)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optiFunc(),
        metrics=["accuracy"]
    )

    train(model, x_train, x_val, t_train, t_val, class_weight, n_epochs)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config_file', '-c', default='./config.yml',
        help='configuration file')
    parser.add_argument(
        '--audio_path', '-a', default='../../musics/shinobue.wav',
        help='path of shinobue file for input.')
    parser.add_argument(
        '--csv_path', '-c', default='../../output/csv/ohkawa_beats_start2.csv',
        help='path of drum file for input.')
    return parser.parse_args()

if __name__ == '__main__':
    main() 