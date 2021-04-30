from argparse import ArgumentParser
import yaml
from tensorflow.python.keras.models import load_model
from my_modules import (
    make_datasets, prediction, save_file
)

def main():
    args = parse_args()
    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)
    
    input_timesteps = config["input_timesteps"]
    hidden_states = config["hidden_states"]
    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]
    class_weight = config["class_weight"]
    csv_path =  args.input_file
    model_path = args.model_file
    output_path = args.output_file

    x_train, x_val = make_datasets(csv_path, input_timesteps, 0)
    t_train, t_val = make_datasets(csv_path, input_timesteps, 1)

    model = load_model(model_path)
    print(model.summary())

    y = prediction(model, input_timesteps, batch_size, x_train)
    save_file(y, output_path)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config_file', '-c', default='config.yml',
        help="configuration file.")
    parser.add_argument(
        '--input_file', '-i', default='../../products/output/csv/ohkawa_interval_labels.csv',
        help='path of input csv file.')
    parser.add_argument(
        '--model_file', '-m', default='../../products/saved_models/LSTM_ohkawa_gen.h5',
        helt='path of LSTM model file.')
    parser.add_argument(
        '__output_file', '-o', default='../../products/output/csv/ohkawa_predicted.csv',
        help='path of output csv file.')
    return parser.parse_args()

if __name__ == '__main__':
    main() 