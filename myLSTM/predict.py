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
    csv_path = config["csv_path"]
    model_path = config["model_path"]
    output_path = config["output_path"]

    x_train, x_val = make_datasets(csv_path, input_timesteps, 0)
    t_train, t_val = make_datasets(csv_path, input_timesteps, 1)

    model = load_model(model_path)
    print(model.summary())

    y = prediction(model, input_timesteps, batch_size, x_train)
    save_file(y, output_path)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default="./config.yml",
        help="configuration file")
    return parser.parse_args()

if __name__ == '__main__':
    main() 