from argparse import ArgumentParser
import yaml
from my_modules import (
    make_datasets, build_model, optiFunc, train
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
    csv_path = "../../products/output/csv/ohkawa_interval_labels.csv"
    print("csv_path={}" .format(csv_path))

    x_train, x_val = make_datasets(csv_path, input_timesteps, 0)
    t_train, t_val = make_datasets(csv_path, input_timesteps, 1)

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
        "--config_file", "-c", default="./config.yml",
        help="configuration file")
    return parser.parse_args()

if __name__ == '__main__':
    main() 