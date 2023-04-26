import argparse

import json5

from util.utils import initialize_config


def main(config, checkpoint_path, output_dir):
    inferencer_class = initialize_config(config["inference"], pass_args=False)
    inferencer = inferencer_class(
        config,
        checkpoint_path,
        output_dir
    )
    inferencer.inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference / Enhancement / Test")
    parser.add_argument("-mc", "--model_config", type=str, required=True, help="Inference config file.")
    parser.add_argument("-dc", "--dataset_config", type=str, required=True)
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True, help="Model checkpoint")
    parser.add_argument("-dist", "--output_dir", type=str, required=True, help="The dir of saving enhanced wav files.")
    args = parser.parse_args()

    dataset_config = json5.load(open(args.dataset_config))
    model_config = json5.load(open(args.model_config))
    dataset_config["model"] = model_config["model"]
    main(dataset_config, args.checkpoint_path, args.output_dir)
