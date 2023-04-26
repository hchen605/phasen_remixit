import os
import random
import argparse

import json5
import numpy as np
import torch
from torch.utils.data import DataLoader

from util.utils import initialize_config


def main(config, resume):

    # Random seed for both CPU and GPU.
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    collate_fn=initialize_config(config["train_dataloader"]["collate_fn"], pass_args=False)

    train_dataloader = DataLoader(
        dataset=initialize_config(config["train_dataset"]),
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        collate_fn=collate_fn,
        pin_memory=config["train_dataloader"]["pin_memory"],
    )

    valid_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        num_workers=0,
        batch_size=1
    )

    teacher_model = initialize_config(config["model"])
    student_model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=student_model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    if config["lr_scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['lr_scheduler']['decay_factor'],
            patience=config['lr_scheduler']['patience'],
            min_lr=1e-5
        )
    else:
        scheduler = None

    trainer_class = initialize_config(config["trainer"], pass_args=False)

    trainer = trainer_class(
        config=config,
        resume=resume,
        teacher_model=teacher_model,
        student_model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CRN")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
    parser.add_argument("-P", "--preloaded_model_path", type=str, help="Path of the *.Pth file of the model.")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    args = parser.parse_args()

    if args.preloaded_model_path:
        assert not args.resume, "Resume conflict with preloaded model. Please use one of them."

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration
    configuration["preloaded_model_path"] = args.preloaded_model_path

    main(configuration, resume=args.resume)
