import argparse
import os
from shutil import copyfile, rmtree

import click
import torch

from loaders import get_loader
from confidnet.learners import get_learner
from confidnet.utils.logger import get_logger
from confidnet.utils.misc import load_yaml, set_seed
# from confidnet.utils.tensorboard_logger import TensorboardLogger

LOGGER = get_logger(__name__, level="DEBUG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    
    args = parser.parse_args()

    config_args = load_yaml(args.config_path)

    seed = config_args.get('seed', 0)

    set_seed(seed)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")
    dloader = get_loader(config_args)

    # Make loaders
    dloader.make_loaders()

    # force train from scratch, so set start epoch to zero.
    start_epoch = 0
    
    # Set learner
    LOGGER.warning(f"Learning type: {config_args['training']['learner']}")
    learner = get_learner(
        config_args,
        dloader.train_loader,
        dloader.val_loader,
        dloader.test_loader,
        start_epoch,
        device,
    )
    
    # Load pretrained base model
    pretrained_checkpoint = torch.load(config_args["model"]["resume"], map_location=device)
    
    # learner.load_checkpoint(pretrained_checkpoint["model_state_dict"], strict=False)
    learner.load_checkpoint(pretrained_checkpoint, strict=False)
    
    # Log files
    LOGGER.info(f"Using model {config_args['model']['name']}")
    learner.model.print_summary(
        input_size=tuple([shape_i for shape_i in learner.train_loader.dataset[0][0].shape])
    )

    # create output directory if not exists
    if os.path.exists(config_args["training"]["output_folder"]):
      rmtree(config_args["training"]["output_folder"])
      
    # create output directory
    os.makedirs(config_args["training"]["output_folder"])
    
    copyfile(
        args.config_path, config_args["training"]["output_folder"] / f"config_{start_epoch}.yaml"
    )
    LOGGER.info(
        "Sending batches as {}".format(
            tuple(
                [config_args["training"]["batch_size"]]
                + [shape_i for shape_i in learner.train_loader.dataset[0][0].shape]
            )
        )
    )
    LOGGER.info(f"Saving logs in: {config_args['training']['output_folder']}")

    # Parallelize model
    nb_gpus = torch.cuda.device_count()
    if nb_gpus > 1:
        LOGGER.info(f"Parallelizing data to {nb_gpus} GPUs")
        learner.model = torch.nn.DataParallel(learner.model, device_ids=range(nb_gpus))

    # Set scheduler
    learner.set_scheduler()

    # Start training
    for epoch in range(start_epoch, config_args["training"]["nb_epochs"] + 1):
        learner.train(epoch)

if __name__ == "__main__":
    main()
