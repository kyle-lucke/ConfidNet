import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from confidnet.loaders import get_loader
from confidnet.learners import get_learner
from confidnet.models import get_model
from confidnet.utils import trust_scores
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics
from confidnet.utils.misc import load_yaml

LOGGER = get_logger(__name__, level="DEBUG")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
    
    args = parser.parse_args()

    args.mode = confidnet
    
    config_args = load_yaml(args.config_path)

    # Overwrite for release
    config_args["training"]["output_folder"] = Path(args.config_path).parent

    config_args["training"]["metrics"] = [
        "accuracy",
        "auc",
        "ap_success",
        "ap_errors",
        "fpr_at_95tpr",
        "aurc"
    ]

    if config_args["training"]["task"] == "segmentation":
        config_args["training"]["metrics"].append("mean_iou")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")
    dloader = get_loader(config_args)

    # Make loaders
    dloader.make_loaders()

    # Set learner
    LOGGER.warning(f"Learning type: {config_args['training']['learner']}")
    learner = get_learner(
        config_args, dloader.train_loader, dloader.val_loader, dloader.test_loader, -1, device
    )

    # Initialize and load model
    ckpt_path = config_args["training"]["output_folder"] / f"best.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    learner.model.load_state_dict(checkpoint["model_state_dict"])

    # Get scores
    LOGGER.info(f"Inference mode: {args.mode}")

    _, scores_test = learner.evaluate(
        learner.test_loader,
        learner.prod_test_len,
        split="test",
        mode=args.mode,
        samples=args.samples,
        verbose=True,
    )

    LOGGER.info("Results")
    print("----------------------------------------------------------------")
    for st in scores_test:
        print(st)
        print(scores_test[st])
        print("----------------------------------------------------------------")


if __name__ == "__main__":
    main()
