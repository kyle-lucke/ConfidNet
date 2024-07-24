import os
import json
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

    parser.add_argument('--beta', type=float, default=1.0)
    
    args = parser.parse_args()
    
    config_args = load_yaml(args.config_path)

    # Overwrite for release
    config_args["training"]["output_folder"] = Path(args.config_path).parent

    config_args["training"]["metrics"] = [
        "accuracy",
        "auc",
        "ap_success",
        "ap_errors",
        "fpr_at_95tpr",
        "aurc",
        "spec_sens"
    ]

    if config_args["training"]["task"] == "segmentation":
        config_args["training"]["metrics"].append("mean_iou")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    LOGGER.info(f"Inference mode: confidnet")

    _, scores_test = learner.evaluate(
        learner.test_loader,
        learner.prod_test_len,
        split="test",
        mode='confidnet',
        samples=0,
        verbose=True
    )

    save_scores = {k.replace('test/', '') : v['value'] for k, v in scores_test.items()}

    for k, v in save_scores.items():
        print(f'{k}: {v:.4f}')
    
    json.dump(save_scores,
              open(os.path.join(config_args["training"]["output_folder"],
                                'test_metrics.json'), 'w'))

if __name__ == "__main__":
    main()
