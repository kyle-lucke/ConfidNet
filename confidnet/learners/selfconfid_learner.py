import os
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from confidnet.learners.learner import AbstractLeaner
from confidnet.utils import misc
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics, sensitivity, specificity, threshold, f_score_sens_spec

LOGGER = get_logger(__name__, level="DEBUG")


class SelfConfidLearner(AbstractLeaner):
    def __init__(self, config_args, train_loader, val_loader, test_loader, start_epoch, device):
        super().__init__(config_args, train_loader, val_loader, test_loader, start_epoch, device)
        self.freeze_layers()
        self.disable_bn(verbose=True)
        if self.config_args["model"].get("uncertainty", None):
            self.disable_dropout(verbose=True)

        self.best_aupr_error = float('-inf')
            
    def train(self, epoch):
        self.model.train()
        self.disable_bn()
        if self.config_args["model"].get("uncertainty", None):
            self.disable_dropout()
        metrics = Metrics(
            self.metrics, self.prod_train_len, self.num_classes, threshold=0.5)
        loss, confid_loss = 0, 0
        len_steps, len_data = 0, 0

        # Training loop
        loop = tqdm(self.train_loader)
        for batch_id, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            # Potential temperature scaling
            if self.temperature:
                output = list(output)
                output[0] = output[0] / self.temperature
                output = tuple(output)

            if self.task == "classification":
                current_loss = self.criterion(output, target)
            elif self.task == "segmentation":
                current_loss = self.criterion(output, target.squeeze(dim=1))
            current_loss.backward()
            loss += current_loss
            self.optimizer.step()
            if self.task == "classification":
                len_steps += len(data)
                len_data = len_steps
            elif self.task == "segmentation":
                len_steps += len(data) * np.prod(data.shape[-2:])
                len_data += len(data)

            # Update metrics
            pred = output[0].argmax(dim=1, keepdim=True)
            confidence = torch.sigmoid(output[1])
            metrics.update(pred, target, confidence)

            # Update the average loss
            loop.set_description(f"Epoch {epoch}/{self.nb_epochs}")
            loop.set_postfix(
                OrderedDict(
                    {
                        "loss_confid": f"{(loss / len_data):05.3e}",
                        "acc": f"{(metrics.accuracy / len_steps):05.2%}",
                    }
                )
            )
            loop.update()

        # Eval on epoch end
        scores = metrics.get_scores(split="train")
        
        logs_dict = OrderedDict(
            {
                "epoch": {"value": epoch, "string": f"{epoch:03}"},
                "lr": {
                    "value": self.optimizer.param_groups[0]["lr"],
                    "string": f"{self.optimizer.param_groups[0]['lr']:05.1e}",
                },
                "train/loss_confid": {
                    "value": loss / len_data,
                    "string": f"{(loss / len_data):05.4e}",
                },
            }
        )
        for s in scores:
            logs_dict[s] = scores[s]

        # Val scores
        val_losses, scores_val = self.evaluate(self.val_loader, self.prod_val_len, split="val")
        logs_dict["val/loss_confid"] = {
            "value": val_losses["loss_confid"].item() / self.nsamples_val,
            "string": f"{(val_losses['loss_confid'].item() / self.nsamples_val):05.4e}",
        }
        for sv in scores_val:
            logs_dict[sv] = scores_val[sv]
        
        # Print metrics
        misc.print_dict(logs_dict)

        if logs_dict["val/ap_errors"]["value"] > self.best_aupr_error:

            LOGGER.info(f'Validation AP Error improved from {self.best_aupr_error:.4e} to {logs_dict["val/ap_errors"]["value"]:.4e}, saving model.')
            
            # Save the model checkpoint
            self.save_checkpoint(epoch)

            self.best_aupr_error = logs_dict["val/ap_errors"]["value"]
            
        # CSV logging
        misc.csv_writter(path=self.output_folder / "logs.csv", dic=OrderedDict(logs_dict))

        # Scheduler step
        if self.scheduler:
            self.scheduler.step()

    def evaluate(self, dloader, len_dataset, split="test", verbose=False, **args):

        if split == 'test':
            threshold = self.determine_threshold()

        else:
            threshold = 0.5
            
        self.model.eval()
        metrics = Metrics(self.metrics, len_dataset, self.num_classes, threshold)
        loss = 0

        # Evaluation loop
        loop = tqdm(dloader, disable=not verbose)
        for batch_id, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                output = self.model(data)
                if self.task == "classification":
                    loss += self.criterion(output, target)
                elif self.task == "segmentation":
                    loss += self.criterion(output, target.squeeze(dim=1))
                # Update metrics
                pred = output[0].argmax(dim=1, keepdim=True)
                confidence = torch.sigmoid(output[1])
                metrics.update(pred, target, confidence)

        scores = metrics.get_scores(split=split)

        losses = {"loss_confid": loss}
        return losses, scores

    @torch.no_grad()
    def determine_threshold(self, max_threshold_step=.01):

        LOGGER.info('Determining best threshold for specificity and sensitivity using validation dataset.')
        
        self.model.eval()

        # NOTE: the actual threhsold step may be slightly lower than
        # max_threshold_step due to roundoff
        misclf_labels = []
        meta_preds = []
        for inps_b, gt_labels_b in self.val_loader:
            
            inps_b = inps_b.to(self.device)
            gt_labels_b = gt_labels_b.to(self.device)
            
            base_preds_b, meta_preds_b = self.model(inps_b)
            
            base_pred_labels_b = torch.argmax(base_preds_b, dim=-1)
                        
            misclf_labels_b = (base_pred_labels_b == gt_labels_b).to(int)

            meta_preds.append(torch.sigmoid(meta_preds_b))
            misclf_labels.append(misclf_labels_b)
    
        meta_preds = torch.concat(meta_preds).detach().cpu().numpy()
        misclf_labels = torch.concat(misclf_labels).detach().cpu().numpy()
        
        # determine how many elements we need for a pre-determined spacing
        # between thresholds. Taken from:
        # https://stackoverflow.com/a/70230433
        num = round((meta_preds.max() - meta_preds.min()) / max_threshold_step) + 1 
        thresholds = np.linspace(meta_preds.min(), meta_preds.max(), num, endpoint=True)

        LOGGER.info(f'Determining best threshold for over {num} thresholds.')
        
        # compute performance over thresholds
        threshold_to_metric = {}
        for tau in thresholds:

            predicted_labels = threshold(meta_preds, tau)

            tn, fp, fn, tp = confusion_matrix(misclf_labels, predicted_labels).ravel()
        
            specificity_value = specificity(tn, fp)
            sensitivity_value = sensitivity(tp, fn)

            f_beta_spec_sens = f_score_sens_spec(sensitivity_value,
                                                 specificity_value, beta=1.0)

            print(f'tau: {tau:.6f}, spec: {specificity_value:.4f}, sens: {sensitivity_value:.4f}, f_beta: {f_beta_spec_sens:.4f}, balance: {abs(specificity_value-sensitivity_value):.4f}')
        
            threshold_to_metric[tau] = f_beta_spec_sens

        # determine best threshold:
        best_item = max(threshold_to_metric.items(), key=lambda x: x[1])

        best_tau, best_metric = best_item
        print(f'best | tau: {best_tau:.6f}, metric: {best_metric:.4f}', end='\n\n')

        return best_item[0]
        
    
    def load_checkpoint(self, state_dict, uncertainty_state_dict=None, strict=True):
        if not uncertainty_state_dict:
            self.model.load_state_dict(state_dict, strict=strict)
        else:
            self.model.pred_network.load_state_dict(state_dict, strict=strict)

            # 1. filter out unnecessary keys
            if self.task == "classification":
                state_dict = {
                    k: v
                    for k, v in uncertainty_state_dict.items()
                    if k not in ["fc2.weight", "fc2.bias"]
                }
            if self.task == "segmentation":
                state_dict = {
                    k: v
                    for k, v in uncertainty_state_dict.items()
                    if k
                    not in [
                        "up1.conv2.cbr_unit.0.weight",
                        "up1.conv2.cbr_unit.0.bias",
                        "up1.conv2.cbr_unit.1.weight",
                        "up1.conv2.cbr_unit.1.bias",
                        "up1.conv2.cbr_unit.1.running_mean",
                        "up1.conv2.cbr_unit.1.running_var",
                    ]
                }
            # 2. overwrite entries in the existing state dict
            self.model.uncertainty_network.state_dict().update(state_dict)
            # 3. load the new state dict
            self.model.uncertainty_network.load_state_dict(state_dict, strict=False)

    def freeze_layers(self):
        # Eventual fine-tuning for self-confid
        LOGGER.info("Freezing every layer except uncertainty")
        for param in self.model.named_parameters():
            if "uncertainty" in param[0]:
                print(param[0], "kept to training")
                continue
            param[1].requires_grad = False

    def disable_bn(self, verbose=False):
        # Freeze also BN running average parameters
        if verbose:
            LOGGER.info("Keeping original BN parameters")
        for layer in self.model.named_modules():
            if "bn" in layer[0] or "cbr_unit.1" in layer[0]:
                if verbose:
                    print(layer[0], "original BN setting")
                layer[1].momentum = 0
                layer[1].eval()

    def disable_dropout(self, verbose=False):
        # Freeze also BN running average parameters
        if verbose:
            LOGGER.info("Disable dropout layers to reduce stochasticity")
        for layer in self.model.named_modules():
            if "dropout" in layer[0]:
                if verbose:
                    print(layer[0], "set to eval mode")
                layer[1].eval()
