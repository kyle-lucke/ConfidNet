import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, auc, confusion_matrix


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist

# also known as recall or true positive rate (TPR)
def sensitivity(tp, fn):
    return tp / (tp + fn)

# Also known as selectivity, or true negative rate (TNR)
def specificity(tn, fp):
    return tn / (tn + fp)

def threshold(preds, tau):
    if isinstance(preds, np.ndarray):
        return np.where(preds>tau, 1.0, 0.0)

    elif torch.is_tensor(preds):
        return torch.where(preds>tau, 1.0, 0.0)

    else:
        raise TypeError(f"ERROR: preds is expected to be of type (torch.tensor, numpy.ndarray) but is type {type(preds)}")

# beta > 1 gives more weight to specificity, while beta < 1 favors
# sensitivity. For example, beta = 2 makes specificity twice as important as
# sensitivity, while beta = 0.5 does the opposite.

# eqn. source: F-score wikipedia page (https://en.wikipedia.org/wiki/F-score)
def f_score_sens_spec(sensitivity, specificity, beta=1.0):

    # return (1 + beta**2) * ( (precision * recall) / ( (beta**2 * precision) + recall ) )

    return (1 + beta**2) * ( (sensitivity * specificity) / ( (beta**2 * sensitivity) + specificity ) )

class Metrics:

    '''

    self.metrics: list of strings corresponding to the metrics to be calculated
    
    self.len_dataset: number of total elements in the dataset
    
    self.n_classes: number of classes in the dataset
    
    self.accurate: indicator for correct base model predicitons. If True,
    predicted label of base model matches the ground truth label.

    self.errors: indicator for incorrect base model predicitons. If True,
    predicted label of base model does not match the ground truth label.

    self.proba_pred: predicted confidence values by base model. 

    self.accuracy: number of correctly classified data samples

    self.confusion_matrix: in the case of segmentation dataset,
    confusion matrix of pixel predictions vs GT labels

    self.tps, self.fps, self.tns, self.fns: accumulated number number
    of true positivies, false positives, true negatives, and false
    negatives, respectively for miscalssication prediction

    '''
    
    def __init__(self, metrics, len_dataset, n_classes, threshold):
        
        self.metrics = metrics
        self.len_dataset = len_dataset
        self.n_classes = n_classes
        self.accurate, self.errors, self.proba_pred = [], [], []
        self.accuracy = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        
        self.tps = 0
        self.fps = 0
        self.tns = 0
        self.fns = 0 

        self.threshold = threshold
        
    def update(self, pred, target, confidence):

        target = target.view_as(pred).detach().to('cpu').numpy()
        pred = pred.detach().to('cpu').numpy()
        confidence = confidence.detach().to("cpu").numpy()

        accurate_b = pred == target
        
        self.accurate.extend(accurate_b)
        self.accuracy += np.sum(accurate_b)
        self.errors.extend( pred != target )
        self.proba_pred.extend(confidence)

        msclf_labels = pred == target

        # assume confidence is true probability value:
        predicted_labels = threshold(confidence, tau=self.threshold)
        
        tn, fp, fn, tp = confusion_matrix(msclf_labels, predicted_labels).ravel()
        
        self.tps += tp
        self.fps += fp
        self.tns += tn
        self.fns += fn
        
        if "mean_iou" in self.metrics:
            pred = pred.cpu().numpy().flatten()
            target = target.cpu().numpy().flatten()
            mask = (target >= 0) & (target < self.n_classes)
            hist = np.bincount(
                self.n_classes * target[mask].astype(int) + pred[mask],
                minlength=self.n_classes ** 2,
            ).reshape(self.n_classes, self.n_classes)
            self.confusion_matrix += hist

    def get_scores(self, split="train"):
        self.accurate = np.reshape(self.accurate, newshape=(len(self.accurate), -1)).flatten()
        self.errors = np.reshape(self.errors, newshape=(len(self.errors), -1)).flatten()
        self.proba_pred = np.reshape(self.proba_pred, newshape=(len(self.proba_pred), -1)).flatten()

        scores = {}
        if "accuracy" in self.metrics:
            accuracy = self.accuracy / self.len_dataset
            scores[f"{split}/accuracy"] = {"value": accuracy, "string": f"{accuracy:05.2%}"}
        if "auc" in self.metrics:
            if len(np.unique(self.accurate)) == 1:
                auc_score = 1
            else:
                auc_score = roc_auc_score(self.accurate, self.proba_pred)
            scores[f"{split}/auc"] = {"value": auc_score, "string": f"{auc_score:05.2%}"}
        if "ap_success" in self.metrics:
            ap_success = average_precision_score(self.accurate, self.proba_pred)
            scores[f"{split}/ap_success"] = {"value": ap_success, "string": f"{ap_success:05.2%}"}
        if "accuracy_success" in self.metrics:
            accuracy_success = np.round(self.proba_pred[self.accurate == 1]).mean()
            scores[f"{split}/accuracy_success"] = {
                "value": accuracy_success,
                "string": f"{accuracy_success:05.2%}",
            }
        if "ap_errors" in self.metrics:
            ap_errors = average_precision_score(self.errors, -self.proba_pred)
            scores[f"{split}/ap_errors"] = {"value": ap_errors, "string": f"{ap_errors:05.2%}"}
        if "accuracy_errors" in self.metrics:
            accuracy_errors = 1.0 - np.round(self.proba_pred[self.errors == 1]).mean()
            scores[f"{split}/accuracy_errors"] = {
                "value": accuracy_errors,
                "string": f"{accuracy_errors:05.2%}",
            }
        if "fpr_at_95tpr" in self.metrics:
            for i,delta in enumerate(np.arange(
                self.proba_pred.min(),
                self.proba_pred.max(),
                (self.proba_pred.max() - self.proba_pred.min()) / 10000,
            )):
                tpr = len(self.proba_pred[(self.accurate == 1) & (self.proba_pred >= delta)]) / len(
                    self.proba_pred[(self.accurate == 1)]
                )
                if i%100 == 0:
                    print(f"Threshold:\t {delta:.6f}")
                    print(f"TPR: \t\t {tpr:.4%}")
                    print("------")
                if 0.9505 >= tpr >= 0.9495:
                    print(f"Nearest threshold 95% TPR value: {tpr:.6f}")
                    print(f"Threshold 95% TPR value: {delta:.6f}")
                    fpr = len(
                        self.proba_pred[(self.errors == 1) & (self.proba_pred >= delta)]
                    ) / len(self.proba_pred[(self.errors == 1)])
                    scores[f"{split}/fpr_at_95tpr"] = {"value": fpr, "string": f"{fpr:05.2%}"}
                    break
        if "mean_iou" in self.metrics:
            iou = np.diag(self.confusion_matrix) / (
                self.confusion_matrix.sum(axis=1)
                + self.confusion_matrix.sum(axis=0)
                - np.diag(self.confusion_matrix)
            )
            mean_iou = np.nanmean(iou)
            scores[f"{split}/mean_iou"] = {"value": mean_iou, "string": f"{mean_iou:05.2%}"}
        if "aurc" in self.metrics:
            risks, coverages = [], []
            for delta in sorted(set(self.proba_pred))[:-1]:
                coverages.append((self.proba_pred > delta).mean())
                selected_accurate = self.accurate[self.proba_pred > delta]
                risks.append(1. - selected_accurate.mean())
            aurc = auc(coverages, risks)
            eaurc = aurc - ((1. - accuracy) + accuracy*np.log(accuracy))
            scores[f"{split}/aurc"] = {"value": aurc, "string": f"{aurc*1000:01.2f}"}
            scores[f"{split}/e-aurc"] = {"value": eaurc, "string": f"{eaurc*1000:01.2f}"}

        if 'spec_sens' in self.metrics:

            specificity_value = specificity(self.tns, self.fps)
            sensitivity_value = sensitivity(self.tps, self.fns)

            for beta in [1.0, 2.0]:
            
                f_beta_spec_sens = f_score_sens_spec(sensitivity_value,
                                                     specificity_value, beta=beta)
            
                scores[f'{split}/f_beta_spec_sens@{beta}'] = {'value': f_beta_spec_sens ,
                                                              'string': f'{f_beta_spec_sens:.4f}' }         
            scores[f'{split}/specificity'] = {'value': specificity_value,
                                              'string': f'{specificity_value:.4f}' }
            
            scores[f'{split}/sensitivity'] = {'value': sensitivity_value,
                                              'string': f'{sensitivity_value:.4f}' }

            # save threshold used for testing
            scores[f'{split}/threshold'] = {'value': self.threshold,
                                            'string': f'{self.threshold:.6f}'}

        return scores
