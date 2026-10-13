import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.flwr.train import masked_bce


@torch.inference_mode()
def evaluate(model, loader, device):

    model.to(device).eval()
    probabilities, targets_all, masks_all = [], [], []
    total_loss = torch.zeros((), device=device)
    total_examples = 0

    for images, targets, masks in loader:
        images, targets, masks = (x.to(device) for x in (images, targets, masks))
        logits = model(images)
        total_loss += masked_bce(logits, targets, masks).sum()
        total_examples += len(images)
        probabilities.append(logits.sigmoid().cpu().numpy())
        targets_all.append(targets.cpu().numpy())
        masks_all.append(masks.cpu().numpy())

    if not total_examples:
        raise ValueError("Evaluation loader is empty")
    
    probabilities = np.concatenate(probabilities)
    targets = np.concatenate(targets_all)
    masks = np.concatenate(masks_all).astype(bool)
    metrics = {"loss": total_loss.item() / total_examples}
    aucs, aps = [], []

    for c in range(targets.shape[1]):
        y, p = targets[masks[:, c], c], probabilities[masks[:, c], c]
        metrics[f"label_{c}_count"] = len(y)
        metrics[f"label_{c}_positives"] = int(y.sum())

        if np.unique(y).size != 2:
            continue

        aucs.append(float(roc_auc_score(y, p)))
        aps.append(float(average_precision_score(y, p)))
        metrics[f"label_{c}_auroc"] = aucs[-1]
        metrics[f"label_{c}_ap"] = aps[-1]

    metrics["evaluated_labels"] = len(aucs)

    if aucs:
        metrics.update(macro_auroc=float(np.mean(aucs)), macro_ap=float(np.mean(aps)))
        
    return metrics
