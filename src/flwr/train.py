import torch
import torch.nn.functional as F


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_bce(logits, targets, masks, pos_weight=None):
    """Average supervised findings per image, returning one loss per image."""

    losses = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none",
    )

    return (losses * masks).sum(1) / masks.sum(1).clamp_min(1)


def train_local(model, loader, device, pos_weight, learning_rate=1e-3, epochs=1):
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    
    model.to(device).train()
    weights = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    total_loss = torch.zeros((), device=device)
    total_examples = 0

    for _ in range(epochs):
        for images, targets, masks in loader:
            images, targets, masks = (x.to(device) for x in (images, targets, masks))
            optimizer.zero_grad(set_to_none=True)
            losses = masked_bce(model(images), targets, masks, weights)
            losses.mean().backward()
            # Stability clipping, not a differential privacy mechanism.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += losses.detach().sum()
            total_examples += len(images)

    if not total_examples:
        raise ValueError("Training loader is empty")
    
    return total_loss.item() / total_examples
