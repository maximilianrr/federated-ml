"""Native PyTorch DP with integer-order sampled-Gaussian RDP accounting.

Integer-order moments: https://arxiv.org/abs/1908.10530, section 3.3.
"""

import json
import math
import secrets

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from src.flwr.train import get_device, masked_bce

ORDERS = (*range(2, 65), 128, 256)
LOG_BINOMIALS = {
    order: np.array([math.log(math.comb(order, i)) for i in range(order + 1)])
    for order in ORDERS
}


def rdp_per_step(sample_rate, noise_multiplier):
    """RDP for one Poisson-sampled, unit-sensitivity Gaussian mechanism."""
    if not 0 < sample_rate <= 1 or not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("Require 0 < sample_rate <= 1 and a finite positive noise multiplier")
    if sample_rate == 1:
        return np.array(ORDERS) / (2 * noise_multiplier ** 2)
    values = []
    for order in ORDERS:
        i = np.arange(order + 1)
        terms = (
            LOG_BINOMIALS[order] + i * math.log(sample_rate)
            + (order - i) * math.log1p(-sample_rate)
            + (i * i - i) / (2 * noise_multiplier ** 2)
        )
        maximum = terms.max()
        log_moment = maximum + np.log(np.exp(terms - maximum).sum())
        values.append(max(0.0, log_moment / (order - 1)))
    return np.array(values)


def epsilon_for_steps(sample_rate, noise_multiplier, steps, delta):
    if steps < 0 or not 0 < delta < 1:
        raise ValueError("Require nonnegative steps and 0 < delta < 1")
    if steps == 0:
        return 0.0
    # Standard RDP -> (epsilon, delta) conversion; finite orders are conservative.
    return float(np.min(
        steps * rdp_per_step(sample_rate, noise_multiplier)
        + math.log(1 / delta) / (np.array(ORDERS) - 1)
    ))


def calibrate_noise(sample_rate, steps, epsilon, delta):
    if not math.isfinite(epsilon) or epsilon <= math.log(1 / delta) / (max(ORDERS) - 1):
        raise ValueError("Requested epsilon is too small for the configured accountant orders")
    
    low, high = 0.0, 1.0
    while epsilon_for_steps(sample_rate, high, steps, delta) > epsilon:
        high *= 2

    for _ in range(40):
        middle = (low + high) / 2

        if epsilon_for_steps(sample_rate, middle, steps, delta) > epsilon:
            low = middle
        else:
            high = middle

    return high


def clipped_gradients(model, images, targets, masks, clip):
    """Clip each image's complete gradient vector, then sum across images."""
    parameters = {name: parameter.detach() for name, parameter in model.named_parameters()}
    buffers = dict(model.named_buffers())

    def loss(params, image, target, mask):
        logits = functional_call(model, (params, buffers), (image.unsqueeze(0),))
        return masked_bce(logits, target.unsqueeze(0), mask.unsqueeze(0)).sum()

    gradients = vmap(
        grad(loss), in_dims=(None, 0, 0, 0), randomness="different",
    )(parameters, images, targets, masks)
    norms = torch.stack([
        gradient.flatten(1).square().sum(1) for gradient in gradients.values()
    ]).sum(0).sqrt()
    factors = (clip / (norms + 1e-6)).clamp(max=1)

    return {
        name: torch.einsum("i,i...->...", factors, gradient)
        for name, gradient in gradients.items()
    }


def train_private(model, loader, config, previous=None):
    server_round = int(config["server-round"])
    epochs, rounds = int(config["local-epochs"]), int(config["num-server-rounds"])
    epsilon, delta = float(config["target-epsilon"]), float(config["dp-delta"])
    clip, physical_batch = float(config["dp-max-grad-norm"]), int(config["dp-physical-batch-size"])
    size = len(loader.dataset)

    if epochs < 1 or rounds < 1 or not 1 <= server_round <= rounds:
        raise ValueError("Invalid DP round/epoch configuration")
    
    if size < 1 or not 0 < delta < 1 / size or not math.isfinite(clip) or clip <= 0 or physical_batch < 1:
        raise ValueError("Use positive DP parameters and delta < 1 / client training size")

    batch_size = min(loader.batch_size, size)
    sample_rate, steps_per_round = batch_size / size, epochs * math.ceil(size / batch_size)
    parameters = json.dumps([epsilon, delta, clip, epochs, rounds, batch_size, size])

    if previous is not None:
        if previous["round"] != server_round - 1 or previous["parameters"] != parameters:
            raise ValueError("DP rounds must be consecutive with unchanged privacy parameters")
        
        if previous["steps"] != (server_round - 1) * steps_per_round:
            raise ValueError("The saved privacy step count does not match previous rounds")
        
    elif server_round != 1:
        raise ValueError("Missing privacy accounting state from previous rounds")

    sigma = float(previous["noise-multiplier"]) if previous else calibrate_noise(
        sample_rate, rounds * steps_per_round, epsilon, delta,
    )
    device = get_device()
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning-rate"]), weight_decay=1e-4)
    # Independent fresh RNG streams; generate noise on CPU for MPS compatibility.
    sampler = torch.Generator().manual_seed(secrets.randbits(63))
    noise = torch.Generator().manual_seed(secrets.randbits(63))
    steps = int(previous["steps"]) if previous else 0

    for _ in range(steps_per_round):
        indices = torch.nonzero(torch.rand(size, generator=sampler) < sample_rate).flatten().tolist()
        summed = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}

        for start in range(0, len(indices), physical_batch):
            batch = loader.collate_fn([loader.dataset[i] for i in indices[start:start + physical_batch]])
            images, targets, masks = (tensor.to(device) for tensor in batch)
            gradients = clipped_gradients(model, images, targets, masks, clip)

            for name in summed:
                summed[name].add_(gradients[name])

        optimizer.zero_grad(set_to_none=True)

        for name, parameter in model.named_parameters():
            perturbation = torch.randn(parameter.shape, generator=noise, dtype=parameter.dtype).to(device)
            parameter.grad = (summed[name] + perturbation * (sigma * clip)) / batch_size

        # Empty Poisson draws still produce noisy, accounted optimizer steps.
        optimizer.step()
        steps += 1

    spent = epsilon_for_steps(sample_rate, sigma, steps, delta)
    if spent > epsilon + 1e-6:
        raise RuntimeError("The accumulated privacy budget exceeded its configured target")
    
    return {
        "epsilon": spent, "delta": delta, "noise_multiplier": sigma,
        "sample_rate": sample_rate, "steps": steps,
    }, {
        "round": server_round, "parameters": parameters,
        "noise-multiplier": sigma, "steps": steps,
    }
