# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
import torch.nn.functional as F


def ValidateCategoricalShape(logits, classes):
    if classes <= 0:
        raise ValueError("classes must be positive")
    if logits.shape[-1] % classes != 0:
        raise ValueError(
            f"Last dimension ({logits.shape[-1]}) must be divisible by "
            f"classes ({classes})"
        )


def Softmax(logits, classes):
    ValidateCategoricalShape(logits, classes)
    return F.softmax(logits.reshape(-1, classes), dim=-1).reshape(logits.shape)


def Sigmoid(logits):
    return torch.sigmoid(logits)


def Standardize(logits, classes=None, eps=1e-6):
    classes = logits.shape[-1] if classes is None else classes
    if classes < 2:
        raise ValueError("classes must be at least 2 for standardization")
    ValidateCategoricalShape(logits, classes)

    grouped = logits.reshape(-1, classes)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    return (centered / scale).reshape(logits.shape)


def Gaussian(logits, classes=None, noise=None):
    classes = logits.shape[-1] if classes is None else classes
    latents = Standardize(logits, classes)
    noise_scale = torch.rand_like(latents[..., :1])
    noise = noise_scale * torch.randn_like(latents)
    return Standardize(latents + noise, classes)


def UnitSphere(logits, classes, eps=1e-6):
    ValidateCategoricalShape(logits, classes)
    grouped = logits.reshape(-1, classes)
    lengths = torch.linalg.vector_norm(grouped, dim=-1, keepdim=True)
    normalized = grouped / lengths.clamp_min(eps)

    fallback = torch.zeros_like(grouped)
    fallback[..., 0] = 1.0
    return torch.where(lengths > eps, normalized, fallback).reshape(logits.shape)


def Argmax(probs, classes):
    return F.one_hot(
        probs.reshape(probs.shape[0], -1, classes).argmax(dim=-1), classes
    ).reshape(probs.shape)


def OneHot(logits, classes, straight_through=False):
    probs = Softmax(logits, classes)
    one_hot = (
        F.one_hot(probs.reshape(probs.shape[0], -1, classes).argmax(dim=-1), classes)
        .reshape(probs.shape)
        .to(dtype=probs.dtype)
    )
    if straight_through:
        return (one_hot - probs).detach() + probs
    else:
        return one_hot


def Categorical(logits, classes, straight_through=False):
    probs = Softmax(logits, classes)
    indices = torch.multinomial(probs.reshape(-1, classes), num_samples=1)
    samples = F.one_hot(indices.squeeze(-1), classes).reshape(probs.shape)
    samples = samples.to(dtype=probs.dtype)
    if straight_through:
        return (samples - probs).detach() + probs
    return samples


def GumbelSoftmax(logits, classes, temperature=1.0, hard=False):
    ValidateCategoricalShape(logits, classes)
    samples = F.gumbel_softmax(
        logits.reshape(-1, classes),
        tau=temperature,
        hard=hard,
        dim=-1,
    )
    return samples.reshape(logits.shape)


def Gumbel(logits, classes, noise=None, temperature=1.0, hard=False, eps=1e-10):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    ValidateCategoricalShape(logits, classes)
    U = torch.rand_like(logits) if noise is None else noise
    N = -torch.log(-torch.log(U + eps) + eps)
    y = logits + N
    y = y.reshape(-1, classes)
    y = F.softmax(y / temperature, dim=-1)

    if not hard:
        return y.reshape(logits.shape)

    size = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, size[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*size)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.reshape(logits.shape)
