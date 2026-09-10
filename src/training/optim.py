import torch

from src.models.gpt import GPTModel
from src.training.seed import maybe_compile


def build_gpt(task, args, device):
    n_embd = getattr(args, "embedding", getattr(args, "n_embd", 128))
    n_head = getattr(args, "heads", getattr(args, "n_head", 4))
    n_layer = getattr(args, "layers", getattr(args, "n_layer", 2))
    dropout = getattr(args, "dropout", 0.0)
    tok = task.tokenizer
    model = GPTModel(
        vocab_size=tok.vocab_size,
        block_size=task.block_size,
        pad_id=tok.pad_id,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)
    return maybe_compile(model, device, enabled=getattr(args, "compile", None))


def make_adamw(params, lr, weight_decay=0.0, device=None):
    fused = False
    if device is not None:
        fused = torch.device(device).type == "cuda"
    else:
        fused = any(getattr(p, "is_cuda", False) for p in params)
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=fused)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_optimizer(model, args, device):
    lr = getattr(args, "lr", 3e-4)
    wd = getattr(args, "weight_decay", 0.0)
    return make_adamw(model.parameters(), lr, weight_decay=wd, device=device)


def make_loader(dataset, args, device, extra_seed=0):
    generator = torch.Generator().manual_seed(getattr(args, "batch_seed", 12345) + extra_seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=getattr(args, "workers", 0),
        generator=generator,
        pin_memory=device.type == "cuda",
    )
