"""(1) bidirectional fp16 SDPA on CUDA vs fp32 eager reference; (2) linear probe of frozen causal vs bidirectional features."""
import sys

import torch
from transformers import AutoModel

from deciformer.attention import apply_attention_mode
from deciformer.backbone import Backbone, load_tokenizer
from deciformer.config import PromptConfig
from deciformer.data import Collator, DecisionDataset
from deciformer.prompting import PromptBuilder, prepare_tokenizer
from deciformer.schema import load_decisions

name = sys.argv[1]
device = torch.device("cuda")
tokenizer = load_tokenizer(name)
prepare_tokenizer(tokenizer, PromptConfig(), "decoder")
builder = PromptBuilder(tokenizer, PromptConfig(), "decoder", max_length=256, overflow="truncate_state_end",
                        max_options=26, need_code_tokens=False)
train = DecisionDataset(load_decisions("data/mnli/train.jsonl")[:2000], builder)
dev = DecisionDataset(load_decisions("data/mnli/dev_small.jsonl"), builder)
collate = Collator(builder, tokenizer.pad_token_id)


def load(mode, dtype, implementation):
    model = AutoModel.from_pretrained(name, dtype=dtype, attn_implementation=implementation).to(device).eval()
    apply_attention_mode(model, mode)
    return Backbone(model, "decoder", mode)


@torch.no_grad()
def features(backbone, dataset):
    rows, labels = [], []
    for start in range(0, len(dataset), 16):
        batch = collate(dataset, list(range(start, min(start + 16, len(dataset))))).to(device)
        hidden = backbone(batch).float()
        mask = batch.attention_mask.unsqueeze(-1).float()
        mean = (hidden * mask).sum(1) / mask.sum(1)
        last = hidden[torch.arange(len(batch), device=device), batch.decision_index]
        rows.append(torch.cat([mean, last], -1).cpu())
        labels.append(batch.label_index.cpu())
    return torch.cat(rows), torch.cat(labels)


def probe(train_x, train_y, dev_x, dev_y, part):
    width = train_x.shape[1] // 2
    sl = slice(0, width) if part == "mean" else slice(width, None)
    x, d = train_x[:, sl], dev_x[:, sl]
    mu, sd = x.mean(0), x.std(0) + 1e-6
    x, d = (x - mu) / sd, (d - mu) / sd
    w = torch.zeros(x.shape[1], 3, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ w + b, train_y) + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(((d @ w + b).argmax(-1) == dev_y).float().mean())


batch = collate(dev, [0, 1, 2, 3]).to(device)  # padded batch of different lengths
for mode in ("causal", "bidirectional"):
    fast, reference = load(mode, torch.float16, "sdpa"), load(mode, torch.float32, "eager")
    with torch.no_grad():
        a, b = fast(batch).float(), reference(batch)
    real = batch.attention_mask.bool()
    rel = float(((a - b)[real].norm(dim=-1) / b[real].norm(dim=-1)).max())
    del reference
    torch.cuda.empty_cache()
    tx, ty = features(fast, train)
    dx, dy = features(fast, dev)
    print(f"{name.split('/')[-1]:24} {mode:13} fp16-sdpa vs fp32-eager max relative error {rel:.4f} | "
          f"probe acc mean-pool {probe(tx, ty, dx, dy, 'mean'):.3f} last-token {probe(tx, ty, dx, dy, 'last'):.3f}",
          flush=True)
    del fast
    torch.cuda.empty_cache()
