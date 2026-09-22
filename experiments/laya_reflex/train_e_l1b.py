"""E-L1b: fine-tune Laya's English checkpoint on Kannaka's evidence-gate decision, single GPU.

Adapted from Laya's `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` (RLCD:
zero-mean noisy-logit groups scored by proper scoring rules + soft cross-entropy), reduced to
one process, one GPU, our rows. Input rows come from `build_e_l1b_dataset.py`; output is a
Laya agent directory that `laya.Agent(dir)` loads, which `e_l1_evidence_gate.py --agent-dir`
then scores on the held-out 450 decisions under the pre-registered E-L1 rule.

    python train_e_l1b.py --train e_l1b_train.jsonl --out ./laya_e_l1b --epochs 4
"""
import argparse
import json
import os
import random
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_model, build_sequence, proper_reward, render_options, QTYPES


def build_item(tok, cfg, state, q, gold_q):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    elif t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    else:
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    k = len(render_options({"t": t, "crit": crit}))
    seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
                                  cfg["max_len"], cfg["head_max_len"])
    if len(markers) != k:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "label": target.index(max(target))}


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


def fit_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="convaiinnovations/laya")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    model_dir = snapshot_download(args.model_id)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    items = []
    for r in rows:
        for qid, q in r["questions"].items():
            it = build_item(tok, cfg, r["state"], q, r["gold"][qid])
            if it:
                items.append(it)
    print(f"{len(items)} training sequences from {len(rows)} rows", flush=True)

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    GROUP, LR_ENC, LR_HEAD, SIG0, SIG1 = 4, 2.5e-5, 1.0e-4, 0.4, 0.1
    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENC}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    total_updates = max(1, (len(items) // (args.micro_batch * args.grad_accum)) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    t0 = time.time()
    log = []
    for epoch in range(args.epochs):
        random.seed(args.seed + epoch)
        random.shuffle(items)
        sigma = SIG0 + (SIG1 - SIG0) * (epoch / max(1, args.epochs - 1))
        ep_loss, nb, step = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for b in range(0, len(items), args.micro_batch):
            chunk = items[b : b + args.micro_batch]
            batch = collate(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                                    batch["qtype"].to(device))
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((GROUP,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            step += 1
            if step % args.grad_accum == 0 or b + args.micro_batch >= len(items):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            ep_loss += loss.item() * args.grad_accum
            nb += 1
            if nb % 50 == 0:
                print(f"  epoch {epoch + 1}/{args.epochs} step {nb} loss {loss.item() * args.grad_accum:.4f} "
                      f"reward {r.mean().item():.3f} {time.time() - t0:.0f}s", flush=True)
        avg = ep_loss / max(1, nb)
        log.append({"epoch": epoch + 1, "avg_loss": avg, "elapsed_s": time.time() - t0})
        print(f"=== epoch {epoch + 1}/{args.epochs} avg loss {avg:.4f} at {time.time() - t0:.0f}s ===", flush=True)

    # Post-training temperature for the noul head, fitted on a training subsample (as upstream).
    model.eval()
    calib = items[::10][:400]
    preds = []
    with torch.no_grad():
        for c in range(0, len(calib), 16):
            cb = collate(calib[c : c + 16], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                l_sub, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device),
                                 cb["marker_pos"].to(device), cb["marker_mask"].to(device), cb["qtype"].to(device))
            l_np = l_sub.float().cpu().numpy()
            for i, it in enumerate(calib[c : c + 16]):
                preds.append((it["qtype"], l_np[i, : len(it["markers"])], it["target"]))
    temps = list(cfg.get("temperature", [1.2, 1.2, 1.2]))
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in preds if q_type == qt]
        if sel:
            temps[qt] = fit_temp(sel)
    print("fitted temperatures (choice, score, noul):", [round(t, 3) for t in temps], flush=True)

    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(args.out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(args.out, "encoder"))
    tok.save_pretrained(os.path.join(args.out, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-e-l1b-evidence-gate"
    cfg["temperature"] = temps
    json.dump(cfg, open(os.path.join(args.out, "rl_agent_config.json"), "w"), indent=2)
    json.dump({"train_rows": len(rows), "items": len(items), "epochs": args.epochs, "log": log,
               "temperatures": temps, "base_model": args.model_id, "seed": args.seed},
              open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
    print(f"saved to {args.out} after {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
