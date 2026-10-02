"""Train a small downscaled event-camera SNN and save it as its model's
checkpoint, with BatchNorm folded into the weights:
  --model gesture16  DVS Gesture at 16x16, 10 classes (models/gesture16.py)
  --model nmnist17   N-MNIST at 17x17 (models/nmnist17.py)

    python tools/train_event_snn.py --model gesture16          # dataset from run.py's DATASET_DIR
    python tools/train_event_snn.py --model nmnist17 --epochs 50 --data /path/to/datasets
    python tools/train_event_snn.py --model gesture16 --resume # continue from the last epoch
    # try other settings without editing the YAML (own --work and --out):
    python tools/train_event_snn.py --model gesture16 --window-ms 3000 --hidden 1024 \
        --work logs/g16_w3000_h1024 --out pretrained/try/g16_w3000_h1024.pth

Needs the dataset as for the full-size model (gesture: DvsGestureNpy/<trial>/<class>.npy
and the trials_to_train/test lists; nmnist: Train/, Test/, Train.txt, Test.txt).

Every sample is read once and kept in memory downscaled at the model's fine
resolution (FINE_MS; cached in --work/data.pt for later runs). Training draws,
each epoch, a random window of the evaluated length (tSample) from every
training sample, with a random shift of up to one input pixel, binned into
time steps of Ts (the model's YAML); the test set uses each sample's first
window, exactly as the evaluation does. LIF neurons with an arctan surrogate
gradient, BatchNorm after the hidden layer, Adam with a cosine learning rate,
and a mean-squared error between the output firing rates and the one-hot
label (--loss ce: cross-entropy on the output spike counts). The epoch with
the best test accuracy is saved, folded, to the checkpoint; the run ends by
evaluating it as run.py loads it.
"""
import argparse
import importlib
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models  # noqa: E402

MODELS = ("gesture16", "nmnist17")
KEEP_MS = 7000.0          # length of every sample kept (unless the model sets KEEP_MS)


def model_module(name):
    """The model's module, with files(data_path, list) for its samples."""
    m = importlib.import_module(models.MODEL_MODULES[name])
    m.files = getattr(m, "sample_files", None) or m.gesture_files
    m.keep_ms = getattr(m, "KEEP_MS", KEEP_MS)
    return m


def read_split(m, root, paths, which, workers):
    """{"spikes": uint8 [N, 2, size, size, keep_ms/FINE_MS], "labels", "active"
    (bins up to the last event)} for the train or test samples."""
    listed = os.path.join(root, paths[f"list_{which}"])
    files = m.files(os.path.join(root, paths[f"dir_{which}"]), listed)
    if m.files.__name__ == "gesture_files":
        expected = m.CLASSES * sum(1 for line in open(listed) if line.strip())
        if len(files) < expected:      # DVS Gesture's errata: some trials lack some gestures
            print(f"{which}: {expected - len(files)} of {expected} gesture files missing, skipped",
                  flush=True)

    class Files(torch.utils.data.Dataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, i):
            return m.fine_spikes(files[i][0], m.keep_ms), files[i][1]

    spikes, labels = [], []
    loader = torch.utils.data.DataLoader(Files(), batch_size=256, num_workers=workers)
    for n, (tensor, label) in enumerate(loader):
        if n % 50 == 0:
            print(f"  {which}: {n * 256} of {len(files)} samples read", flush=True)
        spikes.append(tensor)
        labels.append(label)
    spikes, labels = torch.cat(spikes), torch.cat(labels)
    events = spikes.flatten(1, 3).amax(1).bool()                # [N, bins]: any event
    last = torch.where(events.any(1), events.shape[1] - 1 - events.flip(1).int().argmax(1),
                       torch.zeros_like(labels))
    return {"spikes": spikes, "labels": labels, "active": last + 1}


def batch_windows(m, data, idx, steps, step_ms, train):
    """Binary input windows [B, 2, size, size, steps]: the first (test) or a
    random one within the sample, shifted by up to one pixel (train)."""
    out = []
    span = int(round(steps * step_ms / m.FINE_MS))
    for i in idx.tolist():
        fine = data["spikes"][i]
        start = 0
        if train:
            start = int(torch.randint(0, max(1, int(data["active"][i]) - span + 1), (1,)))
        out.append(m.window(fine, start, steps, step_ms))
    x = torch.stack(out)
    if train:
        dx, dy = (int(v) for v in torch.randint(-1, 2, (2,)))
        x = torch.roll(x, (dy, dx), dims=(2, 3))
        if dy:
            x[:, :, 0 if dy > 0 else -1] = 0
        if dx:
            x[:, :, :, 0 if dx > 0 else -1] = 0
    return x


def rates(output):
    return output.reshape(output.shape[0], output.shape[1], -1).mean(-1)


def evaluate(m, net, data, steps, step_ms, device, batch=256):
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(data["labels"]), batch):
            idx = torch.arange(i, min(i + batch, len(data["labels"])))
            x = batch_windows(m, data, idx, steps, step_ms, train=False).to(device)
            correct += int((rates(net(x)).argmax(1).cpu() == data["labels"][idx]).sum())
    return 100.0 * correct / len(data["labels"])


def main():
    import run
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gesture16", choices=MODELS)
    parser.add_argument("--data", default=run.DATASET_DIR, help="dataset folder (default: run.py's)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate (Adam)")
    parser.add_argument("--loss", choices=("mse", "ce"), default="mse")
    parser.add_argument("--work", help="folder for the data cache and training state "
                                       "(default: logs/<model>_train)")
    parser.add_argument("--out", help="folded checkpoint (default: the model's pretrained file)")
    parser.add_argument("--resume", action="store_true", help="continue from --work/last.pth")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--window-ms", type=float, help="evaluated window (default: the YAML's tSample)")
    parser.add_argument("--step-ms", type=float, help="time step (default: the YAML's Ts)")
    parser.add_argument("--hidden", type=int, help="hidden neurons (default: the YAML's network: hidden)")
    parser.add_argument("--data-cache", help="dataset cache (default: --work/data.pt; share one "
                                             "between runs to read the dataset once)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model_module(args.model)
    SPEC = m.SPEC
    args.work = args.work or os.path.join(ROOT, "logs", f"{args.model}_train")
    args.out = args.out or SPEC.path(SPEC.checkpoint)
    params = models.load_params(SPEC.path(SPEC.params_yaml))
    if args.window_ms:
        params["simulation"]["tSample"] = args.window_ms
    if args.step_ms:
        params["simulation"]["Ts"] = args.step_ms
    if args.hidden:
        params.setdefault("network", {})["hidden"] = args.hidden
    overridden = any((args.window_ms, args.step_ms, args.hidden))
    step_ms = float(params["simulation"]["Ts"])
    steps = int(round(params["simulation"]["tSample"] / step_ms))
    root = models.find_dataset(SPEC, args.data)
    os.makedirs(args.work, exist_ok=True)
    cache = args.data_cache or os.path.join(args.work, "data.pt")
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    if os.path.exists(cache):
        train, test = torch.load(cache, weights_only=True).values()
    else:
        t0 = time.time()
        paths = params["training"]["path"]
        train = read_split(m, root, paths, "train", args.workers)
        test = read_split(m, root, paths, "test", args.workers)
        torch.save({"train": train, "test": test}, cache)
        print(f"read the dataset in {time.time() - t0:.0f} s (cached in {cache})", flush=True)
    print(f"{SPEC.display_name} from {root}: {len(train['labels'])} training, {len(test['labels'])} "
          f"test gestures, {steps} time steps of {step_ms:g} ms, "
          f"{params.get('network', {}).get('hidden', 512)} hidden neurons, on {device}", flush=True)
    if overridden and os.path.abspath(args.out) == os.path.abspath(SPEC.path(SPEC.checkpoint)):
        parser.error(f"--window-ms/--step-ms/--hidden differ from {SPEC.params_yaml}: give an --out "
                     "of its own (and put the chosen settings in the YAML before using it in run.py)")

    net = SPEC.network_class(params, do_enable=True, batchnorm=True).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    start, best = 0, -1.0
    if args.resume:
        state = torch.load(os.path.join(args.work, "last.pth"), map_location=device, weights_only=True)
        net.load_state_dict(state["net"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best = state["epoch"] + 1, state["best"]
        print(f"resumed after epoch {start}, best test accuracy so far {best:.2f}%", flush=True)

    for epoch in range(start, args.epochs):
        net.train()
        t0, loss_sum, correct = time.time(), 0.0, 0
        order = torch.randperm(len(train["labels"]))
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            x = batch_windows(m, train, idx, steps, step_ms, train=True).to(device)
            y = train["labels"][idx].to(device)
            r = rates(net(x))
            if args.loss == "ce":
                loss = F.cross_entropy(r * steps, y)
            else:
                loss = F.mse_loss(r, F.one_hot(y, m.CLASSES).float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(idx)
            correct += int((r.argmax(1) == y).sum())
        scheduler.step()
        accuracy = evaluate(m, net, test, steps, step_ms, device)
        line = (f"epoch {epoch + 1}/{args.epochs}: loss {loss_sum / len(order):.4f}, "
                f"train {100.0 * correct / len(order):.2f}%, lr {scheduler.get_last_lr()[0]:.2e}, "
                f"{time.time() - t0:.0f} s, test {accuracy:.2f}%")
        if accuracy > best:
            best = accuracy
            torch.save({"state_dict": net.state_dict(), "epoch": epoch, "test": accuracy},
                       os.path.join(args.work, "best.pth"))
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"state_dict": {k: v.cpu() for k, v in net.folded_state_dict().items()}},
                       args.out)
            line += " (best, saved)"
        print(line, flush=True)
        torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best},
                   os.path.join(args.work, "last.pth"))

    if not os.path.exists(args.out):
        print(f"nothing saved to {args.out}")
        return
    # The saved checkpoint, as run.py builds it, on the evaluation's own test set.
    folded = SPEC.network_class(params)
    folded.load_state_dict(models.load_tensors(args.out)["state_dict"])
    folded = folded.to(device).eval()
    loader = models.test_loader(SPEC, params, args.data, parallel=100, num_workers=args.workers)
    correct = total = 0
    with torch.no_grad():
        for _, spikes, _, label in loader:
            correct += int((rates(folded(spikes.to(device))).argmax(1).cpu() == label).sum())
            total += len(label)
    print(f"saved {args.out}: test accuracy {100.0 * correct / total:.2f}% "
          f"(best epoch {best:.2f}% in training)")
    print(f"next: python tools/accuracy_sweep.py --model {args.model} --bits 6 float")


if __name__ == "__main__":
    main()
