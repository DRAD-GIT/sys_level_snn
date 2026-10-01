"""Train the CIFAR-10 VGG-11 SNN (models/cifar10.py) and save it as its
model's checkpoint, with BatchNorm folded into the weights: --model cifar10
(rate-coded inputs, pretrained/cifar10_vgg11.pth) or cifar10_thermo
(thermometer-coded inputs, pretrained/cifar10_vgg11_thermo.pth).

    python tools/train_cifar10.py                         # dataset from run.py's DATASET_DIR
    python tools/train_cifar10.py --data /path/to/datasets --epochs 200 --amp
    python tools/train_cifar10.py --resume                # continue from the last epoch
    python tools/train_cifar10.py --amp --gpus 2 --batch 256 --lr 0.1   # both GPUs, larger batches
    python tools/train_cifar10.py --model cifar10_thermo --amp --gpus 2 --batch 256 --lr 0.1
    # fine-tune a trained network (its unfolded best.pth) with cross-entropy, in a new --work folder
    python tools/train_cifar10.py --amp --gpus 2 --batch 256 --init logs/cifar10_train/best.pth \
        --loss ce --lr 0.0002 --epochs 50 --work logs/cifar10_finetune

Needs the CIFAR-10 python release (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
extracted, i.e. a folder cifar-10-batches-py in the dataset folder, and a GPU.

Training: spike-coded inputs (tSample binary frames per image, encoded as the
model's YAML says: rate or thermometer), LIF neurons with an arctan surrogate gradient,
BatchNorm after every convolution, random crop (padding 4) and horizontal
flip, SGD with momentum 0.9 and a cosine learning rate, and a mean-squared
error between each output neuron's firing rate and the one-hot label
(--loss mse) or a cross-entropy on the output spike counts (--loss ce); the
class is the output neuron with the most spikes, as at evaluation.

--init starts from a trained network (a best.pth or last.pth of an earlier
run, which keep the BatchNorm layers) instead of random weights; its test
accuracy is measured first, so the checkpoint is only overwritten by an epoch
that beats it.

Every --eval-every epochs the test accuracy is measured (rate coding drawn
on the GPU, so it differs slightly from the fixed per-image coding of the
evaluation); the epoch with the best test accuracy is saved, folded, to the
checkpoint. Training state (last epoch and best) is kept under --work so a
run can be resumed. At the end, the saved checkpoint is reloaded the way
run.py loads it and evaluated on the exact rate coding of the recordings.
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models  # noqa: E402
from models.cifar10 import read_batches  # noqa: E402
from models.lif import encode  # noqa: E402


def augment(images):
    """Random crop with 4-pixel zero padding and random horizontal flip, on a
    [N, 3, 32, 32] batch."""
    n = images.shape[0]
    padded = F.pad(images, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), device=images.device)
    dy = torch.randint(0, 9, (n,), device=images.device)
    rows = (torch.arange(32, device=images.device)[None, :] + dy[:, None])        # [N, 32]
    cols = (torch.arange(32, device=images.device)[None, :] + dx[:, None])
    out = padded[torch.arange(n, device=images.device)[:, None, None, None],
                 torch.arange(3, device=images.device)[None, :, None, None],
                 rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(n, device=images.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def firing_rates(output):
    """[N, 10, 1, 1, T] output spikes -> [N, 10] firing rate per neuron."""
    return output.reshape(output.shape[0], output.shape[1], -1).mean(-1)


def evaluate(net, images, labels, n_steps, batch, device, encoding, seed=0):
    net.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(labels), batch):
            x = images[i:i + batch].to(device).float().div_(255)
            out = net(encode(x, encoding, n_steps, generator))
            correct += int((firing_rates(out).argmax(1).cpu() == labels[i:i + batch]).sum())
    return 100.0 * correct / len(labels)


def main():
    import run
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="cifar10", choices=("cifar10", "cifar10_thermo"),
                        help="cifar10: rate-coded inputs; cifar10_thermo: thermometer-coded")
    parser.add_argument("--data", default=run.DATASET_DIR, help="dataset folder (default: run.py's)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64, help="training batch (default 64)")
    parser.add_argument("--lr", type=float, default=0.05, help="initial learning rate (SGD)")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--eval-every", type=int, default=1, help="epochs between test evaluations")
    parser.add_argument("--amp", action="store_true", help="mixed precision (faster, less memory)")
    parser.add_argument("--work", default=os.path.join(ROOT, "logs", "cifar10_train"),
                        help="folder for the training state (last.pth, best.pth)")
    parser.add_argument("--out", help="folded checkpoint (default: the model's pretrained file)")
    parser.add_argument("--resume", action="store_true", help="continue from --work/last.pth")
    parser.add_argument("--init", help="start from this trained network (best.pth or last.pth of a "
                                       "run) instead of random weights")
    parser.add_argument("--loss", choices=("mse", "ce"), default="mse",
                        help="mse: firing rates vs one-hot labels; ce: cross-entropy on spike counts")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs to split each batch across (default 1; CUDA_VISIBLE_DEVICES picks which)")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="print progress every N training batches (default 100)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("warning: no GPU found; training on the CPU will be very slow")
    SPEC = models.get_spec(args.model)
    args.out = args.out or SPEC.path(SPEC.checkpoint)
    params = models.load_params(SPEC.path(SPEC.params_yaml))
    encoding = params.get("encoding")
    n_steps = int(params["simulation"]["tSample"] / params["simulation"]["Ts"])
    root = models.find_dataset(SPEC, args.data)
    train_x, train_y = read_batches([os.path.join(root, f"data_batch_{i}") for i in range(1, 6)])
    test_x, test_y = read_batches([os.path.join(root, "test_batch")])
    print(f"CIFAR-10 from {root}: {len(train_y)} training, {len(test_y)} test images, "
          f"{n_steps} time steps, {(encoding or {}).get('type', 'rate')} coding, on {device}", flush=True)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    net = SPEC.network_class(params, do_enable=True, batchnorm=True).to(device)
    # Several GPUs: each batch is split across them (net keeps the weights).
    gpus = min(args.gpus, torch.cuda.device_count()) if device.type == "cuda" else 1
    model = torch.nn.DataParallel(net, device_ids=list(range(gpus))) if gpus > 1 else net
    decay, no_decay = [], []
    for name, p in net.named_parameters():
        (no_decay if p.dim() == 1 else decay).append(p)       # no decay on BN and biases
    optimizer = torch.optim.SGD([{"params": decay, "weight_decay": args.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}],
                                lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    if gpus > 1:
        print(f"training on {gpus} GPUs, {args.batch // gpus} images per GPU per batch", flush=True)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    os.makedirs(args.work, exist_ok=True)
    start, best = 0, -1.0        # the first evaluation is always saved
    if args.resume:
        state = torch.load(os.path.join(args.work, "last.pth"), map_location=device, weights_only=True)
        net.load_state_dict(state["net"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best = state["epoch"] + 1, state["best"]
        print(f"resumed after epoch {start}, best test accuracy so far {best:.2f}%")
    elif args.init:
        state = torch.load(args.init, map_location=device, weights_only=True)
        net.load_state_dict(state.get("net", state.get("state_dict")))
        best = evaluate(model, test_x, test_y, n_steps, 256 * gpus, device, encoding)
        print(f"starting from {args.init}: test accuracy {best:.2f}% (saved only if beaten)", flush=True)

    for epoch in range(start, args.epochs):
        model.train()
        t0, loss_sum, correct = time.time(), 0.0, 0
        order = torch.randperm(len(train_y), device=device)
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            x = augment(train_x[idx].float().div_(255))
            y = train_y[idx]
            with torch.autocast(device.type, enabled=args.amp):
                rates = firing_rates(model(encode(x, encoding, n_steps)))
                if args.loss == "ce":
                    loss = F.cross_entropy(rates.float() * n_steps, y)     # spike counts as logits
                else:
                    loss = F.mse_loss(rates.float(), F.one_hot(y, 10).float())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(idx)
            correct += int((rates.argmax(1) == y).sum())
            done = i + len(idx)
            if done == len(idx) or done // args.batch % args.progress_every == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch + 1}: {done}/{len(order)} images, loss {loss_sum / done:.4f}, "
                      f"train {100.0 * correct / done:.2f}%, {done / elapsed:.0f} images/s, "
                      f"epoch ends in {elapsed * (len(order) - done) / done / 60:.1f} min", flush=True)
        scheduler.step()
        line = (f"epoch {epoch + 1}/{args.epochs}: loss {loss_sum / len(order):.4f}, "
                f"train {100.0 * correct / len(order):.2f}%, lr {scheduler.get_last_lr()[0]:.4f}, "
                f"{time.time() - t0:.0f} s")
        if device.type == "cuda":
            line += f", GPU memory peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            accuracy = evaluate(model, test_x, test_y, n_steps, 256 * gpus, device, encoding)
            line += f", test {accuracy:.2f}%"
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
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best}, os.path.join(args.work, "last.pth"))

    if not os.path.exists(args.out):
        print(f"no epoch beat the starting network ({best:.2f}%); nothing saved to {args.out}")
        return
    # The saved checkpoint, as run.py builds the network, on the recordings' exact rate coding.
    folded = SPEC.network_class(params)
    folded.load_state_dict(models.load_tensors(args.out)["state_dict"])
    folded = folded.to(device).eval()
    loader = models.test_loader(SPEC, params, args.data, parallel=200, num_workers=args.workers)
    correct = total = 0
    with torch.no_grad():
        for _, spikes, _, label in loader:
            out = folded(spikes.to(device))
            correct += int((firing_rates(out).argmax(1).cpu() == label).sum())
            total += len(label)
    print(f"saved {args.out}: test accuracy {100.0 * correct / total:.2f}% "
          f"(best epoch {best:.2f}% in training)")
    print(f"next: python tools/accuracy_sweep.py --model {args.model} --bits 6 float   "
          "(accuracy at the hardware's weight precision)")


if __name__ == "__main__":
    main()
