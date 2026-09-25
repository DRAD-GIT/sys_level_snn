# N-MNIST (placeholder)

Put the N-MNIST dataset here. `models/nmnist.yaml` expects:

```text
datasets/N-MNIST/
├── Test/        # 00000.bin, 00001.bin, ... (binary event files)
├── Test.txt     # sample list: "<index> <label>" per line
├── Train/       # (training only)
└── Train.txt
```

Only `Test/` and `Test.txt` are needed for evaluation. To keep the data
elsewhere, edit the `path:` entries in `models/nmnist.yaml` (absolute paths
work; keep the trailing slash on directories). Dataset files are git-ignored.
