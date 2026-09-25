# IBM DVS-Gesture (placeholder)

Put the DVS-Gesture dataset here, converted to per-class `.npy` spike files.
`models/gesture.yaml` expects:

```text
datasets/DVS_Gesture/
├── DvsGestureNpy/
│   └── <trial>/0.npy ... 10.npy   # one file per gesture class, per trial
└── DvsGesture/
    ├── trials_to_test.txt         # trial file names (e.g. user24_led.aedat)
    └── trials_to_train.txt
```

The test loader reads `DvsGestureNpy/<trial name without extension>/<class>.npy`
for every trial listed in `trials_to_test.txt`. To keep the data elsewhere,
edit the `path:` entries in `models/gesture.yaml`. Dataset files are git-ignored.
