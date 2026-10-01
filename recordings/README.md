# Recorded forward passes

`run.py` and `tools/record.py` save recordings here by default (`RECORDING_DIR`
in `run.py`), one folder per model and weight quantization, e.g.
`nmnist_6b_std3/`, `gesture_6b_std3/`, `cifar10_6b_std3/`: `meta.json` and
`chunk_<first sample>.npz` files (see `evaluation/recording.py`).

They are committed so the hardware evaluation can run on any machine without
the datasets: `python run.py --model <model>` reuses a recording whose
fingerprint (checkpoint, neuron YAML, SNN code, quantization) still matches,
and records again (needing the dataset) only if one of those changed.

To add new recordings after recording them on a machine with the datasets:

    git add recordings/
    git commit -m "Add recordings"
    git push
