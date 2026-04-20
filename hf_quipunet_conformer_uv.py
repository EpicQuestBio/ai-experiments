# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "tables",
#   "torch",
#   "huggingface_hub",
# ]
# ///

from huggingface_hub import HfApi

import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


ROOT = Path.cwd()
WORKDIR = ROOT / "work"
CONFORMER_DIR = WORKDIR / "conformer_repo"
QUIPUNET_DIR = WORKDIR / "QuipuNet"

WORKDIR.mkdir(parents=True, exist_ok=True)

if not CONFORMER_DIR.exists():
    run(["git", "clone", "https://github.com/sooftware/conformer.git", str(CONFORMER_DIR)])

if not QUIPUNET_DIR.exists():
    run(["git", "clone", "https://github.com/kmisiunas/QuipuNet.git", str(QUIPUNET_DIR)])

sys.path.insert(0, str(CONFORMER_DIR))
from conformer.encoder import ConformerEncoder  # noqa: E402

NVIDIA = True
DEBUG_SMALL = False

def noiseLevels(train=None):
    global constantTypicalNoiseLevels
    if 'constantTypicalNoiseLevels' not in globals():
        constantTypicalNoiseLevels = 0.006
    if train is not None:
        tmp = np.array(list(map(np.std, train)))
        constantTypicalNoiseLevels = tmp[~np.isnan(tmp)].mean()
    return constantTypicalNoiseLevels

def barcodeToNumber(barcode):
    if len(np.shape(barcode)) == 0:
        return barcodeEncoding[barcode]
    elif len(np.shape(barcode)) == 1:
        fn = np.vectorize(lambda key: barcodeEncoding[key])
        return fn(barcode)
    elif len(np.shape(barcode)) == 2 and np.shape(barcode)[1] == 1:
        return barcodeToNumber(np.reshape(barcode, (-1,)))
    else:
        raise ValueError("Error: wrong input recieved: " + str(barcode))

def numberToBarcode(number):
    if len(np.shape(number)) == 0:
        return barcodeEncodingReverse[number]
    elif len(np.shape(number)) == 1:
        fn = np.vectorize(lambda key: barcodeEncodingReverse[key])
        return fn(number)
    else:
        raise ValueError("Error: wrong input recieved: " + str(number))

def numberToOneHot(number):
    x = np.asarray(number, dtype=np.int64)
    eye = np.eye(hp["barcodes"], dtype=np.float32)
    return eye[x]

def oneHotToNumber(onehot):
    if np.shape(onehot) == (hp['barcodes'],):
        return np.argmax(onehot)
    elif len(np.shape(onehot)) == 2 and np.shape(onehot)[1] == hp['barcodes']:
        return np.apply_along_axis(arr=onehot, func1d=np.argmax, axis=1)
    else:
        raise ValueError("Error: wrong input recieved: " + str(onehot))

def barcodeToOneHot(barcode):
    return numberToOneHot(barcodeToNumber(barcode))

def oneHotToBarcode(onehot):
    return numberToBarcode(oneHotToNumber(onehot))

def prepareTraces(dataset):
    traces = dataset.trace
    traces_uniform = traces.apply(lambda x: normaliseLength(x, length=hp["traceLength"], trim=hp["traceTrim"]))
    if hp["normalise_levels"]:
        traces_normalised = -traces_uniform / dataset.UnfoldedLevel
        return np.vstack(traces_normalised)
    else:
        return np.vstack(traces_uniform)

def normaliseLength(trace, length=600, trim=0):
    if len(trace) >= length + trim:
        return trace[trim:length + trim]
    else:
        return np.append(
            trace[trim:],
            np.random.normal(0, noiseLevels(), length - len(trace[trim:]))
        )

folder_on_drive = QUIPUNET_DIR

dataset = pd.concat([
    pd.read_hdf(folder_on_drive / "data/dataset_part1.hdf5"),
    pd.read_hdf(folder_on_drive / "data/dataset_part2.hdf5")
])
datasetTestEven = pd.read_hdf(folder_on_drive / "data/datasetTestEven.hdf5")
datasetTestOdd = pd.read_hdf(folder_on_drive / "data/datasetTestOdd.hdf5")
datasetTestMix = pd.read_hdf(folder_on_drive / "data/datasetTestMix.hdf5")
datasetWithAntibodies = pd.concat([
    pd.read_hdf(folder_on_drive / "data/datasetWithAntibodies_part1.hdf5"),
    pd.read_hdf(folder_on_drive / "data/datasetWithAntibodies_part2.hdf5")
])
datasetExtra = pd.read_hdf(folder_on_drive / "data/datasetExtra.hdf5")

print("Fraction of used data: ", dataset.Filter.sum() / len(dataset), flush=True)

hp = {
    "traceLength": 700,
    "traceTrim": 0,
    "barcodes": 8,
    "normalise_levels": True,
}

if DEBUG_SMALL:
    run_cfg = {
        "batch_size": 16,
        "epochs": 3,
        "encoder_dim": 16,
        "num_encoder_layers": 2,
        "input_proj_dim": 8,
        "learning_rate": 5e-4,
        "train_limit": 2000,
        "dev_limit": 500,
        "test_limit": 500,
    }
else:
    run_cfg = {
        "batch_size": 32,
        "epochs": 20,
        "encoder_dim": 32,
        "num_encoder_layers": 3,
        "input_proj_dim": 8,
        "learning_rate": 5e-4,
        "train_limit": None,
        "dev_limit": None,
        "test_limit": None,
    }

print("DEBUG_SMALL =", DEBUG_SMALL, flush=True)
print("run_cfg =", run_cfg, flush=True)
checkpoint_path = (
    "quipu_conformer_best_small.pt" if DEBUG_SMALL
    else "quipu_conformer_best_full.pt"
)
print("checkpoint_path =", checkpoint_path, flush=True)

HF_NAMESPACE = os.environ.get("HF_NAMESPACE", "jadicorn")
HF_MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "quipu-conformer-small" if DEBUG_SMALL else "quipu-conformer-full"
)
HF_TOKEN = os.environ.get("HF_TOKEN")

DEV_GROUPED = True
DEV_GROUP_FRACTION = 0.15
DEV_GROUP_SEED = 42
DEV_SPLIT_PATH = "quipu_grouped_dev_split.json"

barcodeEncoding = {
    "000": 0, "001": 1, "010": 2, "011": 3,
    "100": 4, "101": 5, "110": 6, "111": 7
}
barcodeEncodingReverse = {v: k for k, v in barcodeEncoding.items()}

allDatasets = pd.concat([dataset, datasetExtra, datasetWithAntibodies], ignore_index=True)
allDatasets = allDatasets[allDatasets.Filter]

testSetIndex = [
    ('000', 1017), ('001', 1053), ('010', 1159), ('011', 11),
    ('100', 1933), ('101', 1662), ('110', 12), ('111', 14)
]

testSetSelection = allDatasets[["barcode", "nanopore"]].apply(tuple, axis=1).isin(testSetIndex)

testSet = allDatasets[testSetSelection]
trainSet = allDatasets[~testSetSelection]

print("Trained noise levels:", noiseLevels(train=trainSet.trace.apply(lambda x: x[:20])), flush=True)

# Prepare full training-pool traces/labels first
X_all_train = prepareTraces(trainSet)
Y_all_barcode = np.vstack(trainSet.barcode.values)
Y_all_bound = np.vstack(trainSet.Bound.values)

# Build grouped dev split by (barcode, nanopore)
train_meta = trainSet[["barcode", "nanopore", "Bound"]].reset_index(drop=True).copy()

if DEV_GROUPED:
    train_meta = train_meta.copy()

    if os.path.exists(DEV_SPLIT_PATH):
        print(f"Loading grouped dev split from {DEV_SPLIT_PATH}")
        with open(DEV_SPLIT_PATH, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        dev_groups_by_barcode = {
            barcode: set(groups)
            for barcode, groups in split_info["dev_groups_by_barcode"].items()
        }
    else:
        print(f"Creating grouped dev split and saving to {DEV_SPLIT_PATH}")
        rng = np.random.default_rng(DEV_GROUP_SEED)

        dev_groups_by_barcode = {}

        for barcode in sorted(train_meta["barcode"].unique()):
            sub = train_meta[train_meta["barcode"] == barcode]
            groups = np.array(sorted(sub["nanopore"].unique()))

            n_dev_groups = max(1, int(np.ceil(len(groups) * DEV_GROUP_FRACTION)))
            chosen_groups = rng.choice(groups, size=n_dev_groups, replace=False)

            # convert to plain Python ints/strings for JSON
            dev_groups_by_barcode[barcode] = sorted(
                [int(x) if isinstance(x, (np.integer, int)) else x for x in chosen_groups.tolist()]
            )

        dev_counts_by_barcode = {}
        for barcode, chosen_groups in dev_groups_by_barcode.items():
            mask = (
                (train_meta["barcode"] == barcode) &
                (train_meta["nanopore"].isin(chosen_groups))
            )
            dev_counts_by_barcode[barcode] = int(mask.sum())

        split_info = {
            "seed": DEV_GROUP_SEED,
            "fraction": DEV_GROUP_FRACTION,
            "dev_groups_by_barcode": dev_groups_by_barcode,
            "dev_counts_by_barcode": dev_counts_by_barcode,
        }
        
        with open(DEV_SPLIT_PATH, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)

    dev_mask = np.zeros(len(train_meta), dtype=bool)

    for barcode, chosen_groups in dev_groups_by_barcode.items():
        barcode_mask = (
            (train_meta["barcode"] == barcode) &
            (train_meta["nanopore"].isin(chosen_groups))
        )
        dev_mask |= barcode_mask.to_numpy()

    # Safety check: every class should remain in training
    train_barcodes_after_split = set(train_meta.loc[~dev_mask, "barcode"].unique())
    expected_barcodes = set(train_meta["barcode"].unique())
    missing = expected_barcodes - train_barcodes_after_split
    if missing:
        raise RuntimeError(f"Grouped dev split removed all training samples for barcodes: {missing}")

    X_dev = X_all_train[dev_mask, :]
    Y_dev_barcode = Y_all_barcode[dev_mask, :]
    Y_dev_bound = Y_all_bound[dev_mask, :]

    X_train = X_all_train[~dev_mask, :]
    Y_train_barcode = Y_all_barcode[~dev_mask, :]
    Y_train_bound = Y_all_bound[~dev_mask, :]

    dev_meta = train_meta.loc[dev_mask].reset_index(drop=True)
    train_meta_split = train_meta.loc[~dev_mask].reset_index(drop=True)

    print("Using grouped dev split by (barcode, nanopore)")
    print("Dev groups per barcode:")
    for barcode in sorted(dev_groups_by_barcode):
        print(barcode, dev_groups_by_barcode[barcode])
else:
    # Old random split fallback
    ni_train = int(len(X_all_train) * 0.96)

    randomIndex = np.arange(len(X_all_train))
    np.random.shuffle(randomIndex)

    X_dev = X_all_train[randomIndex[ni_train:], :]
    Y_dev_barcode = Y_all_barcode[randomIndex[ni_train:], :]
    Y_dev_bound = Y_all_bound[randomIndex[ni_train:], :]

    X_train = X_all_train[randomIndex[:ni_train], :]
    Y_train_barcode = Y_all_barcode[randomIndex[:ni_train], :]
    Y_train_bound = Y_all_bound[randomIndex[:ni_train], :]

X_test = prepareTraces(testSet)
Y_test_barcode = np.vstack(testSet.barcode.values)
Y_test_bound = np.vstack(testSet.Bound.values)

Y_train = barcodeToOneHot(Y_train_barcode)
Y_dev = barcodeToOneHot(Y_dev_barcode)
Y_test = barcodeToOneHot(Y_test_barcode)

if run_cfg["train_limit"] is not None:
    idx = np.random.permutation(len(X_train))[:run_cfg["train_limit"]]
    X_train = X_train[idx]
    Y_train = Y_train[idx]

if run_cfg["dev_limit"] is not None:
    idx = np.random.permutation(len(X_dev))[:run_cfg["dev_limit"]]
    X_dev = X_dev[idx]
    Y_dev = Y_dev[idx]

if run_cfg["test_limit"] is not None:
    idx = np.random.permutation(len(X_test))[:run_cfg["test_limit"]]
    X_test = X_test[idx]
    Y_test = Y_test[idx]

print("len(X_train) = ", len(X_train), "   len(Y_train) = ", len(Y_train), flush=True)
print("len(X_dev) = ", len(X_dev), flush=True)
print("len(X_test) = ", len(X_test), flush=True)

print("Train barcode counts:")
print(pd.Series(list(map(oneHotToBarcode, Y_train))).value_counts().sort_index())

print("Dev barcode counts:")
print(pd.Series(list(map(oneHotToBarcode, Y_dev))).value_counts().sort_index())

Y_train_labels = list(map(oneHotToBarcode, Y_train))
Y_dev_labels = list(map(oneHotToBarcode, Y_dev))
Y_test_labels = list(map(oneHotToBarcode, Y_test))
labels = np.array(['000', '001', '010', '011', '100', '101', '110', '111'])

weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=labels,
    y=Y_train_labels
)

print(pd.DataFrame({
    "Train": pd.Series(Y_train_labels).value_counts(),
    "Dev": pd.Series(Y_dev_labels).value_counts(),
    "Test": pd.Series(Y_test_labels).value_counts(),
    "Weights": weights_array
}, index=labels), flush=True)

batch_size = run_cfg["batch_size"]
sequence_length = hp["traceLength"]

if NVIDIA:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

print("device =", device, flush=True)

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
X_dev_t = torch.tensor(X_dev, dtype=torch.float32).unsqueeze(-1).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)

y_train_idx = torch.tensor(oneHotToNumber(Y_train), dtype=torch.long).to(device)
y_dev_idx = torch.tensor(oneHotToNumber(Y_dev), dtype=torch.long).to(device)
y_test_idx = torch.tensor(oneHotToNumber(Y_test), dtype=torch.long).to(device)

train_lengths = torch.full((X_train_t.shape[0],), sequence_length, dtype=torch.long).to(device)
dev_lengths = torch.full((X_dev_t.shape[0],), sequence_length, dtype=torch.long).to(device)
test_lengths = torch.full((X_test_t.shape[0],), sequence_length, dtype=torch.long).to(device)

# Gets worse performance
class QuipuConformerPooledAttentionClassifier(nn.Module):
    def __init__(self, num_classes, encoder_dim=32, num_encoder_layers=3, input_proj_dim=8):
        super().__init__()

        self.input_proj = nn.Linear(1, input_proj_dim)

        self.encoder = ConformerEncoder(
            input_dim=input_proj_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
        )

        self.attn = nn.Linear(encoder_dim, 1)
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x, lengths, return_attention=False):
        x = self.input_proj(x)

        encoder_out, output_lengths = self.encoder(x, lengths)

        max_len = encoder_out.size(1)
        time_idx = torch.arange(max_len, device=encoder_out.device).unsqueeze(0)
        mask = time_idx < output_lengths.unsqueeze(1)

        attn_scores = self.attn(encoder_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)

        pooled = torch.sum(encoder_out * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, output_lengths, attn_weights
        return logits, output_lengths

class QuipuConformerClassifier(nn.Module):
    def __init__(self, num_classes, encoder_dim=32, num_encoder_layers=3, input_proj_dim=8):
        super().__init__()
        self.input_proj = nn.Linear(1, input_proj_dim)
        self.encoder = ConformerEncoder(
            input_dim=input_proj_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
        )
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x, lengths):
        x = self.input_proj(x)
        encoder_out, output_lengths = self.encoder(x, lengths)
        pooled = encoder_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, output_lengths

model = QuipuConformerClassifier(
    num_classes=hp["barcodes"],
    encoder_dim=run_cfg["encoder_dim"],
    num_encoder_layers=run_cfg["num_encoder_layers"],
    input_proj_dim=run_cfg["input_proj_dim"],
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights_array, dtype=torch.float32).to(device)
)

train_ds = TensorDataset(X_train_t, y_train_idx, train_lengths)
dev_ds = TensorDataset(X_dev_t, y_dev_idx, dev_lengths)
test_ds = TensorDataset(X_test_t, y_test_idx, test_lengths)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg["learning_rate"])

def evaluate(loader, split_name="Eval"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for xb, yb, lb in loader:
            logits, _ = model(xb, lb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_count += xb.size(0)
    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    print(f"{split_name} loss: {avg_loss:.4f}   acc: {avg_acc:.4f}", flush=True)
    return avg_loss, avg_acc

print("Initial evaluation:", flush=True)
evaluate(dev_loader, "Dev")

best_dev_loss = float("inf")
for epoch in range(run_cfg["epochs"]):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for xb, yb, lb in train_loader:
        optimizer.zero_grad()
        logits, _ = model(xb, lb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == yb).sum().item()
        total_count += xb.size(0)

    train_loss = total_loss / total_count
    train_acc = total_correct / total_count
    print(f"Epoch {epoch+1}/{run_cfg['epochs']}", flush=True)
    print(f"Train loss: {train_loss:.4f}   acc: {train_acc:.4f}", flush=True)

    dev_loss, _ = evaluate(dev_loader, "Dev")
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved best model to {checkpoint_path}", flush=True)
    print(flush=True)

if os.path.exists(checkpoint_path):
    print(f"Loading best model from {checkpoint_path}", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

print("Final test evaluation:", flush=True)
evaluate(test_loader, "Test")

model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb, lb in test_loader:
        logits, _ = model(xb, lb)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.cpu().numpy())

cm = confusion_matrix(all_true, all_preds)
print("Confusion matrix:", flush=True)
print(cm, flush=True)

all_preds = np.array(all_preds)
all_true = np.array(all_true)
print("Per-class test accuracy:", flush=True)
for i, label in enumerate(labels):
    mask = (all_true == i)
    if mask.sum() == 0:
        continue
    acc_i = (all_preds[mask] == all_true[mask]).mean()
    print(label, "count =", mask.sum(), "acc =", float(acc_i), flush=True)

if os.path.exists(checkpoint_path):
    if not HF_TOKEN:
        print("HF_TOKEN not set; skipping model upload.", flush=True)
    else:
        print("Uploading best checkpoint to Hugging Face...", flush=True)

        api = HfApi(token=HF_TOKEN)
        repo_id = f"{HF_NAMESPACE}/{HF_MODEL_REPO}"

        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
        )

        remote_name = checkpoint_path
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="model",
        )

        print(f"Uploaded checkpoint to https://huggingface.co/{repo_id}", flush=True)

if os.path.exists(DEV_SPLIT_PATH):
    api.upload_file(
        path_or_fileobj=DEV_SPLIT_PATH,
        path_in_repo=DEV_SPLIT_PATH,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded dev split JSON to https://huggingface.co/{repo_id}", flush=True)
else:
    print(f"Warning: {DEV_SPLIT_PATH} not found; skipping split upload.", flush=True)

run_metadata_path = "quipu_run_metadata.json"
run_metadata = {
    "debug_small": DEBUG_SMALL,
    "checkpoint_path": checkpoint_path,
    "dev_split_path": DEV_SPLIT_PATH,
    "run_cfg": run_cfg,
    "hp": hp,
}

with open(run_metadata_path, "w", encoding="utf-8") as f:
    json.dump(run_metadata, f, indent=2)

api.upload_file(
    path_or_fileobj=run_metadata_path,
    path_in_repo=run_metadata_path,
    repo_id=repo_id,
    repo_type="model",
)
print(f"Uploaded run metadata to https://huggingface.co/{repo_id}", flush=True)

