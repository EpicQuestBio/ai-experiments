import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from conformer.encoder import ConformerEncoder

# Whether you have an NVidia or AMD GPU
NVIDIA = True
DEBUG_SMALL = False

## Helper functions

def noiseLevels(train = None):
    """
    Gives typical noise levels in the system 
    
    :param train: data to train on (numpy array)
    :return: typical noise levels (default: 0.006)
    """
    global constantTypicalNoiseLevels
    #constantTypicalNoiseLevels = 4
    if 'constantTypicalNoiseLevels' not in globals():
        constantTypicalNoiseLevels = 0.006 # default
    if train is not None:
        tmp = np.array( list(map(np.std, train)) );
        constantTypicalNoiseLevels = tmp[~np.isnan(tmp)].mean()
    return constantTypicalNoiseLevels

def barcodeToNumber(barcode):
    "translates the barcode string into number"
    if len(np.shape(barcode)) == 0 :
        return barcodeEncoding[barcode]
    elif len(np.shape(barcode)) == 1:
        fn = np.vectorize(lambda key: barcodeEncoding[key])
        return fn(barcode)
    elif len(np.shape(barcode)) == 2 and np.shape(barcode)[1] == 1:
        return barcodeToNumber(np.reshape(barcode, (-1,)))
    else:
        raise ValueError("Error: wrong input recieved: "+str(barcode))

def numberToBarcode(number):
    "number to barcode string"
    if len(np.shape(number)) == 0 :
        return barcodeEncodingReverse[number]
    elif len(np.shape(number)) == 1:
        fn = np.vectorize(lambda key: barcodeEncodingReverse[key])
        return fn(number)
    else:
        raise ValueError("Error: wrong input recieved: "+str(number))
    
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
        raise ValueError("Error: wrong input recieved: "+str(onehot))

    
def barcodeToOneHot(barcode):
    "barcode string to catogory encoding aka One-Hot"
    return numberToOneHot( barcodeToNumber(barcode) )
    
def oneHotToBarcode(onehot):
    "catogory encoding aka One-Hot to barcode string"
    return numberToBarcode( oneHotToNumber(onehot) )


def prepareTraces(dataset):
    "trims, clips, and reformats the traces"
    traces = dataset.trace
    traces_uniform = traces.apply(lambda x: normaliseLength(x, length = hp["traceLength"], trim = hp["traceTrim"]))
    if hp["normalise_levels"]:
        traces_normalised =  - traces_uniform / dataset.UnfoldedLevel                     
        return np.vstack( traces_normalised )
    else:
        return np.vstack( traces_uniform )

def normaliseLength(trace, length = 600, trim = 0):
    """
    Normalizes the length of the trace and trims the front 
    
    :param length: length to fit the trace into (default: 600)
    :param trim: how many points to drop in front of the trace (default: 0)
    :return: trace of length 'length' 
    """
    if len(trace) >= length + trim:
        return trace[trim : length+trim]
    else:
        return np.append(
            trace[trim:],
            np.random.normal(0, noiseLevels(), length - len(trace[trim:]))
        )

# Load cleaned datasets

folder_on_drive = '../QuipuNet/'

dataset =         pd.concat([ 
    pd.read_hdf(folder_on_drive + "data/dataset_part1.hdf5"),
    pd.read_hdf(folder_on_drive + "data/dataset_part2.hdf5")
])
datasetTestEven = pd.read_hdf(folder_on_drive + "data/datasetTestEven.hdf5")
datasetTestOdd =  pd.read_hdf(folder_on_drive + "data/datasetTestOdd.hdf5")
datasetTestMix =  pd.read_hdf(folder_on_drive + "data/datasetTestMix.hdf5")
datasetWithAntibodies =  pd.concat([ 
    pd.read_hdf(folder_on_drive + "data/datasetWithAntibodies_part1.hdf5"),
    pd.read_hdf(folder_on_drive + "data/datasetWithAntibodies_part2.hdf5")
])
datasetExtra =    pd.read_hdf(folder_on_drive + "data/datasetExtra.hdf5")

print("Fraction of used data: ", dataset.Filter.sum()/ len(dataset))

data = pd.concat([dataset, datasetWithAntibodies], ignore_index=True)

# Hyperparameters

hp = {
    "traceLength" : 700,
    "traceTrim"   : 0,
    "barcodes"    : 8,        # distinct barcode count 
    "normalise_levels": True, # wherther to normalise experiments per batch before feetingh into NN
}

if DEBUG_SMALL:
    run_cfg = {
        "batch_size": 16,
        "epochs": 3,
        "encoder_dim": 16,
        "num_encoder_layers": 2,
        "input_proj_dim": 8,
        "learning_rate": 1e-3,
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
        "learning_rate": 1e-3,
        "train_limit": None,
        "dev_limit": None,
        "test_limit": None,
    }

print("DEBUG_SMALL =", DEBUG_SMALL)
print("run_cfg =", run_cfg)
checkpoint_path = "quipu_conformer_best_small.pt" if DEBUG_SMALL else "quipu_conformer_best_full.pt"
print("checkpoint_path =", checkpoint_path)

# barcode binnary encoding: int(barcode,2) 
# explicit for customistation 
barcodeEncoding = {
    "000" : 0,
    "001" : 1,
    "010" : 2,
    "011" : 3,
    "100" : 4,
    "101" : 5,
    "110" : 6,
    "111" : 7
}

# reverse encoding 
barcodeEncodingReverse = {v: k for k, v in barcodeEncoding.items()}

# Select Test Set

#  Important: select independant nanopore experiments to avoid 
#  any correlations between the measurements. 

allDatasets = pd.concat([dataset , datasetExtra, datasetWithAntibodies],ignore_index = True)
allDatasets = allDatasets[allDatasets.Filter] # clear bad points

# Using one test set to develop the NN architecture and another one to report final results
# to reduce over-fitting to the test set

# selected the smallest independent experiments for each barcode 
testSetIndex = [
    ('000', 1017),
    ('001', 1053),
    ('010', 1159),
    ('011', 11),
    ('100', 1933),
    ('101', 1662),
    ('110', 12),
    ('111', 14)
]

testSetSelection = allDatasets[["barcode", "nanopore"]]\
                        .apply(tuple, axis = 1)\
                        .isin(testSetIndex)

testSet = allDatasets[ testSetSelection ]
trainSet = allDatasets[ ~ testSetSelection ]

print("Trained noise levels:", 
    noiseLevels(train = trainSet.trace.apply(lambda x: x[:20]) ) )


X_train = prepareTraces( trainSet )
Y_train_barcode = np.vstack( trainSet.barcode.values )
Y_train_bound =   np.vstack( trainSet.Bound )

# Slit data into training and dev sets(randomly)
ni_train = int( len(X_train)*0.96 ) # training set
ni_dev   = len(X_train) - ni_train  # dev set

randomIndex = np.arange(len(X_train))
np.random.shuffle(randomIndex) 

X_dev = X_train[randomIndex[ni_train:] , :]
Y_dev_barcode = Y_train_barcode[randomIndex[ni_train:], :]
Y_dev_bound = Y_train_bound[randomIndex[ni_train:], :]

X_train = X_train[randomIndex[:ni_train],:]
Y_train_barcode = Y_train_barcode[randomIndex[:ni_train] , :]
Y_train_bound = Y_train_bound[randomIndex[:ni_train] , :]

# test set is independant
X_test = prepareTraces( testSet )
Y_test_barcode = np.vstack( testSet.barcode.values )
Y_test_bound = np.vstack( testSet.Bound.values)

# prepare categories
Y_train = barcodeToOneHot(Y_train_barcode)
Y_dev   = barcodeToOneHot(Y_dev_barcode)
Y_test  = barcodeToOneHot(Y_test_barcode)

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

print("len(X_train) = ", len(X_train) , "   len(Y_train) = ", len(Y_train))
print("len(X_dev) = ", len(X_dev) )
print("len(X_test) = ", len(X_test) )

# estimate class weights to reduce overfitting

from sklearn.utils import class_weight
Y_train_labels = list(map(oneHotToBarcode, Y_train))
Y_dev_labels = list(map(oneHotToBarcode, Y_dev))
Y_test_labels = list(map(oneHotToBarcode, Y_test))
#labels = np.array(['wrong','000', '001', '010', '011', '100', '101', '110', '111'])
labels = np.array(['000', '001', '010', '011', '100', '101', '110', '111'])

weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=labels,
    y=Y_train_labels
)

weights = {i: float(w) for i, w in enumerate(weights_array)}

print(pd.DataFrame({
    "Train": pd.Series(Y_train_labels).value_counts(),
    "Dev": pd.Series(Y_dev_labels).value_counts(),
    "Test": pd.Series(Y_test_labels).value_counts(),
    "Weights": weights_array
}, index=labels))


# Set up Conformer for barcode classification

batch_size = run_cfg["batch_size"]
sequence_length = hp["traceLength"]

if NVIDIA:
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
else:
    import torch_directml
    device = torch_directml.device()

# Convert numpy data to torch tensors
# Shape: (batch, time, channels)
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
X_dev_t   = torch.tensor(X_dev, dtype=torch.float32).unsqueeze(-1).to(device)
X_test_t  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)

# CrossEntropyLoss wants class indices, not one-hot vectors
y_train_idx = torch.tensor(oneHotToNumber(Y_train), dtype=torch.long).to(device)
y_dev_idx   = torch.tensor(oneHotToNumber(Y_dev), dtype=torch.long).to(device)
y_test_idx  = torch.tensor(oneHotToNumber(Y_test), dtype=torch.long).to(device)

# All traces have fixed length after preprocessing
train_lengths = torch.full((X_train_t.shape[0],), sequence_length, dtype=torch.long).to(device)
dev_lengths   = torch.full((X_dev_t.shape[0],), sequence_length, dtype=torch.long).to(device)
test_lengths  = torch.full((X_test_t.shape[0],), sequence_length, dtype=torch.long).to(device)


class QuipuConformerClassifier(nn.Module):
    def __init__(self, num_classes, encoder_dim=32, num_encoder_layers=3, input_proj_dim=8):
        super().__init__()

        # Project raw scalar current at each timestep into a small feature vector
        self.input_proj = nn.Linear(1, input_proj_dim)

        self.encoder = ConformerEncoder(
            input_dim=input_proj_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
        )

        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x, lengths):
        # x: (batch, time, 1)
        x = self.input_proj(x)   # -> (batch, time, input_proj_dim)

        encoder_out, output_lengths = self.encoder(x, lengths)
        # encoder_out: (batch, time', encoder_dim)

        # Simple mean pooling over time
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

# Build datasets and loaders

train_ds = TensorDataset(X_train_t, y_train_idx, train_lengths)
dev_ds   = TensorDataset(X_dev_t, y_dev_idx, dev_lengths)
test_ds  = TensorDataset(X_test_t, y_test_idx, test_lengths)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader   = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg["learning_rate"])

def evaluate(loader, split_name="Eval"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for xb, yb, lb in loader:
            logits, out_lengths = model(xb, lb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_count += xb.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    print(f"{split_name} loss: {avg_loss:.4f}   acc: {avg_acc:.4f}")
    return avg_loss, avg_acc

# Initial sanity check
print("Initial evaluation:")
evaluate(dev_loader, "Dev")

# Train loop
epochs = run_cfg["epochs"]
best_dev_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb, lb in train_loader:
        optimizer.zero_grad()

        logits, out_lengths = model(xb, lb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == yb).sum().item()
        total_count += xb.size(0)

    train_loss = total_loss / total_count
    train_acc = total_correct / total_count

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train loss: {train_loss:.4f}   acc: {train_acc:.4f}")

    dev_loss, dev_acc = evaluate(dev_loader, "Dev")

    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved best model to {checkpoint_path}")

    print()

# Load best model
import os

if os.path.exists(checkpoint_path):
    print(f"Loading best model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print(f"Warning: checkpoint not found: {checkpoint_path}")

print("Final test evaluation:")
evaluate(test_loader, "Test")

from sklearn.metrics import confusion_matrix

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
print("Confusion matrix:")
print(cm)


all_preds = np.array(all_preds)
all_true = np.array(all_true)

print("Per-class test accuracy:")
for i, label in enumerate(labels):
    mask = (all_true == i)
    if mask.sum() == 0:
        continue
    acc_i = (all_preds[mask] == all_true[mask]).mean()
    print(label, "count =", mask.sum(), "acc =", float(acc_i))
