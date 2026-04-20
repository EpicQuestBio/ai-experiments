# Imports

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib, seaborn
import h5py
import scipy
import sklearn
import time

import Quipu
from Quipu.kerasHelpers import resetHistory, nextEpochNo
from Quipu.tools import normaliseLength
from Quipu import augment

import keras
from keras import layers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding
from keras.optimizers import SGD, Adam
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard

# Load cleaned datasets

folder_on_drive = './'

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

pyplot.figure(figsize=[16,14])

data = pd.concat([dataset, datasetWithAntibodies], ignore_index=True)

print("Blue = NBell could indentify; Orange = NBell did not indentify; Red = excluded by filtration step")

examples_included = data[data.Filter ].sample(5*5)
for i in range(5*5):
    one = examples_included.iloc[i]
    pyplot.subplot(5+1,5,i+1)
    if one.nbell_barcode < 8:
        pyplot.plot(one.trace)
    else:
        pyplot.plot(one.trace, color='orange')
    # labels
    pyplot.text(650.0, -0.31, one.barcode, fontsize=12, horizontalalignment='right')
    bound_label = "Bound" if one.Bound else "Unbound"
    pyplot.text(650.0, -0.37, bound_label, fontsize=12, horizontalalignment='right')
    pyplot.plot([0,700], [one.UnfoldedLevel]*2, 'r--')
    pyplot.ylim([-0.4, 0.05])
    pyplot.xlim([0, 700])
    
    
#pyplot.figure(figsize=[16,2.2])
examples_excluded = data[~data.Filter].sample(1*5)
for i in range(1*5):
    one = examples_excluded.iloc[i]
    pyplot.subplot(5+1,5,5*5+i+1)
    pyplot.plot(one.trace, color='red')
    pyplot.plot([0,700], [one.UnfoldedLevel]*2, 'r--')
    pyplot.ylim([-0.4, 0.05])
    pyplot.xlim([0, 700]) 


pyplot.show()

# Hyperparameters

hp = {
    "traceLength" : 700,
    "traceTrim"   : 0,
    "barcodes"    : 8,        # distinct barcode count 
    "normalise_levels": True, # wherther to normalise experiments per batch before feetingh into NN
}

DEV_SPLIT_PATH = "quipu_grouped_dev_split.json"


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


## Helper functions

def prepareTraces(dataset):
    "trims, clips, and reformats the traces"
    traces = dataset.trace
    traces_uniform = traces.apply(lambda x: normaliseLength(x, length = hp["traceLength"], trim = hp["traceTrim"]))
    if hp["normalise_levels"]:
        traces_normalised =  - traces_uniform / dataset.UnfoldedLevel                     
        return np.vstack( traces_normalised )
    else:
        return np.vstack( traces_uniform )
    
def prepareLabels(dataset): 
    "prepare barcode labels for training and testing"
    # for barcodes we use one shot encoding
    return barcodeToOneHot( dataset.barcode )
 
    
    
## Covert between data types
# All should accept arrays OR singular values
    
# reverse encoding 
barcodeEncodingReverse = {v: k for k, v in barcodeEncoding.items()}
    
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
    return keras.utils.to_categorical(number, num_classes= hp["barcodes"])
    
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
    
    
    
def labelToNumber(barcode):
    raise ValueError("Replace labelToNumber")
def numberToLabel(number):
    raise ValueError("Replace numberToLabel")
def toCategories(barcode):
    raise ValueError("Replace toCategories")
def fromCategories(x):
    raise ValueError("Replace fromCategories")
    
    
## tests

assert barcodeEncoding == {v: k for k, v in barcodeEncodingReverse.items()}, "There is non unique numbers in the `barcodeEncoding`"
assert hp["barcodes"] == len(barcodeEncoding), "Hyperparameter `barcodes` does not match the count in `barcodeEncoding`"

assert  numberToOneHot(1).shape == ( hp['barcodes'],)
assert  numberToOneHot([0,1,2]).shape == (3, hp['barcodes'])

assert oneHotToNumber([0,1,0,0,0,0,0,0]) == 1
assert np.all( oneHotToNumber(numberToOneHot([0,1,2])) == np.array([0,1,2])  )

assert barcodeToNumber('001') == 1
assert np.all( barcodeToNumber(['001', '000']) == np.array([1,0]) )

tmpTestBarcode = np.random.choice(list(barcodeEncoding.keys()))
assert numberToBarcode( barcodeToNumber( tmpTestBarcode ) ) == tmpTestBarcode
assert np.all( numberToBarcode( barcodeToNumber( [tmpTestBarcode]*4 ) ) == np.array([tmpTestBarcode]*4) )

assert oneHotToBarcode( barcodeToOneHot( tmpTestBarcode ) ) == tmpTestBarcode
assert np.all( oneHotToBarcode( barcodeToOneHot( [tmpTestBarcode]*4 ) ) == np.array([tmpTestBarcode]*4) )



## Test set selection tool  

print(dataset[dataset.Filter][["barcode", "nanopore", "Bound"]].copy()[:100])

tmp = pd.concat([
    dataset[dataset.Filter][["barcode", "nanopore", "Bound"]].copy() ,
    datasetExtra[datasetExtra.Filter][["barcode", "nanopore", "Bound"]].copy(),
    datasetWithAntibodies[datasetWithAntibodies.Filter][["barcode", "nanopore", "Bound"]].copy()
], ignore_index = True)
tmp = tmp.groupby(tmp.columns.tolist(),as_index=False).size()

tmp2 = pd.concat([
    dataset[dataset.nbell_barcode < 8][["barcode", "nanopore", "Bound"]].copy() 
], ignore_index = True)
tmp2 = tmp2.groupby(tmp2.columns.tolist(),as_index=False).size()

#pd.DataFrame({
#    "Full": tmp,
#    "NBell": tmp2
#})
     

# Select Test Set

#  Important: select independant nanopore experiments to avoid 
#  any correlations between the measurements. 

allDatasets = pd.concat([dataset , datasetExtra, datasetWithAntibodies],ignore_index = True)
allDatasets = allDatasets[allDatasets.Filter] # clear bad points

# Using one test set to develop the NN architecture and another one to report final results
# to reduce over-fitting to the test set


# selected the smallest independent experiments for each barcode 
# testSetIndex = [
#     ('000', 1017),
#     ('001', 1053),
#     ('010', 1159),
#     ('011', 11),
#     ('100', 1933),
#     ('101', 1662),
#     ('110', 12),
#     ('111', 14)
# ]

# # # for bound states 
# testSetIndex = [
#     ('000', 6),
#     ('001', 26),
#     ('010', 9),
#     ('011', 38),
#     ('100', 7),
#     ('101', 1662),
#     ('110', 12),
#     ('111', 32)
# ]




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
     

# prepare data

print("Trained noise levels:", 
    Quipu.tools.noiseLevels(train = trainSet.trace.apply(lambda x: x[:20]) ) )

X_train = prepareTraces( trainSet )
Y_train_barcode = np.vstack( trainSet.barcode.values )
Y_train_bound =   np.vstack( trainSet.Bound )


train_meta = trainSet[["barcode", "nanopore", "Bound"]].reset_index(drop=True).copy()

if not os.path.exists(DEV_SPLIT_PATH):
    raise FileNotFoundError(f"Missing grouped dev split file: {DEV_SPLIT_PATH}")

with open(DEV_SPLIT_PATH, "r", encoding="utf-8") as f:
    split_info = json.load(f)

dev_groups_by_barcode = {
    barcode: set(groups)
    for barcode, groups in split_info["dev_groups_by_barcode"].items()
}

dev_mask = np.zeros(len(train_meta), dtype=bool)

for barcode, chosen_groups in dev_groups_by_barcode.items():
    barcode_mask = (
        (train_meta["barcode"] == barcode) &
        (train_meta["nanopore"].isin(chosen_groups))
    )
    dev_mask |= barcode_mask.to_numpy()

X_dev = X_train[dev_mask, :]
Y_dev_barcode = Y_train_barcode[dev_mask, :]
Y_dev_bound = Y_train_bound[dev_mask, :]

X_train = X_train[~dev_mask, :]
Y_train_barcode = Y_train_barcode[~dev_mask, :]
Y_train_bound = Y_train_bound[~dev_mask, :]

# test set is independant
X_test = prepareTraces( testSet )
Y_test_barcode = np.vstack( testSet.barcode.values )
Y_test_bound = np.vstack( testSet.Bound.values)

print("Loaded grouped dev split from", DEV_SPLIT_PATH)
print("Dev groups per barcode:")
for barcode in sorted(dev_groups_by_barcode):
    print(barcode, sorted(dev_groups_by_barcode[barcode]))

# prepare categories
Y_train = barcodeToOneHot(Y_train_barcode)
Y_dev   = barcodeToOneHot(Y_dev_barcode)
Y_test  = barcodeToOneHot(Y_test_barcode)

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

# Model

input_trace = Input(shape=(hp["traceLength"],1), dtype='float32', name='input')

x = Conv1D(64, 7, padding="same")(input_trace)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)
x = Conv1D(64, 7, padding="same")(x)
x = BatchNormalization(axis=1)(x) 
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=3)(x)
x = Dropout(0.25)(x)

x = Conv1D(128, 5, padding="same")(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)
x = Conv1D(128, 5, padding="same")(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=3)(x)
x = Dropout(0.25)(x)

x = Conv1D(256, 3, padding="same")(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)
x = Conv1D(256, 3, padding="same")(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=3)(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x) # change from 512 to 128 or 256 for fast training 
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)

# Problem specific below 

output_barcode = Dense(hp['barcodes'], activation='softmax', name='output_barcode')(x)
model = Model(inputs=input_trace, outputs=output_barcode)

model.compile(
    loss = 'categorical_crossentropy', 
    optimizer = Adam(learning_rate=0.001),
    metrics = ['accuracy']
)

shapeX = (-1, hp["traceLength"],1); 
shapeY = (-1, hp['barcodes'])
#tensorboard, history = resetHistory()



# training method

lr = 0.001

for n in range(0, 60):
    print("=== Epoch:", n,"===")
    start_time = time.time()
    # data augmentation
    X = np.repeat(X_train, 1, axis=0) # make copies
    Y = np.repeat(Y_train, 1, axis=0)
    X = augment.magnitude(X, std = 0.08) 
    X = augment.stretchDuration(X, std=0.1, probability=0.3)
    X = augment.addNoise( X, std = 0.08) # typical noise after normalisation = 0.044
    # Learning rate decay
    lr = lr*0.97
    model.optimizer.learning_rate.assign(lr)
    preparation_time = time.time() - start_time
    
    global out_history
    # Fit the model
    out_history = model.fit( 
        x = X.reshape(shapeX), 
        y = Y.reshape(shapeY), 
        batch_size=32, shuffle = True,
        initial_epoch = n,  epochs=n+1,
        class_weight = weights, # not used
        validation_data=(X_dev.reshape(shapeX),  Y_dev.reshape(shapeY)),
        #callbacks = [tensorboard, history], verbose = 0
    )
    training_time = time.time() - start_time - preparation_time
    
    # Feedback 
    print('  prep time: %3.1f sec' % preparation_time, 
          '  train time: %3.1f sec' % training_time)
    print('  loss: %5.3f' % out_history.history['loss'][0] ,
          '  accuracy: %5.4f' % out_history.history['accuracy'][0] ,
          '  val_accuracy: %5.4f' % out_history.history['val_accuracy'][0] 
    #       '  acc: %5.4f' % out_history.history['output_barcode_acc'][0] ,
    #       '  acc (bound): %5.4f' % out_history.history['output_binding_acc'][0] ,
    )
    
     

# load trained model from memory

model.save(folder_on_drive + "models/barcode_metric_test.h5")
model = keras.models.load_model(folder_on_drive + "models/barcode_metric_test.h5")
shapeX = (-1, hp["traceLength"],1); 
shapeY = (-1, hp['barcodes'])
     

# save model

model.save(folder_on_drive + "models/barcode_metric_test.h5")
     
# Measure evaluation speed

t0 = time.time()
model.evaluate(x = X_dev.reshape(shapeX),   y = Y_dev,   verbose=False) 
dt = time.time() - t0
print("Evaluation speed: {:.2f} traces/s ".format(len(X_dev) /dt))

# the speed varies depending on the hardware available.

# COLAB cpu 24.03 traces/s 
# COLAB gpu 2333.73 traces/s 

print("       [ loss , accuracy ]")
print("Train:", model.evaluate(x = X_train.reshape(shapeX), y = Y_train, verbose=False) )
print("Dev  :", model.evaluate(x = X_dev.reshape(shapeX),   y = Y_dev,   verbose=False) )
print("Test :", model.evaluate(x = X_test.reshape(shapeX),  y = Y_test,  verbose=False) )

from sklearn.metrics import confusion_matrix, classification_report

# Predict on test
test_probs = model.predict(X_test.reshape(shapeX), verbose=0)
test_pred_idx = np.argmax(test_probs, axis=1)
test_true_idx = np.argmax(Y_test, axis=1)

cm = confusion_matrix(test_true_idx, test_pred_idx)
print("Confusion matrix:")
print(cm)

print("Per-class test accuracy:")
for i, label in enumerate(labels):
    mask = (test_true_idx == i)
    if mask.sum() == 0:
        continue
    acc_i = (test_pred_idx[mask] == test_true_idx[mask]).mean()
    print(label, "count =", int(mask.sum()), "acc =", float(acc_i))

print("\nClassification report:")
print(classification_report(test_true_idx, test_pred_idx, target_names=labels, digits=4))

# Save CSV like the Conformer metrics script
test_meta = testSet[["barcode", "nanopore", "Bound"]].reset_index(drop=False).copy()
test_meta = test_meta.rename(columns={"index": "orig_index"})

results_df = test_meta.copy()
results_df["true_idx"] = test_true_idx
results_df["pred_idx"] = test_pred_idx
results_df["true_label"] = [labels[i] for i in test_true_idx]
results_df["pred_label"] = [labels[i] for i in test_pred_idx]
results_df["correct"] = (test_true_idx == test_pred_idx)
results_df["pred_confidence"] = test_probs.max(axis=1)

for i, label in enumerate(labels):
    results_df[f"prob_{label}"] = test_probs[:, i]

results_df.to_csv("quipunet_test_predictions.csv", index=False)
print("Saved test predictions to quipunet_test_predictions.csv")

