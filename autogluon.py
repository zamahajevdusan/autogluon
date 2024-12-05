from autogluon.tabular import TabularDataset, TabularPredictor # type: ignore
import os
import valohai

target_column = valohai.parameters("target_column").value
training_only = False

if target_column is None or target_column == "":
    print("Need a target column specified!")


train_file = valohai.inputs("train").path()
test_file = valohai.inputs("test").path()

if train_file is None:
    print("Need a training file to start the training.")
    exit(1)

if test_file is None:
    print("Test file not found. Running training only")
    training_only = True

train_data = TabularDataset(train_file)

predictor = TabularPredictor(label=target_column).fit(train_data=train_data)

if not training_only:
    test_data = TabularDataset(test_file)
    predictions = predictor.predict(test_data)

