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

predictor = TabularPredictor(label=target_column, path=valohai.outputs().path(filename="model")).fit(train_data=train_data)
predictor2 = TabularPredictor.load("/valohai/outputs/model")
predictor2.save()

if not training_only:
    test_data = TabularDataset(test_file)
    predictions = predictor.evaluate(test_data)

