from autogluon.tabular import TabularDataset, TabularPredictor # type: ignore
import os
import valohai

target_column = valohai.parameters("target_column").value
training_only = False

if target_column is None or target_column == "":
    print("Need a target column specified!")

data_root = os.getenv("DATA_ROOT", "/valohai/inputs")

train_file = os.path.join(data_root, 'train.csv')
test_file = os.path.join(data_root, "test.csv")

if not os.path.exists(train_file):
    print("Need a training file to start the training.")
    exit(1)

if not os.path.exists(test_file):
    print("Test file not found. Running training only")
    training_only = True

train_data = TabularDataset(train_file)

predictor = TabularPredictor(label=target_column).fit(train_data=train_data)

if not training_only:
    test_data = TabularDataset(test_file)
    predictions = predictor.predict(test_data)
