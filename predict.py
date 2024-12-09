from autogluon.tabular import TabularDataset, TabularPredictor # type: ignore
import valohai
import pandas as pd
from os import path

predictor = TabularPredictor.load(valohai.inputs("model").path())
input_files = valohai.inputs("files").paths()

print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:", predictor.features())

for file in input_files:
    predictions = predictor.predict(file)
    output_csv = path.basename(file) + ".csv"
    predictions.to_csv(path.join("/valohai/outputs", output_csv))