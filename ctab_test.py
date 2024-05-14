import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
import sys
sys.path.append('E:\ML-Research\CTAB_PLUS')
from CTAB_PLUS.model import ctabgan
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality

random_seed = 88
label = "income"
real_path = "Adult.csv"
sampled_path = "Adult_sample.csv"

df = pd.read_csv(real_path)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

X = df.drop(label, axis=1)
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

np.random.seed(random_seed)
classifier = xgb.XGBClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

class_names = df[label].unique().astype(str)

print(classification_report(y_test, y_pred, target_names=class_names))


synthesizer =  ctabgan.CTABGAN(raw_csv_path = real_path,
                 test_ratio = 0.20,
                 categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 general_columns = ["age"],
                 non_categorical_columns = [],
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": 'income'}) 

synthesizer.fit()

fake_data = synthesizer.generate_samples()

fake_data.to_csv("ctab_test_data.csv", index=False)

run_diagnostic(
        real_data=df,
        synthetic_data=fake_data,
        metadata=metadata,
        verbose=True)
    
evaluate_quality(
    real_data=df,
    synthetic_data=fake_data,
    metadata=metadata,
    verbose=True)
