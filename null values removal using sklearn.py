import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
de = pd.read_csv(r"C:\Users\sheik\Downloads\Data.csv")
x = de.iloc[:,:-1].values
y = de.iloc[:,3].values
from sklearn.impute import SimpleImputer


imputer = SimpleImputer()
imputer = imputer.fit(x[:, 1:3])

x[:,1:3] = imputer.transform(x[:,1:3])
