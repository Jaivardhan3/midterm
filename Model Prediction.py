import pandas as pd
df=pd.read_csv("breast-cancer-wisconsin.data", header=None)

#df.head()

df.replace("?", 3, inplace=True)
df.drop([0], 1, inplace=True)

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(df.drop([10], 1)) #taking first 9 columns as our training data
y = np.array(df[10]) #last column as our testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=43, class_weight={4: 2}, max_iter=500, n_jobs=-1)
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.score(X_test, y_test)
print(f"Accuracy of Logistic Regression Classifier is:{lr_accuracy:.2f}")

import pickle
pickle_out = open("lr_model.pkl","wb")
pickle.dump(lr_model, pickle_out)
pickle_out.close()