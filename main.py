import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Read file
titanic = pd.read_csv("titanic.csv")
# Need columns
keep_col = ['Pclass', 'Fare', 'Age', 'Sex', 'Survived']
# Filter by need columns
data = titanic[keep_col]
# Need to change
sex_to_bool = {"male": 1, "female": 0}
# Change sex values to bool
data["Sex"].replace(sex_to_bool, inplace=True)
# Filter nan
data = data[~np.isnan(data['Age'])]
# Save proceed data to file
data.to_csv("titanic_filtered.csv", index=False)
# Needed columns
X = data[['Pclass', 'Fare', 'Age', 'Sex']]
# Target variable
Y = data[['Survived']]
# Decision tree constructor
clf = DecisionTreeClassifier(random_state=241)
# Train model
clf.fit(X, Y)
importances = clf.feature_importances_
# Important predicates
print(importances)
