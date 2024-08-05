# performing linear algebra 
import numpy as np  
  
# data processing 
import pandas as pd 
  
# visualisation 
import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv("data.csv")

# Drop unnecessary columns
df = df.drop(['Unnamed: 32', 'id'], axis=1)

# Convert diagnosis to numerical values
def diagnosis_value(diagnosis):
    return 1 if diagnosis == 'M' else 0

df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)

# Plot using Seaborn
sns.lmplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df)
sns.lmplot(x='smoothness_mean', y='compactness_mean', hue='diagnosis', data=df)

# Prepare the data for KNN
X = np.array(df.iloc[:, 1:])
y = np.array(df['diagnosis'])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.33, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)

# Evaluate the classifier
accuracy = knn.score(X_test, y_test)
print("Accuracy: ", accuracy)

# Perform cross-validation to find the optimal number of neighbors
neighbors = []
cv_scores = []

for k in range(1, 51, 2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

# Determine the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is %d' % optimal_k)

# Plot misclassification error versus k
plt.figure(figsize=(10, 6))
plt.plot(neighbors, MSE)
plt.xlabel('Number of neighbors')
plt.ylabel('Misclassification Error')
plt.show()
