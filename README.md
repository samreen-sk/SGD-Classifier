# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data.

2.Split Dataset into Training and Testing Sets.

3.Train the Model Using Stochastic Gradient Descent (SGD).

4.Make Predictions and Evaluate Accuracy.

5.Generate Confusion Matrix.
## Program and Output:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SHAIK SAMREEN
RegisterNumber:  212223110047
*/
```
```python
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
iris=load_iris() 
df=pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['target']=iris.target 
print(df.head())
```
![433054175-c8abee43-9360-4147-9065-6c618011a393](https://github.com/user-attachments/assets/2e393fbb-5ddb-4576-827d-911b7ef43ed3)
```python
X = df.drop('target',axis=1) 
y=df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)

y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
![433054188-e47fccb5-9406-4048-8943-a3c583a0c91c](https://github.com/user-attachments/assets/9220412a-07b1-4059-9998-e524ceee5d11)
```python
cm=confusion_matrix(y_test,y_pred) 
print("Confusion Matrix:") 
print(cm)
```
![433054207-9260079c-af95-4cc2-aea0-6c4628f517f7](https://github.com/user-attachments/assets/f563277a-1c98-4689-af39-5655348f07fc)
```python
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```
![433054229-cd160cc5-9bd4-4b13-b738-bc22fafefb31](https://github.com/user-attachments/assets/80bb8ce7-d48b-4d4d-be81-d4c2b1a15f59)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
