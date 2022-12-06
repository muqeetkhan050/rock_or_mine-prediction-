
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data=pd.read_csv('Copy of sonar data.csv',header=None)

data.head()
data.shape
data.describe()
data[60].unique()
data[60].value_counts()

data.groupby(60).mean()

X=data.drop(columns=60,axis=1)
y=data[60]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y,random_state=1)


logreg=LogisticRegression()

logreg.fit(X_train,y_train)

#model evaluation
#accuracy
from sklearn.metrics import accuracy_score

X_train_prediction=logreg.predict(X_train)
accuracy=accuracy_score(X_train_prediction,y_train)

print("accuracy:,", accuracy)
#0.83422....

X_test_prediction=logreg.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,y_test)

print(test_data_accuracy)

#0.7619047619047619

#making predictive system

input_data=(0.0090,0.0062,0.0253,0.0489,0.1197,0.1589,0.1392,0.0987,0.0955,0.1895,0.1896,0.2547,0.4073,0.2988,0.2901,0.5326,0.4022,0.1571,0.3024,0.3907,0.3542,0.4438,0.6414,0.4601,0.6009,0.8690,0.8345,0.7669,0.5081,0.4620,0.5380,0.5375,0.3844,0.3601,0.7402,0.7761,0.3858,0.0667,0.3684,0.6114,0.3510,0.2312,0.2195,0.3051,0.1937,0.1570,0.0479,0.0538,0.0146,0.0068,0.0187,0.0059,0.0095,0.0194,0.0080,0.0152,0.0158,0.0053,0.0189,0.0102
)

#changinf input data as numpy array
input_d=np.asarray(input_data)
input_reshape=input_d.reshape(1,-1)

prediction=logreg.predict(input_reshape)

print(prediction)

if(prediction[0]=='R'):
    print('object is rock')
else:
    print('object is mine')

#object is rock
























































































































































