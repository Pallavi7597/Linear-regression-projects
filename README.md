# medical-insurance-linear
linear regression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
dataset= pd.read_csv('insurance.csv',header=0)
dataset.head(5)
dataset.info()
dataset.corr()
dataset['age'].corr(dataset['charges'],method="pearson")
sns.pairplot(dataset,kind='reg')
plt.show()
sns.heatmap(dataset.corr())
plt.show()
Y = dataset['charges']
X = dataset.drop('charges', axis = 1)
X = pd.get_dummies(X, drop_first = True)
X.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,Y,train_size=.7,test_size=.3,random_state=1)
from sklearn.linear_model import LinearRegression
lm= LinearRegression()
lm.fit(x_train, y_train)
y_pred = lm.predict(x_test)
error=y_test-y_pred
table=pd.DataFrame({'Actual':y_test,'predicted':y_pred,'Error':error})
table.head()
print('Intercept: ', lm.intercept_)
print('Coefficients: ', lm.coef_)
table_small = table.head(10)
table_small.plot(kind = 'bar', figsize = (10,6))
plt.show()
#Accuracy - RMSE

from sklearn.metrics import mean_squared_error
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred)))
