#Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# to suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# display all columns of the dataframe
pd.options.display.max_columns = None

# display all rows of the dataframe
pd.options.display.max_rows = None
 
# to display the float values upto 6 decimal places     
pd.options.display.float_format = '{:.6f}'.format
 
#Importing Train-Test split for validation
from sklearn.model_selection import train_test_split

# to perform Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# import various functions from sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# import function to perform feature selection
from sklearn.feature_selection import RFE
#Loading the dataset
df=pd.read_csv('Heart_Disease_Prediction.csv')

#Changing the data types of the columns

df['Sex']=df['Sex'].astype('category')
df['Chest pain type']=df['Chest pain type'].astype('category')
df['FBS over 120']=df['FBS over 120'].astype('category')
df['EKG results']=df['EKG results'].astype('category')
df['Exercise angina']=df['Exercise angina'].astype('category')
df['Slope of ST']=df['Slope of ST'].astype('category')
df['Number of vessels fluro']=df['Number of vessels fluro'].astype('category')
df['Thallium']=df['Thallium'].astype('category')
df['Heart Disease']=df['Heart Disease'].astype('category')
#Since Age feature doesn't require any outlier analysis,changing the type of variable for instance to skip the outlier analysis.
df['Age']=df['Age'].astype('category')

#Outlier Analysis
df_new = df[
    ~((df.select_dtypes(include='number') < (df.select_dtypes(include='number').quantile(0.25) - 1.5 * (df.select_dtypes(include='number').quantile(0.75) - df.select_dtypes(include='number').quantile(0.25)))) |
      (df.select_dtypes(include='number') > (df.select_dtypes(include='number').quantile(0.75) + 1.5 * (df.select_dtypes(include='number').quantile(0.75) - df.select_dtypes(include='number').quantile(0.25))))
    ).any(axis=1)
]
#Chaning the Age Feature back to its original type.
df['Age']=df['Age'].astype('int')

Cat_columns=df.select_dtypes(include='category').columns
Num_columns=df.select_dtypes(include='int').columns
#Removing the target variable from the categorical columns
Cat_columns=Cat_columns.drop('Heart Disease')
#Encoding the categorical variables
Encoded=pd.get_dummies(df_new[Cat_columns], drop_first=True)
df_encoded=pd.concat([df_new[Num_columns], Encoded], axis=1)

df_new['Heart Disease']=df_new['Heart Disease'].replace({'Presence':1, 'Absence':0})
df_encoded['Heart Disease']=df_new['Heart Disease']

#Splitting the data into train and test
X = df_encoded.drop('Heart Disease', axis=1)
y = df_encoded['Heart Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape) 

#Scaling the data
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
Xtrain=pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = scaler.transform(X_test)
Xtest=pd.DataFrame(X_test_scaled, columns=X_test.columns)

#Logistic Regression
Logreg=LogisticRegression()
Logreg.fit(Xtrain, y_train)

print(classification_report(y_test, Logreg.predict(Xtest)))
print(confusion_matrix(y_test, Logreg.predict(Xtest)))
print(roc_auc_score(y_test, Logreg.predict(Xtest)))
