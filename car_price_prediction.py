
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sklearn 
warnings.filterwarnings('ignore')

data = pd.read_csv('car_prediction_data.csv')

print(f'Number of rows: {data.shape[0]}\nNumber of column: {data.shape[1]}')

print(data['Fuel_Type'].unique())
print(data['Seller_Type'].unique())
print(data['Transmission'].unique())
print(data['Owner'].unique())

data.isnull().sum()

data.describe()

new_data = data.drop(['Car_Name'], axis=1)
new_data['Current_year'] = 2023
new_data.head()

new_data['number_of_year'] = new_data['Current_year'] - new_data['Year']
new_data.head()

new_data.drop(['Year'],axis=1,inplace=True)

new_data.head()

new_data = pd.get_dummies(new_data,drop_first=True)

new_data.head()

new_data = new_data.drop(['Current_year'], axis=1)
new_data.head()

import seaborn as sns
sns.pairplot(new_data.corr())

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

new_data.columns

X = new_data.iloc[:,1:]
y = new_data.iloc[:, 0]

print(X['Owner'].unique())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

X_train

X_test

# standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()

X_train= scaler.fit_transform(X_train)

X_test=scaler.fit_transform(X_test)

import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))

X_train

"""#### MODEL TRAINING"""

from sklearn.linear_model import LinearRegression
regression= LinearRegression()

regression.fit(X_train,y_train)

#print the coefficents and intercept
print(regression.coef_)

#on which parametr model is traind
regression.get_params()

# Prediction with Test data
reg_predict=regression.predict(X_test)

reg_predict

#plot a scatter 
plt.scatter(y_test,reg_predict)

#residuals
residuals=y_test-reg_predict

residuals



"""#### Model Selection: Random Forest Regressor"""

from sklearn.ensemble import RandomForestRegressor

cmodel = RandomForestRegressor()

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(5, 30, num = 6)],
               'min_samples_split': [2, 5, 10, 15, 100],
               'min_samples_leaf': [1, 2, 5, 10]}
print(random_grid)

from sklearn.model_selection import RandomizedSearchCV
# Use the random grid to search for best hyperparameters
model=RandomizedSearchCV(
    estimator = cmodel, 
    param_distributions = random_grid,
    scoring='neg_mean_squared_error', 
    n_iter = 10, cv = 5, verbose=2, 
    random_state=42, n_jobs = 1
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

data_frame = {
    'Actual_Value': y_test,
    'Prediction_Value':prediction
}

pred_vs_actu = pd.DataFrame(data_frame, columns=data_frame.keys())
pred_vs_actu.head()

plt.figure(figsize=(5,3))
sns.distplot(y_test-prediction)

from sklearn.metrics import r2_score
score =r2_score(y_test,reg_predict)
print(score)

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_metrics(y_test, prediction):
    print('MAE:', mean_absolute_error(y_test, prediction))
    print('MSE:', mean_squared_error(y_test, prediction))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, prediction)))

evaluate_metrics(y_test, prediction)

import pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickled_model= pickle.load(open('model.pkl','rb'))