import numpy as np
import pandas as pd

import scipy as sp
from scipy.stats import skew, norm
from scipy.optimize import minimize

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error

%matplotlib inline

data = pd.read_csv('households_data.csv')
data.head()

#dropping house numbers not required for analysis
data = data.drop(['hh'], axis=1)

data.info()

#check NAs
data.isna().sum()

#plot connections , this is the pivot for the entire analysis

data.boxplot('connections', rot = 30, figsize=(5,6))
connections = data['connections']
sns.distplot(connections)

# Perform log transformation
sns.distplot(np.log1p(connections))

#check skewness of connections variables

skew(np.log1p(connections))

#since it's skewed we apply log trasnformations

# Apply log transformation
data['connections'] = np.log1p(connections)
data = pd.get_dummies(data, drop_first=True)

# Create interaction terms between independent variables 
terms = data.drop(['connections','loan'], axis=1)


#fit polynomial features

poly = preprocessing.PolynomialFeatures(3, interaction_only=False, include_bias=False)
data_array = poly.fit_transform(terms)

target_feature_names = ['_x_'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(terms.columns,p) for p in poly.powers_]]
data_df = pd.DataFrame(data_array, columns = target_feature_names)

data_df.head()

#then use lasso to predict degree with above interactions
lasso = Lasso(alpha = 0.1)
model = lasso.fit(data_df, data['connections'])


model.score(data_df, data['connections'])
# Very low R^2 score indicating that we can't predict from other interaction terms and this is good. 


# Look at the remaining terms(coef is not 0)
table = pd.DataFrame({'columns':data_df.columns, 'coef':model.coef_})
left_variables = table[table['coef']!=0]
left_variables





# Create dhat and d in the data frame
dhat = model.predict(data_df)
data_df['d']= data['connections']
data_df['dhat'] = dhat

data_df.columns.get_loc('dhat')

loan = data['loan']


def lasso(x):  #define lasso function first
    return (1. / (2 * data_df.shape[0])) * np.square(np.linalg.norm(data_df.dot(x) - loan, 2)) + 0.1 * np.linalg.norm(weights*x, 1)
    
weights = np.ones(data_df.shape[1])# set weights
#set weight of dhat to 0
weights [969] = 0
weights


x0 = np.zeros(data_df.shape[1])
res = minimize(lasso, x0, method='L-BFGS-B', options={'disp': False})


coef_df = pd.DataFrame({'columns':data_df.columns, 'coef':res.x})
coef_df

#Mean squared error

y_pred = data_df.dot(res.x)
mse = np.mean((y_pred - data['loan'])**2) 
print(mse)


#prediction treamtmet
pred_df = pd.DataFrame({'Treatment': dhat, 'Prediction':y_pred, 'Original': data['loan']})
pred_df



plt.scatter(pred_df.Treatment, pred_df.Prediction)
plt.scatter(pred_df.Treatment, pred_df.Original)
plt.xlabel('Treatment')
plt.ylabel('Outcome')

#Implementing Double lLASSO to figure oiut the pure treatment effect

loan = data['loan']
data_df_1 = data_df.drop(['dhat'], axis=1)


# Model naive lasso without dhat
logit = LogisticRegression(penalty='l1', solver='liblinear').fit(data_df_1, loan)
y_pred1 = logit.predict(data_df_1)
print("Accuracy: %.3f" % logit.score(data_df_1 , loan))
mse = mean_squared_error(y_pred1, loan) 
print("MSE: %.3f" % mse)


# Model naive lasso with dhat
logit = LogisticRegression(penalty='l1', solver='liblinear').fit(data_df, loan)
y_pred2 = logit.predict(data_df)
print("Accuracy: %.3f" % logit.score(data_df, loan))
mse = mean_squared_error(y_pred2, loan) 
print("MSE: %.3f" % mse)


# Model lasso cv without dhat
logit_cv = LogisticRegressionCV(penalty='l1', solver='liblinear', cv = 10, max_iter = 10000).fit(data_df_1, loan)
y_pred3 = logit_cv.predict(data_df_1)
print("Accuracy: %.3f" % logit_cv.score(data_df_1, loan))
mse = mean_squared_error(y_pred3, loan) 
print("MSE: %.3f" % mse)



# Model lasso cv with dhat
logit_cv = LogisticRegressionCV(penalty='l1', solver='liblinear', cv = 10, max_iter = 10000).fit(data_df, loan)
y_pred4 = logit_cv.predict(data_df)
print("Accuracy: %.3f" % logit_cv.score(data_df, loan))
mse = mean_squared_error(y_pred4, loan) 
print("MSE: %.3f" % mse)


#Bootstrapping

y_pred = y_pred.ravel()
# Mean
y_pred.mean()


# Standard deviation
y_pred.std()


# Standard Error
np.std(y_pred)/(len(y_pred) ** 0.5)


# Construct the simulated sampling distribution
sample_props = []
for _ in range(1000):
    sample = np.random.choice(y_pred, size = len(y_pred))
    sample_props.append(sample.mean())
    
# The simulated mean of the sampling distribution
np.mean(sample_props)

# The simulated standard deviation of the sampling distribution
np.std(sample_props)

# The simulated standard error of the sampling distribution
np.std(sample_props)/(len(y_pred) ** 0.5)


# Plot the simulated sampling distribution
plt.hist(sample_props)

# The theorical mean and simulated mean
(y_pred.mean(), simulated_mean)

# The theorical standard deviation and simulated standard deviation
(y_pred.std(), simulated_std)


# The theorical standard error and simulated standard error
(np.std(y_pred)/(len(y_pred) ** 0.5), np.std(sample_props)/(len(y_pred) ** 0.5))




