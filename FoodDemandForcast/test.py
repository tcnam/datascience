from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import pandas as pd

# generate dataset
X, y = make_regression(n_samples=100, n_features=50, n_informative=10)
#assign column names 
col_list = ['col_' + str(x) for x in range(0,50)]
#create a dataframe table
df = pd.DataFrame(X, columns=col_list)

#feature selection using f_regression 
fs = SelectKBest(score_func=f_regression, k=5)
fit = fs.fit(X,y)

#create df for scores
dfscores = pd.DataFrame(fit.scores_)
#create df for column names
dfcolumns = pd.DataFrame(df.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['Selected_columns','Score_pearsons'] 

#print 10 best features
print(featureScores.nlargest(5,'Score_pearsons'))  