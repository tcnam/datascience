# %%
import pandas as pd
import numpy as np
import pathlib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report  

# %%
def getParentFolder() -> str:
    return pathlib.Path(__file__).parent.resolve()

# %%
def detect_outliers(train, n, features):
   
    outlier_indices = [] 
    for col in features: 
        Q1 = np.percentile(train[col], 25)
        Q3 = np.percentile(train[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR 
        outlier_list_col = train[(train[col] < Q1 - outlier_step) | (train[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col) 
    multiple_outliers = list(key for key, value in enumerate(outlier_indices) if value > n)
 
    return multiple_outliers

# %%
def preprocessTrainData(inputFile, outoutFile):
    df=pd.read_csv(f'{getParentFolder()}\{inputFile}',sep=',')

    labelEncoder=LabelEncoder()
    oneHotEncoder=OneHotEncoder()
    # for collumn in ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']:
    for collumn in ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']:
        if collumn=='gender' or collumn =='Partner' or collumn=='Dependents' or collumn=='PhoneService' or collumn=='Contract' or collumn=='Churn':
        # if collumn=='gender' or collumn =='Partner' or collumn=='Dependents' or collumn=='PhoneService' or collumn=='Contract':
            df[collumn]=labelEncoder.fit_transform(df[collumn])
        else:
            featureArray=oneHotEncoder.fit_transform(df[[collumn]]).toarray()
            featureLabels=[collumn + ' '+ x for x in oneHotEncoder.categories_]
            feature=pd.DataFrame(featureArray,columns=featureLabels)
            df=df.drop(columns=[f'{collumn}'],axis=1)
            df=pd.concat([df,feature], axis=1)           
    df=df.drop(columns=['TotalCharges'],axis=1)
    # df=df.drop(columns=['TotalCharges'],axis=1)

    outliers_to_drop = detect_outliers(df, 2, ['tenure', 'MonthlyCharges'])
    print("We will drop these {} indices: ".format(len(outliers_to_drop)), outliers_to_drop)

    print("Before: {} rows".format(len(df)))
    df= df.drop(outliers_to_drop, axis = 0).reset_index(drop = True)
    print("After: {} rows".format(len(df)))

    df.to_csv(f'{getParentFolder()}\{outoutFile}',sep=',', encoding='utf-8',index=False)
    # print(df)

# %%
def trainModel(inputFile, validateInputFile):
    df=pd.read_csv(f'{getParentFolder()}\{inputFile}',sep=',')
    y=df['Churn']
    x=df.drop(columns=['customerID','Churn'],axis=1)

    dfResult=pd.read_csv(f'{getParentFolder()}\{validateInputFile}',sep=',')
    xValidate=dfResult.drop(columns=['customerID'],axis=1)

    xTrain, xTest, yTrain, yTest=train_test_split(x,y,test_size=0.3,random_state=101)

    sm = SMOTE(random_state = 2)  
    # print(f"Before Over Sampling, count of the label '1': {sum(xTrain == 1)}")  
    # print(f"Before Over Sampling, count of the label '0': {sum(yTrain == 0)} \n")  
    xTrainRes, yTrainRes = sm.fit_resample(xTrain, yTrain) 
    xTestRes, yTestRes = sm.fit_resample(xTest, yTest) 

    print(f'After Over Sampling, the shape of the train_X: {xTrainRes.shape}')  
    print(f'After Over Sampling, the shape of the train_y: {yTrainRes.shape} \n')  
    print(f"After Over Sampling, count of the label '1': {sum(yTrainRes == 1)}")  
    print(f"After Over Sampling, count of the label '0': {sum(yTrainRes == 0)}") 

    logmodel=LogisticRegression(solver='lbfgs', max_iter=1000)
    logmodel.fit(xTrainRes, yTrainRes)
    predictionsTrain=logmodel.predict(xTrainRes)
    predictionsTest=logmodel.predict(xTestRes)
    predictionsValidate=logmodel.predict(xValidate)
    prediction1=logmodel.predict(x)

    print(classification_report(yTrainRes,predictionsTrain))
    print(classification_report(yTestRes,predictionsTest))
    print(logmodel.score(xTrain,yTrain))
    print(logmodel.score(xTest,yTest))

    dfResult=pd.concat([dfResult,pd.DataFrame(predictionsValidate,columns=['Churn'])],axis=1)
    dfResult['Churn']=dfResult['Churn'].map({1:'Yes',0:'No'})
    dfResult.to_csv(f'{getParentFolder()}\\result.csv',sep=',', encoding='utf-8',index=False)

    df=pd.concat([df,pd.DataFrame(prediction1,columns=['churn_predict'])],axis=1)
    df.to_csv(f'{getParentFolder()}\\result1.csv',sep=',', encoding='utf-8',index=False)
# %%
if __name__=='__main__':
    preprocessTrainData('trainset.csv','preTrainset.csv')
    # preprocessTrainData('testset.csv','preTestset.csv')
    trainModel('preTrainset.csv','preTestset.csv')


# %%
