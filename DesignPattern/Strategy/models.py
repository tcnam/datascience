from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from strategy import Strategy

class DTClassifier(Strategy):
    model=DecisionTreeClassifier()

    def fit(self, xTrain, yTrain):
        self.model.fit(xTrain, yTrain)

    def predict(self, xTEst):
        return self.model.predict(xTEst)

class Svc(Strategy):
    model=SVC()

    def fit(self, xTrain, yTrain):
        self.model.fit(xTrain, yTrain)

    def predict(self, xTEst):
        return self.model.predict(xTEst)

class LogRegression(Strategy):
    model=LogisticRegression()

    def fit(self, xTrain, yTrain):
        self.model.fit(xTrain, yTrain)

    def predict(self, xTEst):
        return self.model.predict(xTEst)

class Gauss(Strategy):
    model=GaussianNB()

    def fit(self, xTrain, yTrain):
        self.model.fit(xTrain, yTrain)

    def predict(self, xTEst):
        return self.model.predict(xTEst)

class RFClassifier(Strategy):
    model=RandomForestClassifier()

    def fit(self, xTrain, yTrain):
        self.model.fit(xTrain, yTrain)

    def predict(self, xTEst):
        return self.model.predict(xTEst)