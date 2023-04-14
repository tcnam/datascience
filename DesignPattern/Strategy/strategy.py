from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, accuracy_score

class Strategy(ABC):
    @property
    def model(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self, xTrain, yTrain):
        pass

    @abstractmethod
    def predict(self, xTest):
        pass

    def evaluate(self, xTest, yTrue):
        print(f'Metrics for evaluating a regression model:')
        print(f'Mean Absolute Error = {mean_absolute_error(yTrue, self.predict(xTest))}')
        print(f'Mean Squared Error = {mean_squared_error(yTrue, self.predict(xTest))}')
        print(f'R2 Score = {r2_score(yTrue, self.predict(xTest))}')
        print(f'Metrics for evaluating a classification model:')
        print(f'Precision: {precision_score(yTrue, self.predict(xTest))}')
        print(f'Recall: {recall_score(yTrue, self.predict(xTest))}')
        print(f'F1: {f1_score(yTrue, self.predict(xTest))}')
        print(f'Accuracy: {accuracy_score(yTrue, self.predict(xTest))}')