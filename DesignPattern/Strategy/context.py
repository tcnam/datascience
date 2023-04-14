
class Context():  
    __strategy = None  
    def __init__(self, strategy):  
        self.__strategy = strategy  
        
    def fit(self, xTrain, yTrain):
        self.__strategy.fit(xTrain, yTrain)
        
    def predict(self, xTest):  
        return self.__strategy.predict(xTest)  
    
    def evaluate(self, xTest, yTrue):
        return self.__strategy.evaluate(xTest, yTrue)