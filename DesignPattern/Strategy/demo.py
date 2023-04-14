from context import Context
from models import DTClassifier

if __name__=='__main__':
    ctx=Context(DTClassifier())
    ctx.evaluate(None, None)