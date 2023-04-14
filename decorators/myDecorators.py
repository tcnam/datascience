from functools import wraps

def myTimer(origFunc):
    import time   
    @wraps(origFunc)
    def wrapper(*args, **kwargs):
        start=time.time()
        result=origFunc(*args, **kwargs)
        end=time.time()
        period=end-start
        print(f'{origFunc.__name__} ran in: {period} sec')
        return result
    
    return wrapper

def myRetry(maxTries=3, delaySeconds=1):
    import time
    def decoratorRetry(origFunc):
        @wraps(origFunc)
        def wrapper(*args, **kwargs):
            tries=0
            while tries<maxTries:
                try:
                    return origFunc(*args, **kwargs)
                except Exception as e:
                    tries+=1
                    if tries ==maxTries:
                        raise e
                    time.sleep(delaySeconds)
        return wrapper
    return decoratorRetry

def myLog(origFunc):
    import logging
    logging.basicConfig(level=logging.INFO)
    @wraps(origFunc)
    def wrapper(*args, **kwargs):
        logging.info(f'Executing {origFunc.__name__}')
        result=origFunc(*args, **kwargs)
        logging.info(f'Finished executing {origFunc.__name__}')
        return result
    return wrapper

