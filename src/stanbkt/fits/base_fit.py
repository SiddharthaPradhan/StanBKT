from abc import ABC, abstractmethod



class BaseFit(ABC):
    def __init__(self):
        super().__init__()
        self.fit_name = self.__class__.__name__
    
    def test():
        pass
    
    def summary():
        raise NotImplemented("This function should be implmented!")
    
    
    