class Train:
    def __init__(self):
        self._meow = 5
        self.__gay = 30
        
    def get_gay(self):
        return self.__gay
    
        
train = Train()
print(train._meow)

try:
    print(train.__gay)
except Exception as e:
    print(f' exception {e} as its a private attribute, must use a getter')
  
    
print(train.get_gay())
