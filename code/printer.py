class Printer:    
    def __init__(self) -> None:
        pass
      
    @staticmethod
    def print(string, file="log.txt") -> None:
        print(str(string))
        with open(file, 'a') as f:
            f.write(str(string))
            f.write('\n')
        
      