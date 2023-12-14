import numpy as np

ListOfColors = np.array(["white","black","yellow","green","red","blue"])

class CodeMaker:
    def __init__(self):
        self._colors = np.array([1,2,3,4,5,6])
        self._code = np.array([0,0,0,0]) # Initializing as empty strings

    def BuildCode(self):
        for i in range(4):
            self._code[i] = np.random.choice(self._colors)