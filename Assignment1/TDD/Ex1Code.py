import numpy as np

class CodeMaker:
    def __init__(self):
        self._colors = np.array(["white","black","yellow","green","red","blue"])
        self._code = np.array(["","","",""]) # Initializing as empty strings