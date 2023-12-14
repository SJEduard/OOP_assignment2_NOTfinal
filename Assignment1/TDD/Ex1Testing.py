import numpy as np
import unittest
import Ex1Code


class Tests(unittest.TestCase):
    def FillingUpTheCode(self):
        testcode = Ex1Code.CodeMaker()
        testcode.BuildCode()
        self.assertNotEqual(0,testcode._code[0]) 

    def TakeUserInput(self):
        testinput = Ex1Code.CodeMaker().TakeUserInput() 
        print(testinput)
        self.assertNotEqual(True, any(testinput==0))

tester = Tests().TakeUserInput()

print(tester)