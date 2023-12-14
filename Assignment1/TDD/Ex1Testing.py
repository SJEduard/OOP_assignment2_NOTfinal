import numpy as np
import unittest
import Ex1Code


class Tests(unittest.TestCase):
    def FillingUpTheCode(self):
        testcode = Ex1Code.CodeMaker()
        testcode.BuildCode()
        self.assertNotEqual(0,testcode._code[0])  # By extension of this one working, so do the others.

    def TakeUserInput(self):
        testinput = Ex1Code.CodeMaker().TakeUserInput()# From words to numbers 1 through 6. Input will be "white white white white"
        self.assertEqual(1, testinput[0])

tester = Tests().TakeUserInput()

print(tester)