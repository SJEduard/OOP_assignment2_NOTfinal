import numpy as np
import unittest
import sys
import Ex1Code


class Tests(unittest.TestCase):
    def FillingUpTheCode(self):
        testcode = Ex1Code.CodeMaker()
        testcode.BuildCode()
        self.assertNotEqual(0,testcode._code[0])

tester = Tests()
tester.FillingUpTheCode()

print(tester)