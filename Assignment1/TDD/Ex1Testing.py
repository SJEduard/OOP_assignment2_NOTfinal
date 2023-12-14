import numpy as np
import unittest
import sys
import Ex1Code


class Tests(unittest.TestCase):
    def FillingUpTheCode(self):
        self.assertNotEqual("",Ex1Code.CodeMaker()._code[0])
        self.assertNotEqual("",Ex1Code.CodeMaker()._code[1])
        self.assertNotEqual("",Ex1Code.CodeMaker()._code[2])
        self.assertNotEqual("",Ex1Code.CodeMaker()._code[3])


tester = Tests()
print(tester.FillingUpTheCode())