import numpy as np
import unittest
import Ex1Code


class Tests(unittest.TestCase):
    def FillingUpTheCode(self):
        testcode = Ex1Code.CodeMaker()
        testcode.build_code()
        self.assertNotEqual(0,testcode._code[0])

    def TakeUserInput(self):
        testinput = Ex1Code.PlayTheGame().take_user_input()
        print(testinput)
        self.assertNotEqual(True, any(testinput==0))

    def TenRounds(self):
        a = Ex1Code.PlayTheGame(round=9)
        a.play_ten_rounds()
        self.assertEqual(10, a._round)

a = Tests()
a.TenRounds()

# I lost on purpose, and it correctly counted the amount of rounds played!
# or at least I started at round 9.