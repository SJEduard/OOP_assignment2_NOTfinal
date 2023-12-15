import numpy as np
import unittest
import Ex1Code

ListOfColors = np.array(["W","B","Y","G","R","b"])
# White Black Yellow Green Red blue

class Tests(unittest.TestCase):
    def FillingUpTheCode(self):
        testcode = Ex1Code.CodeMaker()
        testcode.build_code()
        self.assertNotEqual(0,testcode[0])

    def TakeUserInput(self):
        testinput = Ex1Code.PlayTheGame().take_user_input()
        print(testinput)
        self.assertNotEqual(True, any(testinput==0))

    def TenRounds(self):
        a = Ex1Code.PlayTheGame(round=0)
        a.play_ten_rounds()
        self.assertNotEqual(0, a._round)

        # When playing, the round always increases, so it won't
        # be stuck on 1 where it's initialized.

    def WhichOnesPerfect(self):
        a = Ex1Code.PlayTheGame(round = 9)
        a.play_ten_rounds()
        ## Whatever is stored at the end, I want to check.
        # I get to see the code at the end, so i'll be able to see
        # how many of the guesses were perfect.
        perfect = 0
        for i in range(4):
            if a._code[i] == a._current_guess[i]:
                perfect += 1
        self.assertEqual(perfect, Ex1Code.HowManyRight().how_many_perfect())

    def HowManyReds(self):
        a = Ex1Code.HowManyRight(round = 9)
        
        self.assertEqual(4, Ex1Code.HowManyRight().how_many_right_color())

a = Tests()
a.HowManyReds()

print(a)

