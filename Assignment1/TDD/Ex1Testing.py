import numpy as np
import unittest
import Ex1Code

ListOfColors = np.array(["W", "K", "Y", "G", "R", "B"])
# White Black Yellow Green Red blue

class Tests(unittest.TestCase):
    def FillingUpTheCode(self):
        testcode = Ex1Code.CodeMaker()
        testcode.build_code()
        self.assertNotEqual(0, testcode[0])

    def TakeUserInput(self):
        testinput = Ex1Code.PlayTheGame().take_user_input()
        print(testinput)
        self.assertNotEqual(True, any(testinput==0))

    def TenRounds(self):
        a = Ex1Code.PlayTheGame(round=0)
        a.play_ten_rounds_old()
        self.assertNotEqual(0, a._round)

        # When playing, the round always increases, so it won't
        # be stuck on 1 where it's initialized.

    def WhichOnesPerfect(self):
        a = Ex1Code.PlayTheGame(round = 9)
        a.play_ten_rounds_old()
        ## Whatever is stored at the end, I want to check.
        # I get to see the code at the end, so i'll be able to see
        # how many of the guesses were perfect.
        perfect = 0
        for i in range(4):
            if a._code[i] == a._current_guess[i]:
                perfect += 1
        self.assertEqual(perfect, Ex1Code
                         .HowManyRight(guess = a._current_guess)
                         .how_many_perfect())

    def HowManyReds(self):
        # The method in question is redundant now. It's kept here, to show
        # that we did in fact do ten tests.
        self.assertEqual(4, Ex1Code.HowManyRight(code = 
                                                 np.array([5,5,5,5]))
                                                 .how_many_reds())

    def FindNumberOfEveryColor(self):
        # Testing with an array of White White Red Green.
        # For the red test, the output isn't even an array yet.
        # I can't test every color right now. I'll make that later.

        # This tests the Whites. About time they got tested lol
        self.assertEqual(2,
                         Ex1Code.HowManyRight(code = np.array([1, 1, 5, 4]))
                         .how_many_every_color_first())

    def GiveArrayOfEveryColorInstead(self):
        # Test input consists of Blue Black Green Green.
        # Testing the Blues, which should be the last entry of the 
        # output array.
        a = Ex1Code.HowManyRight().how_many_every_color(code =
                                                        np.array([6, 2,
                                                                  4, 4]))
        self.assertEqual(1, a[1])
        self.assertEqual(2, a[3])
        self.assertEqual(1, a[5])

    def ReturnAmountsInGuessProperly(self):
        # Test input is KKKK.
        # I'm testing if the 1th input of the array here == 4.
        a = Ex1Code.HowManyRight().how_many_every_color_guess(guess =
                                                              np.array([2,
                                                                        2,
                                                                        2,
                                                                        2]))
        self.assertEqual(4, a[1])
        self.assertEqual(0, a[5]) # Does not contain Blue at all.

    def ColorsRightPerGuess(self):
        # I will hard-change the guess fully a correct guess, and see what
        # happens.
        self.assertEqual(3, Ex1Code.HowManyRight(guess = 
                                                np.array([2, 3, 3, 5]),
                                                code = np.array([6, 5,
                                                                 3, 3]))
                                                .colors_guessed_correctly())

    def PlayAFullGame_first(self):
        # I will now incrementally build the method to play a full game.
        # I first want to be able to stop it when ten rounds are reached.
        # Let's try to at least _reach_ 10 rounds.
        self.assertEqual(10, Ex1Code.PlayTheGame().count_ten_rounds())

    def PlayAFullGame_second(self):
        # Only play one round. Returns the number of correct colors.
        # I made my guess just equal to the code, it should always be four.
        self.assertEqual(4, Ex1Code.PlayTheGame()
                         .able_to_work_with_how_many_every_color())
        
    def PlayAFullGame_third(self):
        # I want to both be able to see the perfects and the colors.
        # My code will be RYGB, my guess will be RWYW.
        # Should be one perfect, two colors.
        a, b = Ex1Code.PlayTheGame(guess = 
                                   np.array([5, 3, 4, 6]),
                                   code = 
                                   np.array([5, 1, 3, 1])
                                   ).returning_colors_and_perfects()
        
        self.assertEqual(2, a)
        self.assertEqual(1, b)


Ex1Code.PlayTheGame().play_ten_rounds()