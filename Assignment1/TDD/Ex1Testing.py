import numpy as np
import unittest
import Ex1Code

ListOfColors = np.array(["W", "K", "Y", "G", "R", "B"])
# White Black Yellow Green Red blue

class Tests(unittest.TestCase):
    def FillingUpTheCode(self):
        '''
        This method aims to confirm that the CodeMaker can
        indeed fill up the code. Since the default is set to
        [0, 0, 0, 0], if the first entry is no longer a zero,
        we can confirm that it is indeed filled up.
        '''
        testcode = Ex1Code.CodeMaker()
        testcode.build_code()
        self.assertNotEqual(0, testcode[0])

    def TakeUserInput(self):
        '''
        This method aims to confirm that during PlayTheGame,
        user input can be collected. The default again is
        [0, 0, 0, 0], so if this is no longer the case,
        we can confirm that it is indeed filled up.
        '''
        testinput = Ex1Code.PlayTheGame().take_user_input()
        print(testinput)
        self.assertNotEqual(True, any(testinput==0))

    def TenRounds(self):
        '''
        This method aims to confirm that we are able to count
        the rounds properly. It is initialized to zero, so if
        it is no longer equal to zero at the end of the game,
        we will know that it has updated properly.
        '''
        a = Ex1Code.PlayTheGame(round=0)
        a.play_ten_rounds_old()
        self.assertNotEqual(0, a._round)

    def WhichOnesPerfect(self):
        '''
        We start the game at round 9, and play ten rounds.
        This method aims to confirm that the code for how
        many perfect guesses were made, works properly.
        The one chance at a guess that we get here has
        however many correct, and if this is equal to the
        output of the how_many_perfect method, we're happy.
        '''
        a = Ex1Code.PlayTheGame(round = 9)
        a.play_ten_rounds_old()
        perfect = 0
        for i in range(4):
            if a._code[i] == a._current_guess[i]:
                perfect += 1
        self.assertEqual(perfect, Ex1Code
                         .HowManyRight(guess = a._current_guess)
                         .how_many_perfect())

    def HowManyReds(self):
        '''
        This method will be reused and expanded.
        It aims to check the code, to find how many reds it contains.
        The code is now hardcoded to contain four reds.
        '''
        self.assertEqual(4, Ex1Code.HowManyRight(code = 
                                                 np.array([5,5,5,5]))
                                                 .how_many_reds())

    def FindNumberOfEveryColor(self):
        '''
        This method aims to expand upon the how_many_reds method.
        It aims to be able to extract how many entries of every
        color some code contains.
        The test code is set to be White White Red Green,
        the Whites are tested - about time -, and we expect
        the method to return 2.
        '''
        self.assertEqual(2,
                         Ex1Code.HowManyRight(code = np.array([1, 1, 5, 4]))
                         .how_many_every_color_first())

    def GiveArrayOfEveryColorInstead(self):
        '''
        This method aims to expand upon the how_many_every_color_first method.
        Instead of returning simply the White for testing purposes, it now
        returns a histogram for the colors of whatever code is put into it.
        '''
        a = Ex1Code.HowManyRight().how_many_every_color(code =
                                                        np.array([6, 2,
                                                                  4, 4]))
        self.assertEqual(1, a[1])
        self.assertEqual(2, a[3])
        self.assertEqual(1, a[5])

    def ReturnAmountsInGuessProperly(self):
        '''
        This method aims to confirm that not just for a code, but also for a
        guess, a histogram of the colors is returned properly.
        The guess is simply passed to the same histogram.
        '''
        a = Ex1Code.HowManyRight().how_many_every_color_guess(guess =
                                                              np.array([2,
                                                                        2,
                                                                        2,
                                                                        2]))
        self.assertEqual(4, a[1])
        self.assertEqual(0, a[5]) # Does not contain Blue at all.

    def ColorsRightPerGuess(self):
        '''
        This method aims to check whether the colors_guessed_correctly method
        works.
        A guess with 3 right colors is passed, and this should give us 3 right
        colors.
        '''
        self.assertEqual(3, Ex1Code.HowManyRight(guess = 
                                                np.array([2, 3, 3, 5]),
                                                code = np.array([6, 5,
                                                                 3, 3]))
                                                .colors_guessed_correctly())

    def PlayAFullGame_first(self):
        '''
        This method is the first in a series to confirm some of the steps of
        playing a full game.
        Let's start by playing for exactly 10 rounds. Later, this can be set
        to user input.
        '''
        self.assertEqual(10, Ex1Code.PlayTheGame().count_ten_rounds())

    def PlayAFullGame_second(self):
        '''
        This method aims to confirm that, while playing, the how_many_-
        every_color method works. We just make the code and the guess
        equal, and expect four correct colors.
        '''
        self.assertEqual(4, Ex1Code.PlayTheGame()
                         .able_to_work_with_how_many_every_color())

    def PlayAFullGame_third(self):
        '''
        This method aims to both pass the perfect guesses and right colors,
        to make sure both can be used throughout the rounds.
        '''
        a, b = Ex1Code.PlayTheGame(guess = 
                                   np.array([5, 3, 4, 6]),
                                   code = 
                                   np.array([5, 1, 3, 1])
                                   ).returning_colors_and_perfects()

        self.assertEqual(2, a)
        self.assertEqual(1, b)


Ex1Code.PlayTheGame().play_the_game()