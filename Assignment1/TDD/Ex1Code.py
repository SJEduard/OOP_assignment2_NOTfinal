import numpy as np

ListOfColors = np.array(["W", "K", "Y", "G", "R", "B"])
# List of possible inputs.
# White Black Yellow Green Red blue


class CodeMaker:
    def __init__(self, round = 0, guess = np.array([0, 0, 0, 0]), 
                 code = np.array([0, 0, 0, 0])):
        self._colors = np.array([1, 2, 3, 4, 5, 6])
        self._code = code
        self._current_guess = guess
        self._round = round

    def build_code(self):
        for i in range(4):
            self._code[i] = np.random.choice(self._colors)
        return self._code


class PlayTheGame(CodeMaker):
    def do_it_right_please(self):
        print("Guess the code in 10 rounds.\n"
              "Give input containing no spaces, consisting "
              "of W for White, K for Black, Y for Yellow, "
              "G for Green, R for Red and B for Blue.")

    def take_user_input(self):
        self._current_guess = np.array([0, 0, 0, 0])
        user_input = input()
        for i in range(6):
            if user_input[0] == ListOfColors[i]:
                self._current_guess[0] = i+1
            if user_input[1] == ListOfColors[i]:
                self._current_guess[1] = i+1
            if user_input[2] == ListOfColors[i]:
                self._current_guess[2] = i+1
            if user_input[3] == ListOfColors[i]:
                self._current_guess[3] = i+1
        if any(self._current_guess == 0):
            print("Invalid Input! Game Over! See the instructions "
                  "at the start.")
            raise ValueError("The Input was invalid.")
        return self._current_guess

    def play_ten_rounds_old(self):
        ''' This method is redundant. It was useful for
            testing TDD. It will not be called.'''
        code = CodeMaker().build_code()
        PlayTheGame().do_it_right_please()
        for i in range(10):
            if self._round==10:
                print("Sorry, you did not win in 10 rounds!")
                return
            else:
                self._round += 1
                self._current_guess = PlayTheGame(
                    guess = self._current_guess).take_user_input()
                self._code = code
                HowManyRight(guess = self._current_guess,
                             code = code).how_many_perfect()
                if np.array_equal(self._current_guess, code):
                    print("You win!")
                    return
                print(code)

    def play_ten_rounds(self):
        return


class HowManyRight(PlayTheGame):
    def how_many_perfect(self):
        perfect = 0
        for i in range(4):
            if self._code[i] == self._current_guess[i]:
                perfect += 1
        return perfect

    def how_many_reds(self):
        ''' This method is redundant. It was useful as part of the testing
            with TDD. It will not be called.'''
        self._code = np.array([5,5,5,5])  # RRRR
        howmany = 0
        for i in range(4):
            if self._code[i] == 5:
                howmany += 1
        return howmany

    def how_many_every_color_first(self):
        ''' This method is redundant. It was useful as part of the testing
            with TDD. It will not be called.'''
        self._code = np.array([1,1,5,4])  # WWRG
        howmany = np.array([0,0,0,0,0,0])
        for i in range(4):
            if self._code[i] == 1:
                howmany[0] += 1
            if self._code[i] == 2:
                howmany[1] += 1
            if self._code[i] == 3:
                howmany[2] += 1
            if self._code[i] == 4:
                howmany[3] += 1
            if self._code[i] == 5:
                howmany[4] += 1
            if self._code[i] == 6:
                howmany[5] += 1
        return howmany[0]

    def how_many_every_color(self, code = np.array([0, 0, 0, 0])):
        code_hist = np.array([0,0,0,0,0,0])
        for i in range(4):
            if code[i] == 1:
                code_hist[0] += 1
            if code[i] == 2:
                code_hist[1] += 1
            if code[i] == 3:
                code_hist[2] += 1
            if code[i] == 4:
                code_hist[3] += 1
            if code[i] == 5:
                code_hist[4] += 1
            if code[i] == 6:
                code_hist[5] += 1
        return code_hist

    def how_many_every_color_guess(self, guess = np.array([0, 0, 0, 0])):
        return HowManyRight().how_many_every_color(guess)

    def colors_guessed_correctly(self):
        correctcolors = 0
        hist_guess = HowManyRight().how_many_every_color(self._current_guess)
        hist_code = HowManyRight().how_many_every_color(self._code)
        for i in range(6):
            color_guessed = hist_guess[i]
            color_in_code = hist_code[i]
            while (color_guessed > 0) and (color_in_code) > 0:
                correctcolors += 1
                color_guessed -= 1
                color_in_code -= 1
        return correctcolors
