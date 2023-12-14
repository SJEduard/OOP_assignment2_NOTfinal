import numpy as np

ListOfColors = np.array(["W","B","Y","G","R","b"])
# White Black Yellow Green Red blue


class CodeMaker:
    def __init__(self, round = 0, guess = np.array([0,0,0,0]), code = np.array([0,0,0,0])):
        self._colors = np.array([1,2,3,4,5,6])
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
              "of W for White, B for Black, Y for Yellow, "
              "G for Green, R for Red and b for Blue.")

    def take_user_input(self):
        self._current_guess = np.array([0,0,0,0])
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

    def play_ten_rounds(self):
        code = CodeMaker().build_code()
        PlayTheGame().do_it_right_please()
        for i in range(10):
            if self._round==10:
                print("Sorry, you did not win in 10 rounds!")
                return
            else:
                self._round += 1
                self._current_guess = PlayTheGame(guess = self._current_guess).take_user_input()
                self._code = code
                HowManyRight(guess = self._current_guess, code = code).how_many_perfect()
                if np.array_equal(self._current_guess, code):
                    print("You win!")
                    return
                print(code)


class HowManyRight(PlayTheGame):
    def how_many_perfect(self):
        perfect = 0
        return perfect