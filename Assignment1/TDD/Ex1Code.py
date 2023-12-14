import numpy as np

ListOfColors = np.array(["W","B","Y","G","R","b"]) 
# White Black Yellow Green Red blue

class CodeMaker:
    def __init__(self):
        self._colors = np.array([1,2,3,4,5,6])
        self._code = np.array([0,0,0,0]) # Initializing as empty strings
        self._current_guess = np.array([0,0,0,0])

    def BuildCode(self):
        for i in range(4):
            self._code[i] = np.random.choice(self._colors)

    def TakeUserInput(self):
        self._current_guess = np.array([0,0,0,0])
        print("Guess the code.\n"
              "Give input containing no spaces, consisting "
              "of W for White, B for Black, Y for Yellow, "
              "G for Green, R for Red and b for Blue.")
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
        if any(self._current_guess) == 0:
            raise ValueError("Invalid input!"
                             "Give input containing no spaces, consisting "
                             "of W for White, B for Black, Y for Yellow, "
                             "G for Green, R for Red and b for Blue.")
        return self._current_guess
