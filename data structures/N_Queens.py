import matplotlib.pyplot as plt
import numpy as np

def solve_n_queens(n):
    def is_safe(board, row, col):
        # Check for queens in the same column
        for i in range(row):
            if board[i][col] == "Q":
                return False

        # Check for queens in the upper left diagonal
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == "Q":
                return False

        # Check for queens in the upper right diagonal
        for i, j in zip(range(row, -1, -1), range(col, n)):
            if board[i][j] == "Q":
                return False

        return True

    def solve(row):
        if row == n:
            solution = ["".join(row) for row in board]
            solutions.append(solution)
            return

        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = "Q"
                solve(row + 1)
                board[row][col] = "."

    board = [["." for _ in range(n)] for _ in range(n)]
    solutions = []
    solve(0)
    return solutions


def display_solutions(solutions):
    def draw_chessboard(solution):
        n = len(solution)
        board = np.zeros((n, n))
        for i, row in enumerate(solution):
            for j, cell in enumerate(row):
                if cell == "Q":
                    board[i, j] = 1

        fig, ax = plt.subplots()
        ax.matshow(board, cmap="binary")
        for i in range(n):
            for j in range(n):
                if board[i, j] == 1:
                    ax.text(j, i, "Q", ha="center", va="center", color="red", fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    for solution in solutions:
        draw_chessboard(solution)


def main():
    n = int(input("Enter the value of n for the N-Queens problem: "))
    solutions = solve_n_queens(n)

    # Console output
    print(f"Total solutions for {n}-Queens: {len(solutions)}")
    print(solutions)

    # Visual output
    display_solutions(solutions)


if __name__ == "__main__":
    main()