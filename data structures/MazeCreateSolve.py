import random
import matplotlib.pyplot as plt
from collections import deque

def generate_maze(rows, cols):
    """
    Generate a 2D maze where 0 represents a wall and 1 represents a path.
    Ensures there is at least one valid path from the top-left to the bottom-right.
    """
    maze = [[0 for _ in range(cols)] for _ in range(rows)]

    # Create a random maze with paths
    for i in range(rows):
        for j in range(cols):
            maze[i][j] = random.choice([0, 1])

    # Ensure there's a guaranteed path using a DFS-like approach
    def ensure_path(x, y):
        if x == rows - 1 and y == cols - 1:
            maze[x][y] = 1
            return True
        if x < 0 or x >= rows or y < 0 or y >= cols or maze[x][y] == 1:
            return False

        maze[x][y] = 1
        if ensure_path(x + 1, y) or ensure_path(x, y + 1):
            return True

        return False

    ensure_path(0, 0)
    return maze


def find_shortest_path(maze):
    """
    Find the shortest path from the top-left corner to the bottom-right corner
    using Breadth-First Search (BFS).
    Returns the path as a list of (row, col) tuples.
    """
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    # BFS initialization
    queue = deque([(0, 0)])
    visited = set()
    visited.add((0, 0))
    parent = {}  # To reconstruct the path

    while queue:
        x, y = queue.popleft()

        # If we reached the bottom-right corner
        if (x, y) == (rows - 1, cols - 1):
            path = []
            while (x, y) in parent:
                path.append((x, y))
                x, y = parent[(x, y)]
            path.append((0, 0))
            return path[::-1]

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and maze[nx][ny] == 1:
                queue.append((nx, ny))
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)

    return None  # No path found


def visualize_maze(maze, path=None):
    """
    Visualize the maze and optionally overlay the path.
    """
    rows, cols = len(maze), len(maze[0])
    fig, ax = plt.subplots(figsize=(cols, rows))

    # Draw the maze
    for i in range(rows):
        for j in range(cols):
            color = "black" if maze[i][j] == 0 else "white"
            rect = plt.Rectangle((j, rows - i - 1), 1, 1, facecolor=color)
            ax.add_patch(rect)

    # Draw the path if it exists
    if path:
        for (x, y) in path:
            rect = plt.Rectangle((y, rows - x - 1), 1, 1, facecolor="green", alpha=0.5)
            ax.add_patch(rect)

    # Grid settings
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which="both", color="black", linewidth=0.5)
    ax.set_aspect("equal")

    plt.show()


# Example usage
if __name__ == "__main__":
    rows, cols = 15, 15  # Maze dimensions
    maze = generate_maze(rows, cols)
    print("Generated Maze:")
    for row in maze:
        print(row)

    path = find_shortest_path(maze)
    if path:
        print("\nShortest Path:")
        print(path)
    else:
        print("\nNo path found!")

    visualize_maze(maze, path)