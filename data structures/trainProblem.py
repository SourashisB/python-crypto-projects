import random
import matplotlib.pyplot as plt

# Create a graph representing the train map as an adjacency list
def create_train_map():
    stations = [f"Station {i}" for i in range(1, 11)]
    adjacency_list = {
        "Station 1": ["Station 2", "Station 10"],
        "Station 2": ["Station 1", "Station 3", "Station 7"],
        "Station 3": ["Station 2", "Station 4", "Station 8"],
        "Station 4": ["Station 3", "Station 5", "Station 9"],
        "Station 5": ["Station 4", "Station 6"],
        "Station 6": ["Station 5", "Station 7"],
        "Station 7": ["Station 6", "Station 8", "Station 2"],
        "Station 8": ["Station 7", "Station 9", "Station 3"],
        "Station 9": ["Station 8", "Station 10", "Station 4"],
        "Station 10": ["Station 1", "Station 9"]
    }
    return adjacency_list, stations

# Randomly create train schedules
def create_trains(stations, num_trains=5):
    trains = []
    for i in range(num_trains):
        start = random.choice(stations)
        end = random.choice(stations)
        while end == start:  # Ensure start and end are different
            end = random.choice(stations)
        trains.append({"train_id": f"Train {i+1}", "start": start, "end": end})
    return trains

# Find the shortest path using BFS
def bfs_shortest_path(graph, start, end):
    queue = [[start]]  # Queue of paths
    visited = set()

    while queue:
        path = queue.pop(0)  # Get the first path in the queue
        station = path[-1]  # Get the last station from the path

        if station == end:
            return path  # Return path if the destination is reached

        if station not in visited:
            visited.add(station)
            for neighbor in graph[station]:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return None  # Return None if no path exists

# Find efficient paths for trains while avoiding station conflicts
def find_train_paths(graph, trains):
    paths = {}
    occupied_stations = {}  # Dictionary to track station occupancy at each time step

    for train in trains:
        train_id = train['train_id']
        start = train['start']
        end = train['end']

        # Find the shortest path using BFS
        path = bfs_shortest_path(graph, start, end)

        # Ensure no two trains occupy the same station simultaneously
        adjusted_path = []
        for station in path:
            time_step = len(adjusted_path)  # Current "time step" based on path length
            while occupied_stations.get((station, time_step), False):  # Check for conflict
                time_step += 1  # Delay train if there's a conflict
            adjusted_path.append((station, time_step))
            occupied_stations[(station, time_step)] = train_id  # Mark station as occupied
        
        paths[train_id] = adjusted_path
    return paths

# Visualize the train map and paths
def visualize_train_map(adjacency_list, trains, paths):
    # Assign fixed positions for stations for visualization purposes
    positions = {
        "Station 1": (0, 1),
        "Station 2": (1, 2),
        "Station 3": (2, 2),
        "Station 4": (3, 1),
        "Station 5": (3, 0),
        "Station 6": (2, -1),
        "Station 7": (1, -1),
        "Station 8": (0, 0),
        "Station 9": (1, 1),
        "Station 10": (2, 0)
    }

    plt.figure(figsize=(10, 8))

    # Draw stations (nodes)
    for station, (x, y) in positions.items():
        plt.scatter(x, y, s=500, color="lightblue", zorder=2)
        plt.text(x, y, station, fontsize=10, ha="center", va="center", zorder=3)

    # Draw connections (edges)
    for station, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            x1, y1 = positions[station]
            x2, y2 = positions[neighbor]
            plt.plot([x1, x2], [y1, y2], color="gray", zorder=1)

    # Draw train paths
    colors = ["red", "green", "blue", "purple", "orange"]
    for i, train in enumerate(trains):
        train_id = train["train_id"]
        path = paths[train_id]
        path_stations = [station for station, _ in path]

        for j in range(len(path_stations) - 1):
            station1 = path_stations[j]
            station2 = path_stations[j + 1]
            x1, y1 = positions[station1]
            x2, y2 = positions[station2]
            plt.plot([x1, x2], [y1, y2], color=colors[i % len(colors)], linewidth=2, zorder=4)
        
        # Annotate train start and end
        start_x, start_y = positions[path_stations[0]]
        end_x, end_y = positions[path_stations[-1]]
        plt.text(start_x, start_y + 0.2, f"{train_id} Start", fontsize=8, color=colors[i % len(colors)])
        plt.text(end_x, end_y - 0.2, f"{train_id} End", fontsize=8, color=colors[i % len(colors)])

    plt.title("Train Map and Train Paths")
    plt.axis("off")
    plt.show()

# Main function
def main():
    # Step 1: Create train map
    adjacency_list, stations = create_train_map()

    # Step 2: Create train schedules
    trains = create_trains(stations)

    # Step 3: Find efficient paths for trains
    paths = find_train_paths(adjacency_list, trains)

    # Step 4: Display results
    print("Train Schedules:")
    for train in trains:
        print(f"{train['train_id']}: Start at {train['start']}, End at {train['end']}")
    
    print("\nTrain Paths:")
    for train_id, path in paths.items():
        print(f"{train_id}: {' -> '.join([f'{station}(t={time})' for station, time in path])}")

    # Step 5: Visualize train map and paths
    visualize_train_map(adjacency_list, trains, paths)

# Run the program
if __name__ == "__main__":
    main()