import cv2
import matplotlib.pyplot as plt
import numpy as np
import heapq
from matplotlib.animation import FuncAnimation
import tkinter as tk



def convert_image_to_bool_matrix(img):
    """
    Converts an image to a boolean matrix where white pixels (255) represent open paths,
    and all other pixels represent walls.
    
    Args:
        img (numpy.ndarray): The input image.
    
    Returns:
        numpy.ndarray: A boolean matrix where True represents open paths and False represents walls.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray == 255


def get_neighbors(matrix, position):
    """
    Finds valid neighboring positions in the maze matrix.
    
    Args:
        matrix (numpy.ndarray): The maze matrix.
        position (tuple): The current position as (row, col).
    
    Returns:
        list: A list of valid neighboring positions.
    """

    neighbors = []
    for dx, dy in [(1, 1), (-1, -1), (-1, 1), (1, -1), (-1, 0), (0, -1), (1, 0), (0, 1)]: 
        new_position = (position[0] + dx, position[1] + dy)
        if 0 <= new_position[0] < matrix.shape[0] and 0 <= new_position[1] < matrix.shape[1]:
            if matrix[new_position]:
                neighbors.append(new_position)
    return neighbors


def dijkstra(matrix, start, end): # O((V +E) log V)
    """
    Implements Dijkstra's algorithm to find the shortest path in a boolean maze.
    
    Args:
        matrix (numpy.ndarray): The maze matrix.
        start (tuple): The starting position (row, col).
        end (tuple): The ending position (row, col).
    
    Returns:
        list: The shortest path as a list of positions, or None if no path is found.
    """

    distances = np.full(matrix.shape, np.inf)
    distances[start] = 0
    predecessors = np.full(matrix.shape, None, dtype=object)
    pq = [(0, start)]
    visited = set()

    while pq:
        current_distance, current_position = heapq.heappop(pq) # o(v log v)

        if current_position in visited:
            continue

        visited.add(current_position)

        if current_position == end:
            return reconstruct_path(predecessors, start, end)

        for neighbor in get_neighbors(matrix, current_position):
            if neighbor in visited:
                continue
            distance = current_distance + 1
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_position
                heapq.heappush(pq, (distance, neighbor))

    return None  # No path found


def reconstruct_path(predecessors, start, end):
     """
    Reconstructs the shortest path from start to end using the predecessors matrix.
    
    Args:
        predecessors (numpy.ndarray): The matrix storing the previous position for each visited node.
        start (tuple): The starting position.
        end (tuple): The ending position.
    
    Returns:
        list: The shortest path as a list of positions.
    """
    path = []
    current = end
    while current != start:
        path.append(current)
        current = predecessors[current]
    path.append(start)
    return path[::-1]


def drawPath(img, path, thickness=2):
    """
    Draws the computed path on the maze image.
    
    Args:
        img (numpy.ndarray): The input maze image.
        path (list): The list of positions in the shortest path.
        thickness (int, optional): Thickness of the path lines. Defaults to 2.
    """
    for i in range(len(path) - 1):
        pt1 = (path[i][1], path[i][0])  
        pt2 = (path[i + 1][1], path[i + 1][0])
        cv2.line(img, pt1, pt2, (0, 0, 255), thickness)


def solve_and_draw_maze(image_path, start, end):
    """
    Loads a maze image, solves it using Dijkstra's algorithm, and visualizes the solution.
    
    Args:
        image_path (str): Path to the maze image file.
        start (tuple): Starting position (row, col).
        end (tuple): Ending position (row, col).
    """
    img = cv2.imread(image_path)
    if img is None:
        # print(f"Failed to read the image from path: {image_path}")
        return

    bool_matrix = convert_image_to_bool_matrix(img)

    # Debug: Print maze structure
    # print("Maze structure:")
    # print(bool_matrix.astype(int))

    path = dijkstra(bool_matrix, start, end)

    fig, ax = plt.subplots(figsize=(10, 7))
    im_display = ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    def update(frame):
        if frame < len(path):
            point = path[frame]
            cv2.circle(img, (point[1], point[0]), 2, (0, 255, 0), -1)
            im_display.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return [im_display]

    ani = None

    def on_key(event):
        nonlocal ani
        if event.key == 'enter':
            if path:
                # print("Path found:", path)
                ani = FuncAnimation(fig, update, frames=range(0, len(path), 2), interval=1, blit=True)
                fig.canvas.draw_idle()

            else:
                print("No path found from start to end.")
                # Debug: Mark start and end points
                cv2.circle(img, (start[1], start[0]), 5, (0, 255, 0), -1)
                cv2.circle(img, (end[1], end[0]), 5, (0, 0, 255), -1)
                im_display.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                fig.canvas.draw_idle()



    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()





# Usage
image_path = "maze6.png"
start = (558, 22)  # (row, col)
end = (9, 69)  # (row, col)

solve_and_draw_maze(image_path, start, end)
