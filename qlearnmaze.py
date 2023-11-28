import numpy as np
import random
import matplotlib.pyplot as plt

# Stops the q-table values from printing in scientific notation for easier analysis of q-tables
np.set_printoptions(suppress=True)

# Controls how many episodes are run
num_episodes = 100

# Controls how many times the Q-agent trains in an episode
max_epochs = 1000

# Define the maze environment
# 0 = traversable, -1 = wall, 10 = goal

# you can freely move around the goal position
# (0,0) must be 0, the starting position is static!
maze = np.array([
    [0, 0, 0, -1, 0],
    [0, -1, 0, 0, 0],
    [0, -1, 0, -1, 0],
    [0, -1, 0, -1, 0],
    [0, 0, 0, 0, 10]
])


# Works with larger mazes such as 10x10
# Warning, too complex of a maze will make the agent take much longer to learn optimal paths
# Change maze1 to maze to use, only choose one maze!
maze1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, -1, -1, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, -1, -1, 0, -1],
    [0, 0, -1, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
    [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0],
    [10, 0, 0, 0, 0, 0, 0, -1, 0, 0]
])

maze_num_rows = len(maze)
maze_num_cols = len(maze[0])

def find_position(matrix, element):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == element:
                return i, j  # Return the position (row, column) of the element
            
def find_position_i(matrix, element):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == element:
                return i

def find_position_j(matrix, element):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == element:
                return j

position_goal = find_position(maze, 10)
position_i = find_position_i(maze, 10)
position_j = find_position_j(maze, 10)

# Define possible actions (up, down, left, right)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Define epsilon-greedy policy parameters
epsilon = 0.1 # Randomness chance
alpha = 0.1 # Learning rate, lower rates make take longer, but is more stable in finding the optimal path
gamma = 0.9 # Discount factor
step_penalty = -0.01 # Penalizes the agent for taking long routes (affects the reward)

# Initialize Q-table with random values
q_table = np.random.rand(maze.shape[0], maze.shape[1], len(actions))

# Function to choose an action based on epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))  # Explore randomly
    else:
        return np.argmax(q_table[state])

# Function to update Q-values using approximate Q-learning
def update_q_value(state, action, reward, next_state, count):
    old_value = q_table[state][action]
    next_max = np.max(q_table[next_state])
    new_value = (1 - alpha) * old_value + alpha * ((reward + (count * step_penalty)) + gamma * next_max)
    q_table[state][action] = new_value

def q_learning():
    # print("in q learning") # for debugging purposes
    convergence_threshold = 1e-6
    prev_q_table = np.copy(q_table)

    goal_position = position_goal

    for epoch in range(max_epochs):
        count = 0
        max_steps = 1000000
        state = (0, 0)  # Start from the top-left corner
        print("Current epoch:", epoch + 1)

        while state != goal_position:  # Until reaching the goal
            action_idx = choose_action(state)
            action = actions[action_idx]

            next_row, next_col = state[0] + action[0], state[1] + action[1]
            if 0 <= next_row < maze.shape[0] and 0 <= next_col < maze.shape[1] and maze[next_row][next_col] != -1:
                next_state = (next_row, next_col)
                reward = maze[next_row][next_col]
                update_q_value(state, action_idx, reward, next_state, count)
                state = next_state
                count += 1

            if count > max_steps:
                print("too many steps, breaking")
                count = 0
                break

        # Check for convergence by comparing Q-values with a threshold
        q_difference = np.sum(np.abs(q_table - prev_q_table))
        if q_difference < convergence_threshold:
            print(f"Converged at epoch {epoch + 1}")
            print("# of actions taken:", count)
            break

        prev_q_table = np.copy(q_table)

    print("Q-values:")
    print(q_table)

# Function to visualize the agent's path during training for a single iteration
def visualize_training():
    # print("in visualize training") # for debugging purposes
    path = []
    state = (0, 0)  # Start from the top-left corner
    path.append(state)

    goal_position = position_goal
    
    while state != goal_position:  # Until reaching the goal
        action_idx = choose_action(state)
        action = actions[action_idx]

        next_row, next_col = state[0] + action[0], state[1] + action[1]
        if 0 <= next_row < maze.shape[0] and 0 <= next_col < maze.shape[1] and maze[next_row][next_col] != -1:
            next_state = (next_row, next_col)
            path.append(next_state)
            state = next_state

    return path

# Function to run multiple episodes of Q-learning
def q_learning_multiple_episodes(num_episodes, visualize=False):
    all_paths = []
    for episode in range(num_episodes):
        print("Episode", episode+1)
        if visualize and episode in [0, num_episodes - 1]:
            path = visualize_training()  # Visualize the path for this episode
            all_paths.append(path)
        else:
            q_learning()  # Run Q-learning for a single episode
    
    return all_paths

# Function to visualize the agent's path during the last episode
def visualize_last_episode(all_paths):
    i_position = position_i
    j_position = position_j
    if all_paths:
        last_episode_path = all_paths[-1]  # Get the path of the last episode
        rows, cols = zip(*last_episode_path)
        plt.figure(figsize=(maze_num_rows + 1, maze_num_cols + 1)) # Dynamically update size of graph w/ maze size

        # Modify the maze for visualization purposes
        maze_for_plot = np.copy(maze)
        maze_for_plot[maze_for_plot == -1] = -2  # Change -1 to -2 for walls
        maze_for_plot[maze_for_plot == 10] = 2  # Change 10 to 2 for the goal

        plt.imshow(maze_for_plot, cmap='viridis', interpolation='none', vmin=-2, vmax=2)  # Plot the modified maze

        # Plot the path taken by the agent
        plt.plot(cols, rows, color='red', marker='o')

        # Mark start and end points
        plt.text(0, 0, 'Start', ha='center', va='center', color='blue')
        plt.text(j_position, i_position, 'Goal', ha='center', va='center', color='green')

        # Calculate the number of actions taken (length of the path - 1)
        num_actions = len(last_episode_path) - 1
        print(f"Number of actions taken in the last episode: {num_actions}")

        plt.xticks(range(maze_num_rows))
        plt.yticks(range(maze_num_cols))

        plt.axis('off')

        plt.title('Agent\'s Path in the Last Episode')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.grid(True)
        plt.show()

# Function to visualize the agent's path during the first episode
def visualize_first_episode(all_paths):
    i_position = position_i
    j_position = position_j
    if all_paths:
        first_episode_path = all_paths[0]  # Get the path of the first episode
        rows, cols = zip(*first_episode_path)
        plt.figure(figsize=(maze_num_rows + 1, maze_num_cols + 1))

        # Modify the maze for visualization purposes
        maze_for_plot = np.copy(maze)
        maze_for_plot[maze_for_plot == -1] = -2  # Change -1 to -2 for walls
        maze_for_plot[maze_for_plot == 10] = 2  # Change 10 to 2 for the goal

        plt.imshow(maze_for_plot, cmap='viridis', interpolation='none', vmin=-2, vmax=2)  # Plot the modified maze

        # Plot the path taken by the agent
        plt.plot(cols, rows, color='red', marker='o')

        # Mark start and end points
        plt.text(0, 0, 'Start', ha='center', va='center', color='blue')
        plt.text(j_position, i_position, 'Goal', ha='center', va='center', color='green')

        # Calculate the number of actions taken (length of the path - 1)
        num_actions = len(first_episode_path) - 1
        print(f"Number of actions taken in the first episode: {num_actions}")

        plt.xticks(range(maze_num_rows))
        plt.yticks(range(maze_num_cols))

        plt.axis('off')

        plt.title('Agent\'s Path in the First Episode')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.grid(True)
        plt.show()



# Run episodes of Q-learning
all_paths = q_learning_multiple_episodes(num_episodes, visualize=True)

# Visualize the path taken in the first and last episode
visualize_first_episode(all_paths)
visualize_last_episode(all_paths)