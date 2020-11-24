import imageio
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
LIGHT_GREY   = '#D3D3D3';

class Hood:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    BANK_REWARD = 10
    CAUGHT_REWARD = -50
    #IMPOSSIBLE_REWARD = -100
    #EATEN_REWARD = -100


    def __init__(self, hood, starting_point=(0,0), starting_point_police=(1,2)):
        """ Constructor of the environment Maze.
        """
        self.hood                     = hood;
        self.starting_point, self.starting_point_police       = starting_point, starting_point_police;
        self.actions                  = self.__actions();
        self.actions_police           = self.__actions_police();
        self.states, self.map, self.states_police, self.map_police = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards();

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1, 0);
        return actions;

    def __actions_police(self):
        """ if stay=True, the minotaur is allowed to perform the action "STAY"
        """
        actions_police = dict();
        actions_police[self.MOVE_LEFT]  = (0,-1);
        actions_police[self.MOVE_RIGHT] = (0, 1);
        actions_police[self.MOVE_UP]    = (-1,0);
        actions_police[self.MOVE_DOWN]  = (1, 0);
        return actions_police;

    def __states(self):
        states = dict();
        map = dict();
        states_police = dict()
        map_police = dict()
        end = False;
        s = 0;
        for i in range(self.hood.shape[0]):
            for j in range(self.hood.shape[1]):
                states[s] = (i,j);
                map[(i,j)] = s;
                states_police[s] = (i,j);
                map_police[(i,j)] = s;
                s += 1;
        return states, map, states_police, map_police

    def __move(self, state, state_police, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.hood.shape[0]) or \
                              (col == -1) or (col == self.hood.shape[1])
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state;
        else:
            return self.map[(row, col)];

    def __possible_moves_police(self, state, state_police):
        actions = self.actions_police
        possible_next_states = list()
        for _, action in actions.items():
            row = self.states_police[state_police][0]
            col = self.states_police[state_police][1]
            # Compute the future position given current (state, action)
            row_next = row + action[0];
            col_next = col + action[1];
            if ((row_next != -1) and (row_next != self.hood.shape[0]) and (col_next != -1) and (col_next != self.hood.shape[1])):
                # Check if police is not going to the opposite direction (pos_police*next_move_police <= pos_player*next_move_police)
                if (row*action[0] + col*action[1]) <= (self.states[state][0]*action[0] + self.states[state][1]*action[1]):
                    possible_next_states.append((row_next, col_next))
        return possible_next_states

    def __move_police(self,state, state_police):
        possible_next_states = self.__possible_moves_police(state, state_police)
        n = len(possible_next_states)
        random_number = random.randint(0,n-1)
        chosen_action = possible_next_states[random_number]
        return self.map_police[chosen_action]

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S_next,S,next_S_minotaur, S_minotaur,A)
        dimensions = (self.n_states, self.n_states, self.n_states, self.n_states, self.n_actions); # add minotaur dimension (n_states)
        transition_probabilities = np.zeros(dimensions);
        starting_state = self.map[self.starting_point]
        starting_state_police = self.map_police[self.starting_point_police]
        # Compute the transition probabilities.
        for s in range(self.n_states):
            for s_police in range(self.n_states):
                if s == s_police:
                    transition_probabilities[starting_state, s, starting_state_police, s_police, :] = 1
                else:
                    next_states_police = self.__possible_moves_police(s, s_police)
                    for a in range(self.n_actions):
                        next_s = self.__move(s,s_police,a);
                        for next_pos_police in next_states_police:
                            next_s_police = self.map_police[next_pos_police]
                            transition_probabilities[next_s, s, next_s_police, s_police, a] = 1/(len(next_states_police))
        return transition_probabilities;

    def __rewards(self):
        rewards = np.zeros((self.n_states, self.n_states, self.n_actions));
        for s in range(self.n_states):
            for s_police in range(self.n_states):
                if s == s_police:
                    rewards[s, s_police, :] = self.CAUGHT_REWARD
                elif self.hood[self.states[s]] == 1:
                    rewards[s, s_police, :] = self.BANK_REWARD
                else:
                    rewards[s, s_police, :] = self.STEP_REWARD
                    #for a in range(self.n_actions):
                    #    next_s = self.__move(s,s_minotaur,a);
                    #    # Reward for reaching the exit
                    #    if s != next_s and self.hood[self.states[next_s]] == 2:
                    #        rewards[s,s_minotaur,a] = self.GOAL_REWARD;
                    #    # Reward for taking a step to an empty cell that is not the exit
                    #    else:
                    #        rewards[s,s_minotaur,a] = self.STEP_REWARD;
        return rewards;

    def simulate(self, start, policy, method, life_mean=30, start_police=(6, 5)):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        path_police = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[2]#-1;
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            s_police = self.map[start_police]
            # Add the starting position in the maze to the path
            path.append(start);
            path_police.append(start_police)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,s_police,policy[s,s_police,t]);
                next_s_police = self.__move_police(s, s_police)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                path_police.append(self.states[next_s_police])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
                s_police = next_s_police
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            p = 1 / life_mean
            T = np.random.geometric(p)
            s = self.map[start];
            s_police = self.map[start_police]
            # Add the starting position in the maze to the path
            path.append(start);
            path_police.append(start_police)
            # Move to next state given the policy and the current state
            next_s = self.__move(s,s_police,policy[s,s_police]);
            next_s_police = self.__move_police(s, s_police)
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            path_police.append(self.states[next_s_police])
            # Loop while state is not the goal state
            while t < T:
                # Update state
                s = next_s;
                s_police = next_s_police
                # Move to next state given the policy and the current state
                next_s = self.__move(s,s_police,policy[s,s_police]);
                next_s_police = self.__move_police(s, s_police)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                path_police.append(self.states[next_s_police])
                # Update time and state for next iteration
                t +=1;
        return path, path_police, t


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;
    # The variables involved in the dynamic programming backwards recursions
    # added a dimension for the value function and policy for the position of the minotaur
    V      = np.zeros((n_states, n_states, T+1));
    policy = np.zeros((n_states, n_states, T+1));
    #Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q               = np.copy(r);
    V[:, :, T]      = np.max(Q,2);
    policy[:, :, T] = np.argmax(Q,2);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                for s_police in range(n_states):
                    Q[s,s_police,a] = r[s,s_police,a] + np.dot(p[:,s,:,s_police,a].flatten(),V[:,:,t+1].flatten().T)
        # Update by taking the maximum Q value w.r.t the action a
        V[:,:,t] = np.max(Q,2);
        # The optimal action is the one that maximizes the Q function
        policy[:,:,t] = np.argmax(Q,2);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros((n_states, n_states));
    Q   = np.zeros((n_states, n_states, n_actions));
    BV  = np.zeros((n_states, n_states));
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI/
    for s in range(n_states):
        for s_police in range(n_states):
            for a in range(n_actions):
                Q[s, s_police, a] = r[s, s_police, a] + gamma*np.dot(p[:,s,:,s_police,a].flatten(),V.flatten().T);
    BV = np.max(Q, 2);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for s_police in range(n_states):
                for a in range(n_actions):
                    Q[s, s_police, a] = r[s, s_police, a] + gamma*np.dot(p[:,s,:,s_police,a].flatten(),V.flatten().T);
        BV = np.max(Q, 2);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,2);
    # Return the obtained policy
    return V, policy;

def draw_hood(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: LIGHT_GREY, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path, path_police, gif_name, end_state=(6,5)):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

    figure_list = list()
    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i])].get_text().set_text('Player')
        grid.get_celld()[(path_police[i])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path_police[i])].get_text().set_text('Police')
        if i > 0:
            if path[i] != path[i-1]:
                if path_police[i] != path[i-1]:
                    grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
                    grid.get_celld()[(path[i-1])].get_text().set_text('')
                if path[i] != path_police[i-1]:
                    grid.get_celld()[(path_police[i-1])].set_facecolor(col_map[maze[path_police[i-1]]])
                    grid.get_celld()[(path_police[i-1])].get_text().set_text('')
            if path[i] == path_police[i]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i])].get_text().set_text('Player dead')
            if path[i] == end_state:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Player escaped')
            if path[i] == path[i-1]:
                grid.get_celld()[(path_police[i-1])].set_facecolor(col_map[maze[path_police[i-1]]])
                grid.get_celld()[(path_police[i-1])].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        # Used to return the plot as an image array
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))        


        figure_list.append(image)
        time.sleep(0.1)

    imageio.mimsave(gif_name, figure_list, fps=2)