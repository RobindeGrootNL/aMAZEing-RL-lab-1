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

class Maze:

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
    GOAL_REWARD = 1
    #IMPOSSIBLE_REWARD = -100
    #EATEN_REWARD = -100


    def __init__(self, maze, weights=None, random_rewards=False, minotaur_stay=False, cross_minotaur=True, jumping_allowed=True):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.cross_minotaur           = cross_minotaur
        self.jumping_allowed          = jumping_allowed
        self.actions_minotaur         = self.__actions_minotaur(stay=minotaur_stay);
        self.states, self.map, self.states_minotaur, self.map_minotaur = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1, 0);
        return actions;

    def __actions_minotaur(self, stay=False):
        """ if stay=True, the minotaur is allowed to perform the action "STAY"
        """
        actions_minotaur = dict();
        if stay == True:
            actions_minotaur[self.STAY]   = (0, 0);
        actions_minotaur[self.MOVE_LEFT]  = (0,-1);
        actions_minotaur[self.MOVE_RIGHT] = (0, 1);
        actions_minotaur[self.MOVE_UP]    = (-1,0);
        actions_minotaur[self.MOVE_DOWN]  = (1, 0);
        return actions_minotaur;

    def __states(self):
        states = dict();
        map = dict();
        states_minotaur = dict()
        map_minotaur = dict()
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    states[s] = (i,j);
                    map[(i,j)] = s;
                    states_minotaur[s] = (i,j);
                    map_minotaur[(i,j)] = s;
                    s += 1;
        return states, map, states_minotaur, map_minotaur

    def __move(self, state, state_minotaur, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls or (self.map[(row,col)] == state_minotaur and not self.cross_minotaur):
            return state;
        else:
            return self.map[(row, col)];

    def __possible_moves_minotaur(self, state):
        actions = self.actions_minotaur
        possible_actions = list()
        for _, action in actions.items():
            row = self.states_minotaur[state][0]
            col = self.states_minotaur[state][1]
            # Compute the future position given current (state, action)
            row += action[0];
            col += action[1];
            if ((row != -1) and (row != self.maze.shape[0]) and (col != -1) and (col != self.maze.shape[1])):
                if (self.maze[row,col] == 1):
                    if self.jumping_allowed == True:
                        if action[0]!=0: 
                            row += action[0]; #vertical jump
                        elif action[1]!=0: 
                            col += action[1]; #horizontal jump
                        if (self.maze[row,col] != 1): 
                            possible_actions.append((row, col)) #save this action if the jump doe snot end in a wall
                    else:
                        pass
                else:
                    possible_actions.append((row, col)) # TO DO: RENAME TO NEXT POSITIONS

        #n = len(possible_actions)
        #chosen_action = possible_actions[random.randint(0, n)]
        return possible_actions #self.map_minotaur[chosen_action]

    def __move_minotaur(self,state):
        possible_moves = self.__possible_moves_minotaur(state)
        n = len(possible_moves)
        random_number = random.randint(0,n-1)
        chosen_action = possible_moves[random_number]
        return self.map_minotaur[chosen_action]

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S_next,S,next_S_minotaur, S_minotaur,A)
        dimensions = (self.n_states, self.n_states, self.n_states, self.n_states, self.n_actions); # add minotaur dimension (n_states)
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities.
        for s in range(self.n_states):
            for s_minotaur in range(self.n_states):
                if s == s_minotaur or self.maze[self.states[s]] == 2:
                    transition_probabilities[s, s, s_minotaur, s_minotaur, :] = 1
                else:
                    next_states_minotaur = self.__possible_moves_minotaur(s_minotaur) #CALL FUNCTION RETURNING future positions of min
                    for a in range(self.n_actions):
                        next_s = self.__move(s,s_minotaur,a);
                        # check current states of player and minotaur. If current state is the same --> cannot move = action0
                        for next_pos_minotaur in next_states_minotaur:
                            next_s_minotaur = self.map_minotaur[next_pos_minotaur]
                            transition_probabilities[next_s, s, next_s_minotaur, s_minotaur, a] = 1/(len(next_states_minotaur))
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_states, self.n_actions));
        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for s_minotaur in range(self.n_states):
                    if s == s_minotaur:
                        rewards[s, s_minotaur, :] = 0
                    else:
                        for a in range(self.n_actions):
                            next_s = self.__move(s,s_minotaur,a);
                            # Reward for reaching the exit
                            if s != next_s and self.maze[self.states[next_s]] == 2:
                                rewards[s,s_minotaur,a] = self.GOAL_REWARD;
                            # Reward for taking a step to an empty cell that is not the exit
                            else:
                                rewards[s,s_minotaur,a] = self.STEP_REWARD;
        return rewards;

    def simulate(self, start, policy, method, life_mean=30, start_minotaur=(6, 5)):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        path_minotaur = list()
        exited_maze = False
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[2]#-1;
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            s_minotaur = self.map[start_minotaur]
            # Add the starting position in the maze to the path
            path.append(start);
            path_minotaur.append(start_minotaur)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,s_minotaur,policy[s,s_minotaur,t]);
                next_s_minotaur = self.__move_minotaur(s_minotaur)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                path_minotaur.append(self.states[next_s_minotaur])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
                s_minotaur = next_s_minotaur
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            p = 1 / life_mean
            T = np.random.geometric(p)
            s = self.map[start];
            s_minotaur = self.map[start_minotaur]
            # Add the starting position in the maze to the path
            path.append(start);
            path_minotaur.append(start_minotaur)
            # Move to next state given the policy and the current state
            next_s = self.__move(s,s_minotaur,policy[s,s_minotaur]);
            next_s_minotaur = self.__move_minotaur(s_minotaur)
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            path_minotaur.append(self.states[next_s_minotaur])
            # Loop while state is not the goal state
            while not exited_maze and t < T:
                # Update state
                s = next_s;
                s_minotaur = next_s_minotaur
                # Move to next state given the policy and the current state
                next_s = self.__move(s,s_minotaur,policy[s,s_minotaur]);
                next_s_minotaur = self.__move_minotaur(s_minotaur)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                path_minotaur.append(self.states[next_s_minotaur])
                # Update time and state for next iteration
                t +=1;

                if self.maze[self.states[next_s]] == 2:
                    exited_maze = True
        return path, path_minotaur, exited_maze, t


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
                for s_minotaur in range(n_states):
                    Q[s,s_minotaur,a] = r[s,s_minotaur,a] + np.dot(p[:,s,:,s_minotaur,a].flatten(),V[:,:,t+1].flatten().T)
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
        for s_minotaur in range(n_states):
            for a in range(n_actions):
                Q[s, s_minotaur, a] = r[s, s_minotaur, a] + gamma*np.dot(p[:,s,:,s_minotaur,a].flatten(),V.flatten().T);
    BV = np.max(Q, 2);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for s_minotaur in range(n_states):
                for a in range(n_actions):
                    Q[s, s_minotaur, a] = r[s, s_minotaur, a] + gamma*np.dot(p[:,s,:,s_minotaur,a].flatten(),V.flatten().T);
        BV = np.max(Q, 2);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,2);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

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

def animate_solution(maze, path, path_minotaur, gif_name, end_state=(6,5)):

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
        grid.get_celld()[(path_minotaur[i])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path_minotaur[i])].get_text().set_text('Minotaur')
        if i > 0:
            if path[i] != path[i-1]:
                if path_minotaur[i] != path[i-1]:
                    grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
                    grid.get_celld()[(path[i-1])].get_text().set_text('')
                if path[i] != path_minotaur[i-1] and path_minotaur[i] != path_minotaur[i-1]:
                    grid.get_celld()[(path_minotaur[i-1])].set_facecolor(col_map[maze[path_minotaur[i-1]]])
                    grid.get_celld()[(path_minotaur[i-1])].get_text().set_text('')
            if path[i] == path_minotaur[i]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i])].get_text().set_text('Player dead')
            if path[i] == end_state:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Player escaped')
            if path[i] == path[i-1]:
                grid.get_celld()[(path_minotaur[i-1])].set_facecolor(col_map[maze[path_minotaur[i-1]]])
                grid.get_celld()[(path_minotaur[i-1])].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        # Used to return the plot as an image array
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))        


        figure_list.append(image)
        time.sleep(0.1)

    imageio.mimsave(gif_name, figure_list, fps=2)

if __name__ == '__main__':
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])
    # with the convention 
    # 0 = empty cell
    # 1 = obstacle
    # 2 = exit of the Maze

    #--------------------------------------------------------------------------------------------------------------------
    # DYNAMIC PROGRAMMING
    #--------------------------------------------------------------------------------------------------------------------

    # MINOTAUR IS NOT ALLOWED TO STAY
    env = Maze(maze, minotaur_stay=False, cross_minotaur=True, jumping_allowed=True)

    horizon = 20
    V, policy= dynamic_programming(env,horizon);

    method = 'DynProg';
    start  = (0,0);
    path, path_minotaur, _, _ = env.simulate(start, policy, method,start_minotaur=(6, 5));
    animate_solution(maze, path, path_minotaur, gif_name="DP_nostay.gif")
    plt.clf()

    max_probs = []
    start_player = env.map[(0,0)]
    start_minotaur = env.map[(6,5)]
    for T in range(0,31):
        horizon = T    
        V, policy = dynamic_programming(env,horizon)
        max_probs.append(V[start_player, start_minotaur, 0]) 

    plt.scatter(x=range(0,len(max_probs)), y=max_probs)
    plt.grid(True)
    plt.suptitle("Max P(Escape) as a function of the time horizon", fontsize=16)
    plt.title("Minotaur cannot stay", fontsize=12)
    plt.xlabel("Time horizon")
    plt.ylabel("Max P(Escape)")
    plt.savefig("DP_nostay.png")
    plt.clf()
    # MINOTAUR IS ALLOWED TO STAY
    env = Maze(maze, minotaur_stay=True, cross_minotaur=True, jumping_allowed=True)

    horizon = 20
    V, policy= dynamic_programming(env,horizon);

    method = 'DynProg';
    start  = (0,0);
    path, path_minotaur, _, _ = env.simulate(start, policy, method,start_minotaur=(6, 5));
    animate_solution(maze, path, path_minotaur, gif_name="DP_stay.gif")
    plt.clf()

    max_probs_stay = []
    start_player = env.map[(0,0)]
    start_minotaur = env.map[(6,5)]
    for T in range(0,31):
        horizon = T    
        V, policy = dynamic_programming(env,horizon)
        max_probs_stay.append(V[start_player, start_minotaur, 0])

    plt.scatter(x=range(0,len(max_probs_stay)), y=max_probs_stay)
    plt.grid(True)
    plt.suptitle("Max P(Escape) as a function of the time horizon", fontsize=16)
    plt.title("Minotaur can stay", fontsize=12)
    plt.xlabel("Time horizon")
    plt.ylabel("Max P(Escape)")
    plt.savefig("DP_stay.png")
    plt.clf()
    #--------------------------------------------------------------------------------------------------------------------
    # VALUE ITERATION
    #--------------------------------------------------------------------------------------------------------------------

    # MINOTAUR IS NOT ALLOWED TO STAY
    env = Maze(maze, minotaur_stay=False, cross_minotaur=True, jumping_allowed=True)
    
    # Discount Factor 
    gamma   = 1-1/30;
    # Accuracy treshold 
    epsilon = 0.001;
    V, policy = value_iteration(env, gamma, epsilon)
    
    method = 'ValIter';
    start  = (0,0);
    
    exited_maze_list = list()
    
    n_iters = 10000
    for i in range(n_iters):
        path, path_minotaur, exited_maze, t = env.simulate(start, policy, method, life_mean=30, start_minotaur=(6,5))
        exited_maze_list.append(exited_maze)
    
    probability_of_exit = sum(exited_maze_list)/n_iters
    print("Probability of exiting the maze: ", probability_of_exit)
    animate_solution(maze, path, path_minotaur, gif_name="VI_nostay.gif")
    plt.clf()

    # MINOTAUR IS NOT ALLOWED TO STAY
    env = Maze(maze, minotaur_stay=True, cross_minotaur=True, jumping_allowed=True)
    
    # Discount Factor 
    gamma   = 1-1/30;
    # Accuracy treshold 
    epsilon = 0.001;
    V, policy = value_iteration(env, gamma, epsilon)
    
    method = 'ValIter';
    start  = (0,0);
    
    exited_maze_stay_list = list()
    
    n_iters = 10000
    for i in range(n_iters):
        path, path_minotaur, exited_maze, t = env.simulate(start, policy, method, life_mean=30, start_minotaur=(6,5))
        exited_maze_stay_list.append(exited_maze)
    
    probability_of_exit_stay = sum(exited_maze_stay_list)/n_iters
    print("Probability of exiting the maze: ", probability_of_exit_stay)
    animate_solution(maze, path, path_minotaur, gif_name="VI_stay.gif")
    plt.clf()
