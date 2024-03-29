""""Grid World Environment
Custom Environment for the Grid World Problem as in the book 
Deep RL, Chapter 3

Runtime: Python 3.9.12
Dependencies: None
DocStrings: NumpyStyle
"""

class GridWorldEnv():
    """Grid-World EnvironmentClass
    """
    def __init__(self, gridsize=7, startState='00', terminalStates=['64'], ditches=['52'], ditchPenalty=-10, turnPenalty=-1, winReward=100, mode='prod'):
        """Initialize an instance of Grid-World environment 
        
        Parameters
        ----------
        gridsize : int
            The size of the (square) grid n x n.
        
        startState : str
            The entry point for the game in terms of coordinates as string.
        
        terminalStates : str
            The goal/ terminal state for the game in terms of coordinates as string.
        
        ditches : list(str)
            A list of ditches/penalty-spots. Each element coded as str of coordinates.
            
        ditchPenalty : int
            A Negative Reward for arriving at any of the ditch cell as in ditches parameter.
            
        turnPenalty : int
            A Negative Reward for every turn to ensure that agent completes the epi-sode in minimum number of turns.
            
        winReward : int            
            A Negative Reward for reaching the goal/ terminal state.
            
        mode : str
            Mode (prod/debug) indicating the run mode. Effects the information/ ver-bosity of messages.

        Examples
        ----------
        env = GridWorldEnv(mode='debug')
        """
        self.mode = mode
        self.gridsize = min(gridsize, 9)
        self.create_statespace()
        self.actionspace = [0, 1, 2, 3]
        self.actionDict = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.startState = startState
        self.terminalStates = terminalStates
        self.ditches = ditches
        self.winReward = winReward
        self.ditchPenalty = ditchPenalty
        self.turnPenalty = turnPenalty
        self.stateCount = self.get_statespace_len()
        self.actionCount = self.get_actionspace_len()
        self.stateDict = {k: v for k, v in zip(self.statespace, range(self.stateCount))}
        self.currentState = self.startState

        if self.mode == 'debug':
            print("State Space", self.statespace)
            print("State Dict", self.stateDict)
            print("Action Space", self.actionspace)
            print("Start State", self.startState)
            print("Terminal States", self.terminalStates)
            print("Ditches", self.ditches)
            print("WinReward:{}, TurnPenalty:{}, DitchPenalty {}".format(self.winReward, self.turnPenalty, self.ditchPenalty))

    def create_statespace(self):
        """Create Statespace
    
        Makes the grid world space with as many grid-cells as requested during 
        instantiation gridsize parameter.
        """
        self.statespace=[]
        for row in range(self.gridsize):
            for col in range(self.gridsize):
                self.statespace.append(str(row) + str(col))

    def set_mode(self, mode):
        self.mode = mode

    def get_statespace(self):
        return self.statespace

    def get_actionspace(self):
        return self.actionspace

    def get_actiondict(self):
        return self.actionDict

    def get_statespace_len(self):
        return len(self.statespace)

    def get_actionspace_len(self):
        return len(self.actionspace)

    def next_state(self, current_state, action):
        """Next State

        Determines the next state, given the current state and action as per the game rule.

        Parameters
        ----------
        current_state: (int, int)
            A tuple of current state coordinate
        
        action: int
            Action index

        Returns
        ----------
        str
            New state coded as str of coordinates
        """
        s_row = int(current_state[0])
        s_col = int(current_state[1])
        next_row = s_row
        next_col = s_col
        if action == 0: next_row = max(0, s_row - 1)
        if action == 1: next_row = min(self.gridsize - 1, s_row + 1)
        if action == 2: next_col = max(0, s_col - 1)
        if action == 3: next_col = min(self.gridsize - 1, s_col + 1)

        new_state = str(next_row) + str(next_col)
        if new_state in self.statespace:
            if new_state in self.terminalStates: self.isGameEnd = True
            if self.mode=='debug':
                print("CurrentState:{}, Action:{}, NextState:{}".format(current_state, action, new_state))
            return new_state
        else: 
            return current_state
    def compute_reward(self, state):
        """Compute Reward
            Computes the reward for arriving at a given state based on ditches, and
            goals as requested during instatiations.

            Parameters
            ----------
            state: str
                Current state in coordinates coded as single str
            Returns
            ----------
            float
                reward corresponding to the entered state        
        """
        reward = 0 
        reward += self.turnPenalty
        if state in self.ditches: reward += self.ditchPenalty
        if state in self.terminalStates: reward += self.winReward
        return reward
    
    def reset(self):
        """Resets the environment. 
            Required in gym standard format
        
        Returns 
        ----------
        str
            A string representing the reset state, i.e. the entry point for the agent at start of game.

        Example
        ----------
        env.reset()
        """

        self.isGameEnd=False
        self.totalAccumulatedReward=0
        self.totalTurns=0
        self.currentState=self.startState
        return self.currentState

    def step(self, action):
        """step

            Takes a step corresponding to the action suggested. Required in gym standard format

        Parameters
        ----------
        action: int
            Index of the action taken

        Returns 
        ----------
        tuple
            A tuple of (next_state, instantaneous_reward, done_flag, info)

        Examples
        ----------
        observation_tuple=env.step(1)
        next_state, reward, done, _ =env.step(2)
        """
        if self.isGameEnd:
            raise('Game is Over Exception')
        if action not in self.actionspace:
            raise('Invalid Action Exception')
        self.currentState = self.next_state(self.currentState, action)
        obs = self.currentState
        reward = self.compute_reward(obs)
        done = self.isGameEnd
        self.totalTurns += 1
        if self.mode == 'debug':
            print("Obs: {}, Reward: {}, Done: {}, TotalTurns: {}".format(obs, reward, done, self.totalTurns))
        return obs, reward, done, self.totalTurns

if __name__ == '__main__':
    """Main function
    Mainfunction to test the code and show an example
    """
    env = GridWorldEnv(mode='debug')
    # env = GridWorldEnv()
    print("Resetting Env...")
    env.reset()
    print("Go DOWN")
    env.step(1)
    print("Go RIGHT")
    env.step(3)
    print("Go LEFT")
    env.step(2)
    print("Go UP")
    env.step(0)
    # opt_pol = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1,1,1,1,3,3,3,3,1,2,2]
    # for act in opt_pol: 
    #     env.step(act)
    
