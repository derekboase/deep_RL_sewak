"""Policy Iteration Algorithm
Code to demonstrate the Value Iteration method for solving Grid World
Deep RL, Chapter 3

Runtime: Python 3.9.7
Dependencies: numpy
DocStrings: NumpyStyle
"""

from envs.gridworld import GridWorldEnv
import numpy as np

class PolicyIteration:
    """The Policy Iteration Algorithm
    """
    def __init__(self, env=GridWorldEnv(), discountingFactor=0.9, convergenceThreshold=1e-4, 
                    iterationThresholdValue=1000, iterationThresholdPolicy=100, mode='prod'):
        """Initialize the PolicyIteration Class
        
        Parameters
        ----------
        env : (object)
            An instance of environment type
        
        discountingFactor : float
            The discounting factor for future rewards
        
        convergenceThreshold : float
            Threshold value for determining convergence
        
        iterationThresholdValue : int
            The maximum number of iteration to check for convergence of value

        iterationThresholdPolicy : int
            The maximum number of iteration to check for convergence of policy
        
        mode : str
            Mode (prod/debug) indicating the run mode. Effects the information/ verbosity
            of messages.
        
        Examples
        --------
        valueIteration = ValueIteration(env = GridWorldEnv(),mode='debug')
        """
        self.env = env
        self.gamma = discountingFactor
        self.th = convergenceThreshold
        self.maxIterValue = iterationThresholdValue
        self.maxIterPolicy = iterationThresholdPolicy
        self.stateCount = self.env.get_statespace_len()
        self.actionCount = self.env.get_actionspace_len()
        self.uniformActionProbability = 1.0/self.actionCount
        self.stateDict = self.env.stateDict
        self.actionDict = self.env.actionDict
        self.mode = mode
        self.V = np.zeros(self.stateCount)
        self.Q = [np.zeros(self.actionCount) for s in range(self.stateCount)]
        self.Policy = np.zeros(self.stateCount)
        self.totalReward = 0
        self.totalSteps = 0


    def reset_episode(self):
        """Resets the episode
        """
        self.totalReward = 0
        self.totalSteps = 0

    def compute_value_under_policy(self):
        self.V = np.zeros(self.stateCount)
        for i in range(self.maxIterValue):
            last_V = np.copy(self.V)
            for state_index in range(self.stateCount):
                current_state = self.env.statespace[state_index]
                for action in self.env.actionspace:
                    next_state = self.env.next_state(current_state, action)
                    reward = self.env.compute_reward(next_state)
                    next_state_index = self.env.stateDict[next_state]
                    self.Q[state_index][action] = reward + self.gamma*last_V[next_state_index]
                if self.mode == 'debug':
                    print('Q(s={}):{}'.format(current_state, self.Q[state_index]))
                self.V[state_index] = max(self.Q[state_index])
            if np.sum(np.fabs(last_V - self.V)) <= self.th:
                print("Convergence Achieved in {}th iteration. Breaking V-iteration loop!".format(i))
                break

    def iterate_policy(self):
        """Iterates over different (updated) policies
        
        Returns
        -------
        list
            A list of int with each element representing the action index in that state
        """
        self.Policy = [np.random.choice(self.actionCount) for s in range(self.stateCount)]
        for i in range(self.maxIterPolicy):
            self.compute_value_under_policy()
            old_policy = self.Policy
            self.improve_policy()
            new_policy = self.Policy 
            if np.all(old_policy == new_policy):
                print("Policy convergence achieved in step', i")
                break
            self.Policy = new_policy
        return self.Policy

    def improve_policy(self):
        """Improves a policy 
        Improves a policy by setting action for any state that gives the best Q value in that state
        """
        self.Policy = np.argmax(self.Q, axis=1)
        if self.mode == 'debug':
            print("Optimal Policy:", self.Policy)

    def run_episode(self):
        """Starts and runs a new episode
        
        Returns
        --------
        float:
            Total episode reward
        """
        self.reset_episode()
        obs = self.env.reset()
        while True:
            action = self.Policy[self.env.stateDict[obs]]
            new_obs, reward, done, _ = self.env.step(action)
            if self.mode == 'debug':
                print("PrevObs:{}, Action:{}, Obs:{}, Reward:{}, Done:{}".format(obs, action, new_obs, reward, done))
            self.totalReward += reward
            self.totalSteps += 1
            if done:
                break
            else:
                obs = new_obs
        return self.totalReward

    def evaluate_policy(self, n_episodes = 100):
        """Evaluates the goodness (mean score across different episodes) as per a policy
        
        Returns 
        --------
        float: 
            Policy score
        """
        episode_scores = []
        if self.mode == 'debug': print("Running {} episodes!".format(n_episodes))
        for e, episode in enumerate(range(n_episodes)):
            score = self.run_episode()
            episode_scores.append(score)
            if self.mode == 'debug': print("Score in {} episodes = {}".format(e, score))
        return np.mean(episode_scores)

    def solve_mdp(self, n_episode=10): 
        """Solves an MDP
        
        Returns
        --------
        float:
            The best/converged policy score
        """
        if self.mode == 'debug':
            print("Initializing variables and setting environment...")
        self.iterate_policy()

        if self.mode == 'debug':
            print("Scoring Policy")
        return self.evaluate_policy(n_episode)

if __name__ == '__main__':
    """Main function
    Main function to test the code and show an example.
    """
    
    print("Initializing variables and setting environment...")
    policyIteration = PolicyIteration(env=GridWorldEnv(), mode='debug')
    print('Policy Evaluation Score = ', policyIteration.solve_mdp())