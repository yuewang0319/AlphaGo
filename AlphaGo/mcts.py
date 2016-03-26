


LAMBDA = 0.5
THRESHOLD = 50



class TreeNode(object):
    """Tree Representation of MCTS that covers Selection, Expansion, Evaluation, Backup
    """
    def __init__(self):
              
        self.nVisits = 0  
        self.u_value = 0
        self.Q_value = 0
        self.total_value_rewards = 0
        self.total_rollout_rewards = 0
        self.children = {} 


    def expandChildren(self, actions):
        """Expand subtree --a dictionary with a tuple of (x,y) position as keys, TreeNode object as values

        Keyword arguments:
        Output from policy function-- a list of tuples of (x, y) position and prior probability

        Return:
        None

        """
        for action in actions: 
            self.children[action[0]] = TreeNode()

    def selection(self):
        """Select among subtree to get the position that gives maximum action value Q plus bonus u(P)

        Keyword arguments:
        None. 

        Return:
        action -- a tuple of (x, y)
        treenode object
        

        """
        selectednode = self.children.values()[0]
        selectedaction = self.children.keys()[0]
        maxValue = selectednode.toValue()
                
        for child in self.children.items():
            if(child[1].toValue() > maxValue):
                selectednode = child[1]
                maxValue = child[1].toValue()
                selectedaction = child[0]
        return selectednode, selectedaction


                    
    def isLeaf(self):
        """Check if leaf state is reached
        """

        return self.children == {}

    def updateVisits(self):
        """Update the count of visit times 
        """
        self.nVisits += 1

    def updateQ_value(self):
        """Update the action value Q in two different ways:
        The leaf node is evaluated with the weighted average of value network 
        and outcome of a random rollout played out until terminal step T using the fast rollout policy
        The non-leaf node is evaluated by taking the mean value of all evaluations in the subtree below
        """

        if(self.isLeaf() == True):

            self.Q_value = (1 - LAMBDA) * (self.total_value_rewards / self.nVisits) + LAMBDA * (self.total_rollout_rewards / self.nVisits)

        else:

            self.Q_value = self.backupQ_value()
       
    def update_value_rewards(self, value):
        """Update leaf evaluations accumlated 
        """

        self.total_value_rewards += value


    def update_rollout_rewards(self, value):
        """Update rollout reward accumlated 
        """

        self.total_rollout_rewards += value

        

    def updateU_value(self, actions): 

        """Update the bonus value u(P)--proportional to the prior probability but decays with the number of visits to encourage exploration

        Keyword arguments:
        Output from policy function-- a list of tuples of (x, y) position and prior probability

        Return:
        None

        """

        for index in range(0, len(self.children)):  
            self.children[actions[index][0]].u_value = actions[index][1] / (1 + self.children[actions[index][0]].nVisits)

    def backupQ_value(self):

        """Track the mean value of evaluations in the subtrees

        Keyword arguments:
        value of traversed subtree evaluation each simulation

        Return:
        Mean value

        """
        total_value = 0
        for childnode in self.children.values():
            total_value += childnode.Q_value

        return total_value / len(self.children)


    def toValue(self):
        """Return action value Q plus bonus u(P)
        """
        return self.Q_value + self.u_value




class MCTS(object):
    """Monte Carlo tree search, takes an input of game state, value network function, policy network function, rollout policy function, outputs an action after lookahead search is complete.
    """

    def __init__(self, state, value_network, policy_network, rollout_policy):        
                
        self.state = state
        self.root = TreeNode()
        self._value = value_network
        self._policy = policy_network
        self._rollout = rollout_policy

  
    def DFS_in_tree_phase(self, L):
        """In-tree phase search of each simulation begins at the root of the search tree and finishes when the simulation reaches a leaf node at time step L.
        At each of these time steps, an action is selected according to the statistics in the search tree.

        Keyword arguments:
        Search Depth L. 

        Return:
        Leaf state and a list of visited TreeNode objects
        
        """

        visited = []
        state = self.state.copy()
        root = self.root

        while(L > 0):
            actions = self._policy(state)
            if(root.isLeaf() == True):
                root.expandChildren(actions)
            root.updateU_value(actions)
            selectedchild, selectedaction = root.selection()
            state.do_move(selectedaction)
            root = selectedchild
            root.updateVisits()
            visited.insert(0, root)
            L -= 1

            """The leaf state may be expaned by adding the successor state to the search tree, when the visit count of current leaf state exceeds a threshold.
            When the visit count exceeds a threshold, Nr(s, a) > nthr , the successor state is added to the search tree. 
            The new node is initialized, with a policy function to provide placeholder prior probabilities for action selection
        
            """
            if((L == 0) and (visited[0].nVisits > THRESHOLD)):
                L += 2

        return state, visited

    def value_evaluation(self, state, visited):
        """update value statistics with the output of value network of leaf state evaluation, then update Q_value in a backward pass
        through each step t<=L

        Keyword arguments:
        Leaf state and a list of visited TreeNode objects

        Return:
        None.
        """

        value = self._value(state)
        visited[0].update_value_rewards(value)
        self.backup(visited)

      
    def rollout_evaluation(self, state, visited):
        """update rollout statistics with the outcome of the game, then update Q_value in a backward pass
        through each step t<=L

        Keyword arguments:
        Leaf state and a list of visited TreeNode objects

        Return:
        None.
        """
        #rollout = self.DFS_rollout_phase(state)
        rollout = self._rollout(state)
        visited[0].update_rollout_rewards(rollout)
        self.backup(visited)


    def backup(self, visited):
        """Update overall evaluations of each visited state with a weighted average of Monte Carlo estimates

        Keyword arguments:
        List of visited TreeNode objects. 

        Return:
        None.
        """

        for treenode in visited:
            treenode.updateQ_value()


    def simulation(self, nSimulation, L):
        """ Run In-tree phase search, leaf state evaluations, and a backward pass update for a number of simulation times.
        
        Keyword arguments:
        In-tree phase search depth, number of simulation times.

        Return:
        None.
        """
        for index in range(0, nSimulation):
            state, visited = self.DFS_in_tree_phase(L)
            self.value_evaluation(state, visited)
            self.rollout_evaluation(state, visited)

   
    def DFS_rollout_phase(self, state):
        """ At each of these time-steps after L, actions are selected by both players according
        to the rollout policy. When the game reaches a terminal state, the outcome zt is computed from the final score
        from the perpective of current player.
        """
        pass


    def get_move(self):
        """At the end of search AlphaGo selects the action with maximum visit count
        this is less sensitive to outliers than maximizing action value

        Keyword arguments:
        None.

        Return:
        Action -- a tuple of (x, y)
        """


        selectednode = self.root.children.values()[0]
        selectedaction = self.root.children.keys()[0]
        maxCount = selectednode.nVisits
        
        for child in self.root.children.items():
            if(child[1].nVisits > maxCount):
                selectednode = child[1]
                maxCount = child[1].nVisits
                selectedaction = child[0]
        return selectedaction


    def make_move(self, move):
        """Update root position and starting state after a move being made, the rest of tree can be garbage collected
        """

        newroot = self.root.children[move]
        self.root = newroot
        self.state.do_move(move)




class ParallelMCTS(MCTS):
      pass


















