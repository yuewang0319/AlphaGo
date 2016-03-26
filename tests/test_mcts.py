from AlphaGo.go import GameState
from AlphaGo.mcts import MCTS
from AlphaGo.mcts import TreeNode
import random
import numpy as np
import unittest


class TestMCTS(unittest.TestCase):
    
    def setUp(self):
        self.s = GameState()
        self.mcts = MCTS(self.s, value_network, policy_network, rollout_policy)
        self.treenode = TreeNode()


    def test_treenode_selection(self):
        actions = self.mcts._policy(self.s)
        self.treenode.expandChildren(actions)
        self.treenode.updateU_value(actions)
        selectednode, selectedaction = self.treenode.selection()
        self.assertEqual(max(actions, key = lambda x:x[1])[1], selectednode.toValue(), 'incorrect node selected')
        self.assertEqual(max(actions, key = lambda x:x[1])[0], selectedaction, 'incorrect action selected')

    
    def test_mcts_DFS_in_tree_phase(self):
        state, visited = self.mcts.DFS_in_tree_phase(20) 
        for treenode in visited:
            self.assertEqual(1, treenode.nVisits, 'incorrect visit count')
 

    def test_mcts_get_move(self):
        self.mcts.simulation(10, 5)
        action = self.mcts.get_move()
        self.assertIsNotNone(action,'no output action') 
        


def policy_network(state):
    moves = state.get_legal_moves()
    actions = []
    for move in moves:
        actions.append((move, random.uniform(0, 1)))                              
    return actions 
   
def value_network(state):

    return 0.5

def rollout_policy(state):

    return 1

if __name__ == '__main__':
    unittest.main()


