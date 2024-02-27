import numpy as np
import cvxpy as cp
from collections import deque
from knapsack_prob import Generator
from queue import PriorityQueue

class Node:
    def __init__(self, instance, constraint_np, tree_path):
        self.constraint_np = constraint_np
        self.solved = 0
        self.instance = instance
        self.tree_path = tree_path
        self._solve()
        
    def _solve(self):
        if self.solved:
            print('solved')
        else:
            instance = self.instance
            # initiate problem from instance
            x = cp.Variable(instance.num_items)
            objective = cp.Maximize(instance.value_items @ x)
            constraint_bag_capacity = [instance.weight_items @ x<=instance.bag_capcity]
            
            prob = cp.Problem(objective, constraint_bag_capacity+[x<=self.constraint_np[0],x>=self.constraint_np[1]])
            prob.solve()
        
            self.obj_value = objective.value
            self.solution = x.value
            self.solved = 1
    
    def _branching(self):
        x = self.solution
        variable_postions = np.argwhere(np.abs(x-np.round(x))>=1e-5) # to do tolerance
        variable_postions = variable_postions.flatten()
        if len(variable_postions) != 0: # fractional values' positions
            return variable_postions # position to branch
        # no fractional values
        self.solution = np.rint(x)
        self.obj_value = self.instance.value_items@self.solution
        return None
    
    def __repr__(self):
        return f'Node:{repr(self.solution)}'
        
    @classmethod
    def bounding(cls, j, parent_node, type_bound):
        i = 0 if type_bound == 'upper' else 1
        constraint_np = parent_node.constraint_np.copy()
        constraint_np[i,j] = i
        return cls(parent_node.instance, constraint_np, parent_node.tree_path +[(j,type_bound)])

class TreeNode(Node):
    def __gt__(self, other):
        return self.obj_value - other.obj_value > 1e-3

    def __eq__(self, other):
        return np.abs(self.obj_value - other.obj_value)<= 1e-3

    
class BranchAndBound:
    
    def __init__(self, verbose=0) -> None:
        super().__init__()
        self.verbose = verbose
    
    def _solve(self, instance):
        # clean the incumbent solution and constraints tracking
        incumbent_node = None
        branching_queue = PriorityQueue()
        
        # np array 2xn the upper bound and lower bound for constraints
        init_contraints_np = np.stack( (np.ones(instance.num_items),
                                    np.zeros(instance.num_items)))
        root = TreeNode(instance, init_contraints_np,[])
        
        branching_queue.put(root)
        
        while not branching_queue.empty():
            
            node = branching_queue.get()
        
            # if left branches has smaller upper bound
            if incumbent_node is not None\
                        and node.obj_value <= incumbent_node.obj_value:
                break
            branching_rst = node._branching()
            
            if branching_rst is None: # update incumbednt
                if self.verbose:
                        print(f'end at the leaf: {node.tree_path}')
                if incumbent_node is None or (node.obj_value +1e-3 >= incumbent_node.obj_value):
                    if self.verbose:
                        old_value = None if incumbent_node is None\
                                            else incumbent_node.obj_value
                        print('     update incumbent solution' +
                            f'for better obj value: {old_value}->{node.obj_value}')
                        print(f'           solution: {node}')
                    incumbent_node = node
                else:
                    print('       no updates')
            else: # branching 
                
                for j in branching_rst:
                    new_node = TreeNode.bounding(j, parent_node=node, type_bound='upper')
                    a = new_node.obj_value
                    if a is None:
                        a = 'infeasible'
                    elif incumbent_node is None or \
                    new_node.obj_value+1e-3>= incumbent_node.obj_value:
                        branching_queue.put(new_node)
                    else:
                        a = 'f'
                    new_node = TreeNode.bounding(j, parent_node=node, type_bound='lower')
                    b = new_node.obj_value
                    if b is None:
                        b = 'infeasible'
                    elif incumbent_node is None or \
                    new_node.obj_value+1e-3>= incumbent_node.obj_value:
                        branching_queue.put(new_node)
                    else:
                        b = 'f'
                    if self.verbose:
                        print(f'Node {node.tree_path} branching on x_{j}')
                        print( f'              upper -> {a}')
                        print( f'              lower -> {b}')
                    
        return incumbent_node
    
if __name__ == '__main__':
    ks_generator = Generator()
    new_instance = ks_generator.random_instance()
    solver = BranchAndBound(verbose=1)
    solver._solve(new_instance)
    
    