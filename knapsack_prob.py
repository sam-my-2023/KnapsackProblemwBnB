from collections import namedtuple

import numpy as np
import pickle
import cvxpy as cp

KS_Instance = namedtuple('KS_Instance', ['num_items',
                                         'bag_capcity',
                                         'weight_items',
                                         'value_items'])

class Generator():
    '''
    Knapsack problem: Given a bag which can carry limited weight,
                    Also given a bunch of items, every item has a value and a weight,
                    Select items to put them into your bag,
                    The goal is to get the maximal value sum for the bag, but the weights sum does not exceed the capacity.
    parameters:
        num_items_range: pair, lower bound and upder bound for number of items to generate
        bound_weight_items: pair, bounds for the weight of all items 
        bound_value_items: pair, bounds for values
        integer: boolean, flag if values and weights is restricted to be integer

    method:
        random_instance: generate an instance using the given bounds
    '''
    def __init__(self, num_items_range=(4,10),
                 bound_weight_items=(1,10), 
                 bound_value_items=(1,10), 
                 integer=True) -> None:
        self.num_range = num_items_range
        self.bound_weight_items = bound_weight_items
        self.bound_value_items = bound_value_items
        self.integer_flag = integer

    def random_instance(self):
        num_items = np.random.randint(*self.num_range)
        if self.integer_flag:
            weights = np.random.randint(*self.bound_weight_items, size = num_items)
            values = np.random.randint(*self.bound_value_items, size = num_items)
            capcity = np.random.randint(min(weights),sum(weights))
        else:
            weights = np.random.rand(num_items)*(self.bound_weight_items[1]-self.bound_weight_items[0])
            values = np.random.rand(num_items)*(self.bound_value_items[1]-self.bound_value_items[0])
            capcity = (sum(weights)-min(weights))*np.random.rand()+min(weights)
        return KS_Instance(num_items,
                    capcity,
                    weights,
                    values)

if __name__ == '__main__':
    ks_generator = Generator()
    new_instance = ks_generator.random_instance()
    print('========================instnace=====================')
    print(new_instance)
    print('========================solution=====================')
    x = cp.Variable(new_instance.num_items)
    objective = cp.Maximize(new_instance.value_items @ x)
    constraints_binary = [0 <= x, x <= 1]
    constraint_bag_capacity = [new_instance.weight_items @ x]

    prob = cp.Problem(objective, constraints_binary+constraint_bag_capacity)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    print(x.value)