import numpy as np

def init_state(data, lambda_):
    '''
    Initializes a state
    Parameters:
        data: array-like
            A generated dataset
        lambda_: float
            A parameter in the target functio.
    Return:
        state: State
            An instance of class State
    '''
    indices = np.argsort(data.v)[-2:][::-1]
    # Choose which cities are included in initial solution
    is_chosen = np.zeros((100,), dtype = bool) # true if a city is included in a list
    is_chosen[indices[0]] = True
    is_chosen[indices[1]] = True
    state = State(data, is_chosen, lambda_)
    return state

class State(object):
    """
    Represents the state of Metropolis Hasting algorithm. State is defined by dataset (v and x), 
    boolean array is_chosen_point that represent is the point included in the state and lambda_ is the objective function parameter
    """
    def __init__(self, dataset, is_chosen_point, lambda_= 1):
        self.dataset = dataset #the whole dataset (instance of DatasetGenerator subclass)
        self.is_chosen_point = is_chosen_point #true/false array
        self.lambda_ = lambda_ #the parameter from the description
        self.chosen_indices = np.where(is_chosen_point)[0] #this is S in the project description
        self.number_of_chosen_points = np.sum(self.is_chosen_point)
        self.chosen_x = np.array([self.dataset.x[i] for i in range(self.dataset.N) if self.is_chosen_point[i]])
        self.chosen_v = np.array([self.dataset.v[i] for i in range(self.dataset.N) if self.is_chosen_point[i]])
        self.calculate_cost()

    def __eq__(self, other):
        if isinstance(other, State):
            return self.dataset == other.dataset and (self.is_chosen_point == other.is_chosen_point).all() and self.lambda_ == other.lambda_
        return False

    def calculate_maximum_distance_square(self):
        """
        calculates max d(x_i, x_j)^2 over all city-pairs from the chosen_indices
        this is calculated only once (when the state is created)
        """
        self.maximum_distance_square = 0
        self.most_distant_cities = (0,0) #indices of most distant cities
        for i in self.chosen_indices:
            for j in self.chosen_indices:
                current_distance_square = self.dataset.distance_matrix[i][j]**2
                if current_distance_square > self.maximum_distance_square:
                    self.maximum_distance_square = current_distance_square
                    self.most_distant_cities = (i, j)
                    if i > j:
                        self.most_distant_cities = (j, i)
                    else:
                        self.most_distant_cities = (i, j)
        
    def calculate_cost(self):
        """
        cost of a state
        this is calculated when the state is created
        """
        self.calculate_maximum_distance_square()
        self.cost = (np.sum(self.chosen_v) - self.lambda_ * self.number_of_chosen_points * np.pi * self.maximum_distance_square / 4)

    def get_new_state_flip_ith_bit(self, i):
        """
        Creating neighboring states for Dummy Chain
        """
        is_chosen_point = np.copy(self.is_chosen_point)
        is_chosen_point[i] = not is_chosen_point[i]
        return State(self.dataset, is_chosen_point, self.lambda_)
    
    def get_new_state_2dRandomWalk(self, i, j):
        """
        Creating neighboring states for 2dRandomWalk Chain
        """
        is_chosen_point = np.zeros_like(self.is_chosen_point)
        is_chosen_point[self.dataset.covering_cities[i][j]] = True
        return State(self.dataset, is_chosen_point, self.lambda_)

    
