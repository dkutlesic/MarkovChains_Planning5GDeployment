from abc import ABC,abstractmethod
import numpy as np
from random import choices
from State import State

class BaseChain(ABC):
    """
    needs to be aperiodic, irreducible and (i,j)>0 iff (j,i)>0
    """
    @abstractmethod
    def matrix(self, i, j):
        """
        returns a probability for a transition from state i to state j in the base chain
        Parameters:
        i: state from
        j: state to
        """
        pass

    @abstractmethod
    def get_neighbors(self, state):
        """
        returns all neighbors of a given state (as a list of states)
        Parameters:
        state: state which neighbors are calculated
        """
        pass

    def make_move(self, state):
        """
        makes a move on the chain from the current state
        Parameters:
        state: current state
        """
        neighbors = self.get_neighbors(state)
        neighbor_probabilities = []
        for neighbor in neighbors:
            neighbor_probabilities.append(self.matrix(state, neighbor))
        return choices(neighbors, weights= neighbor_probabilities)[0]
    

class DummyBaseChain(BaseChain):
    """
    two states are neighbors if they are different only in one bit (in the state representation)
    initial state is #TODO
    all transitions are equally probable

    """
    def matrix(self, i, j):
        if i == j or np.sum(np.logical_xor(i.is_chosen_point, j.is_chosen_point)) == 1:
            return 1 / (i.dataset.N + 1)
        else:
            return 0

    def get_neighbors(self, state):
        neighbors = []
        neighbors.append(State(state.dataset, state.is_chosen_point, state.lambda_))
        for bit in range(state.dataset.N):
            neighbors.append(state.get_new_state_flip_ith_bit(bit))
        return neighbors
    

class TwoDRandomWalk(BaseChain):
    """
    state: (i,j), where i,j are index numbers of 2 cities and i<j<N
    neighbors of the state (i,j): (i-1, j), (i+1, j), (i, j-1), (i, j+1)
    if the state don't satisfy condition i<j<N, then the corresponding transition is self-loop
    initial state is 2 the most populated cities
    all transitions are equally probable
    """
    def matrix(self, i, j):
        return 1 / 4
        
    def get_neighbors(self, state):
        neighbors = []
        (i, j) = state.most_distant_cities#np.nonzero(state.is_chosen_point==True)[0]
        if (i-1 >= 0):
            new_state = state.get_new_state_2dRandomWalk(i-1, j)
        else:
            new_state = state.get_new_state_2dRandomWalk(i, j)
        neighbors.append(new_state)

        if (i+1 < j):
            new_state = state.get_new_state_2dRandomWalk(i+1, j)
        else: 
            new_state = state.get_new_state_2dRandomWalk(i, j)
        neighbors.append(new_state)

        if (i < j-1):
            new_state = state.get_new_state_2dRandomWalk(i, j-1)
        else: 
            new_state = state.get_new_state_2dRandomWalk(i, j)
        neighbors.append(new_state)

        if (j+1 < state.dataset.N):
            new_state = state.get_new_state_2dRandomWalk(i, j+1)
        else: 
            new_state = state.get_new_state_2dRandomWalk(i, j)
        neighbors.append(new_state)

        return neighbors

    
class HeatBathBaseChain(BaseChain):
    """
    Each next state is chosen by taking uniformly at random
    one city and changing it state. Unlike Dummy chain it has 
    transition probabilities depending on the objective function
    and resulting in acceptance probabilities that are equal to 1 
    """
    def __init__(self, beta):
        self.beta = beta
        
    def matrix(self, i, j):
        if i == j:
            prob_ii = 0
            for neighbor in self.neighbors[1:]:
                prob_ii += 1 / (1 + np.exp(-self.beta * (neighbor.cost - i.cost)))
            prob_ii /= i.dataset.N
            return prob_ii
        else:
            return 1 / (i.dataset.N * (1 + np.exp(-self.beta * (j.cost - i.cost))))

    def get_neighbors(self, state):
        neighbors = [state]
        for bit in range(state.dataset.N):
            neighbors.append(state.get_new_state_flip_ith_bit(bit))
        self.neighbors = neighbors
        return neighbors



class ImprovedBaseChain(BaseChain):
    """
    state: (i,j), where i,j are index numbers of 2 cities and i<j<N
    neighbors of the state (i,j): (0,j), ..., (i-1, j); (i+1, j),...,(j-1,j); (i,i+1), ..., (i, j-1); (i, j+1), ..., (i, N-1)
    if the state don't satisfy condition i<j<N, then the corresponding transition is self-loop
    initial state is 2 the most populated cities
    for given state (i,j) all transitions are equally probable and depend on i and j
    """
    def matrix(self, i, j):
        (d1, d2) = i.most_distant_cities
        return 1 / (2*(d2-d1-1)+d1+(i.dataset.N-d2)-1)
        
    def get_neighbors(self, state):
        neighbors = []
        (i, j) = state.most_distant_cities#np.nonzero(state.is_chosen_point==True)[0]
        new_state = state.get_new_state_2dRandomWalk(i, j)
        neighbors.append(new_state)

        k=i
        if i==0:
            new_state = state.get_new_state_2dRandomWalk(i, j)
            neighbors.append(new_state)
        while (k-1 >= 0):
            new_state = state.get_new_state_2dRandomWalk(k-1, j)
            k -= 1
            neighbors.append(new_state)

        k=i
        if i+1==j:
            new_state = state.get_new_state_2dRandomWalk(i, j)
            neighbors.append(new_state)
        while (k+1 < j):
            new_state = state.get_new_state_2dRandomWalk(k+1, j)
            k += 1
            neighbors.append(new_state)

        k=j
        if i==j-1:
            new_state = state.get_new_state_2dRandomWalk(i, j)
            neighbors.append(new_state)
        while (i < k-1):
            new_state = state.get_new_state_2dRandomWalk(i, k-1)
            k -= 1
            neighbors.append(new_state)

        k=j
        if j+1 == state.dataset.N:
            new_state = state.get_new_state_2dRandomWalk(i, j)
            neighbors.append(new_state)
        while (k+1 < state.dataset.N):
            new_state = state.get_new_state_2dRandomWalk(i, k+1)
            k += 1
            neighbors.append(new_state)

        return neighbors