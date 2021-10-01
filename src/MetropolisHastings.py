import DatasetGenerator
from State import State
from BaseChain import HeatBathBaseChain
from random import uniform
import numpy as np
from tqdm import tqdm

class MetropolisHastings(object):
    def __init__(self, 
                base_chain, 
                beta, 
                initial_state, 
                beta_multiplicative_increase= 1,
                beta_iterations_increase= 100):
        """
        Parameters:
        base_chain: BaseChain class instance
        beta: parameter for the distribution
        initial_state: State class instance
        beta_multiplicative_increase: explained below
        beta_iterations_increase: after (beta_iterations_increase) iterations, beta is multiplied by (beta_multiplicative_increase)
        """
        self.base_chain = base_chain
        self.beta = beta
        self.initial_state = initial_state
        self.beta_multiplicative_increase = beta_multiplicative_increase
        self.beta_iterations_increase = beta_iterations_increase
        self.best_state = initial_state
        self.proposed_states = [] # [(state, true/false)], true if the state is accepted
        self.accepted_states = [] # [state]
        self.number_of_chosen_cities = [] # [int], number of the cities chosen in each step
        self.rejected_states = [] # [state]
        self.walk_executions = [] # [] multiple walks on the chain are storedi
 
    def get_pi_for_ratio(self, state):
        """
        calculates pi(state) without the denominator factor (denominator can not be 
        calculated easily and it is not needed for acceptance probabilities) 
        """
        return np.exp(self.beta * state.cost)

    def get_acceptance_probability(self, i, j):
        """
        calculates acceptance probability to make a move from the state i to the state j
        Parameters:
        i: state from
        j: state to
        """
        return min(1, self.get_pi_for_ratio(j) * self.base_chain.matrix(j,i)/ ( self.get_pi_for_ratio(i) * self.base_chain.matrix(i,j)) + 10e-5)

    def print_solution(self):
        """
        Prints the details about best state
        """
        print("\n\nSolution: chosen cities" + str(self.best_state.chosen_indices) + 
                " | cost: " + str(self.best_state.cost) + " | number of chosen cities: " + str(len(self.best_state.chosen_indices)))


    def perform_walk(self, iters= 1000, print_solution= True):
        """
        performs a walk on the final chain using base_chain and get_acceptance_probability
        stores the best solution to self.best_state
        Parameters:
        iters: number of proposed transitions in the chain
        """

        # cleaning up after previous walk
        self.best_state = self.initial_state
        self.proposed_states = [] 
        self.accepted_states = [] 
        self.number_of_chosen_cities = [] 
        self.rejected_states = [] 
        self.function_values = []
        
        i = self.initial_state
        self.best_function_value = i.cost
        for k in range(iters):
            # make a move in the base chain
            possible_j = self.base_chain.make_move(i)
            # calculate acceptance probability for transition from i to possible_j
            acceptance_probability = self.get_acceptance_probability(i, possible_j)
            is_move_accepted = (uniform(0,1) < acceptance_probability)
            self.proposed_states.append((possible_j, is_move_accepted))
            if is_move_accepted: # the move is accepted
                self.accepted_states.append(possible_j)
                i = possible_j
                if i.cost > self.best_state.cost: # the best state seen so far is stored
                    self.best_state = i
                    self.best_function_value = i.cost
                #print("chosen cities: " + str(i.chosen_indices) + " | cost: " + str(i.cost) + " | Best cost: " + str(self.best_function_value))
            else: #the move is not accepted
                self.rejected_states.append(possible_j) 
            self.number_of_chosen_cities.append(len(i.chosen_indices))
            self.function_values.append(i.cost)
            # simulated annealing beta increase after
            if k % self.beta_iterations_increase == 0:
                self.beta *= self.beta_multiplicative_increase
                if isinstance(self.base_chain, HeatBathBaseChain):
                    self.base_chain.beta = self.beta
        if print_solution:
            self.print_solution()
        return self.best_function_value
    

    @staticmethod
    def repeat_walks_for_fixed_parameters(base_chain, 
                    beta, 
                    original_initial_state,
                    iters = 1000, 
                    beta_multiplicative_increase= 1,
                    beta_iterations_increase= 100, 
                    repetition_number= 10, 
                    lambda_lower_limit= 0, 
                    lambda_upper_limit= 1, 
                    number_of_lambdas= 100,
                    dataset_generator = None,
                    algorithms = None):
        """
        Repeat the walk repetition_number times with given base chain, beta and initial_state
        store inform

        Parameters:
        base_chain: BaseChain class instance
        beta: parameter for the distribution
        original_initial_state: State class instance (lambda_ of this state will be ignored)
        iters: number of iterations in the walk
        beta_multiplicative_increase: explained below
        beta_iterations_increase: after (beta_iterations_increase) iterations, beta is multiplied by (beta_multiplicative_increase)
        dataset_generator: if it is necessary to change dataset every time, this should be passed
        repetition_number: how many times to repeat the walk (for fixed lambda_)

        Return:
        tuple (dictionary_cardinality, dictionary_function_value)
        where dictionary_cardinality[lambda] = mean of the cardinality of S* for fixed lambda
        where dictionary_function_value[lambda] = optimized function value for fixed lambda
        """
        # initialize lambdas
        lambdas = np.linspace(start= lambda_lower_limit, stop= lambda_upper_limit, num= number_of_lambdas)

        #means to be stored
        number_of_chosen_cities_means = {}
        function_values_means = {}
        for j, lambda_ in tqdm(enumerate(lambdas)):
            if dataset_generator == None:
                initial_state = State(original_initial_state.dataset, original_initial_state.is_chosen_point, lambda_)
            # initialization
            number_of_chosen_cities = np.zeros(repetition_number)
            function_values = np.zeros(repetition_number)
            # repeat the experiment repetition_number times for fixed lambda
            for i in range(repetition_number):
                if dataset_generator != None:
                    data = dataset_generator()
                    indices = np.argsort(data.v)[-2:][::-1]
                    is_chosen = np.zeros((100,), dtype = bool) 
                    is_chosen[indices[0]] = True
                    is_chosen[indices[1]] = True
                    initial_state = State(data, is_chosen, lambda_)
                # initialize MH algorithm
                algorithm = MetropolisHastings(base_chain, beta[j], initial_state, beta_multiplicative_increase[j],
                                               beta_iterations_increase)
                # perform MH algorithm
                algorithm.perform_walk(iters= iters, print_solution= False)
                # determinte the cardinality S*
                number_of_chosen_cities[i] = algorithm.best_state.number_of_chosen_points
                # determine the function value of S*
                function_values[i] = algorithm.best_function_value
                
            # store result
            number_of_chosen_cities_means[lambda_] = np.mean(number_of_chosen_cities)
            function_values_means[lambda_] = np.mean(function_values) 

        return (number_of_chosen_cities_means, function_values_means)
        

    @staticmethod
    def repeat_walks_for_custom_parameters(algorithms,
                    iters, 
                    repetition_number= 10, 
                    lambda_lower_limit= 0, 
                    lambda_upper_limit= 1, 
                    number_of_lambdas= 100):
        """
        Repeat the walk repetition_number times with given custom algorithm instances

        Parameters:
        algorithms: [MetropolisHastings instance] (list of MH initializations for the comparison)
        original_initial_state: State class instance (lambda_ of this state will be ignored)
        iters: [int], number of iterations in the walk
        repetition_number: how many times to repeat the walk (for fixed lambda_)

        Return:
        tuple (dictionary_cardinality, dictionary_function_value)
        where dictionary_cardinality[lambda] = mean of the cardinality of S* for fixed lambda
        where dictionary_function_value[lambda] = optimized function value for fixed lambda
        """

        # initialize lambdas
        lambdas = np.linspace(start= lambda_lower_limit, stop= lambda_upper_limit, num= number_of_lambdas)

        #means to be stored
        number_of_chosen_cities_means = {}
        function_values_means = {}
        for lambda_, index in enumerate(lambdas):
            # initialization
            number_of_chosen_cities = np.zeros(repetition_number)
            function_values = np.zeros(repetition_number)
            # repeat the experiment repetition_number times for fixed lambda
            for i in range(repetition_number):
                # initialize next MH algorithm instance (with custom parameters)
                algorithm = algorithms[index] # initialize algorithm with custom parameters
                # perform MH algorithm
                algorithm.perform_walk(iters= iters[index], print_solution= False)
                # determinte the cardinality S*
                number_of_chosen_cities[i] = algorithm.best_state.number_of_chosen_points
                # determine the function value of S*
                function_values[i] = algorithm.best_function_value
                
            # store result
            number_of_chosen_cities_means[lambda_] = np.mean(number_of_chosen_cities)
            function_values_means[lambda_] = np.mean(function_values) 

        return (number_of_chosen_cities_means, function_values_means)
    