from abc import ABC,abstractmethod
import scipy.stats as st
from sklearn.metrics import pairwise_distances
import numpy as np

class DatasetGenerator(ABC):
    @abstractmethod
    def __init__(self, N=100):
        self.N = N #number of datapoints
        self.x = None #to be specified in subclass
        self.v = None #to be specified in subclass
        self.load_dataset()
        #calculates self.distance_matrix
        self.calculate_distances_between_cities()
        self.calculate_covering_cities()

    def __getitem__(self, key):
        if key < self.N and key >= 0:
            return (self.x[key], self.v[key])
        return None
  
    @abstractmethod
    def load_dataset(self):
        """
        load specific dataset
        """
        pass

    def calculate_distances_between_cities(self):
        """
        Calculates Eucledean distance between every two cities (calculated only when the data is created)
        """
        self.distance_matrix = pairwise_distances(self.x)

    def calculate_covering_cities(self):
        """
        calculates covering cities dictionary of dictionaries (precomputed before the algorithm)
        self.covering_cities[i][j] = {all cities that are in the circle
                with dimeter dist(city[i], city[j]) and the center in the middle
                between city[i] and city[j]}
        """
        self.covering_cities = {}
        for i, city1 in enumerate(self.x):
            dict_for_city = {}
            for j, city2 in enumerate(self.x):
                circle_radius = np.linalg.norm(city1 - city2) / 2
                circle_center = (city1 + city2) / 2
                #1e-8 is added to be sure to include cities on the border of the circle
                dict_for_city[j] = np.where(np.array([np.linalg.norm(point - circle_center) for point in self.x]) < circle_radius + 1e-8)[0]
            self.covering_cities[i] = dict_for_city

class G1(DatasetGenerator):
    def __init__(self):
        super(G1, self).__init__()

    def load_dataset(self):
        """
        generating dataset from the task 2
        v comes from U[0,1], x comes from U[0,1]xU[0,1]
        """
        self.x = st.uniform().rvs((self.N,2))
        self.v = st.uniform().rvs((self.N,))

class G2(DatasetGenerator):
    def __init__(self):
        super(G2, self).__init__()

    def load_dataset(self):
        """
        generating dataset from the task 3
        v comes from e^N(-0.85, 1.3), x comes from U[0,1]xU[0,1]
        """
        self.x = st.uniform().rvs((self.N,2))
        self.v = np.exp(st.norm(-0.85, 1.3).rvs((self.N,)))
