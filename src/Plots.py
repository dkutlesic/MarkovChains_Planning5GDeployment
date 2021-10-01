import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_state(state, figsize=(6, 6), figname=None):
    '''
    Plots a given state. Chosen cities are red, others are blue.
    The size of scatters is proportional to its population.
    The big blue circle shows the occupied area.
    '''
    _, ax = plt.subplots(figsize=figsize)
    dataset = state.dataset
    chosen_cities = state.is_chosen_point
    not_chosen_cities = np.invert(chosen_cities)
    scaled_v = 50 * dataset.v / dataset.v.max() #scaling to have proper sizes of points
    ax.scatter(dataset.x[not_chosen_cities, 0], dataset.x[not_chosen_cities, 1], s=scaled_v[not_chosen_cities], c='b', edgecolors='k', label='Not selected cities')
    if np.sum(state.is_chosen_point) != 0:
        ax.scatter(dataset.x[chosen_cities, 0], dataset.x[chosen_cities, 1], s=scaled_v[chosen_cities], c='r', edgecolors='k', label='Selected cities')
        # Finding the occupied area
        chosen_cities_dists = np.tril(dataset.distance_matrix[chosen_cities][:, chosen_cities])
        diameter = chosen_cities_dists.max()
        furthest_cities = np.array(np.where(np.tril(dataset.distance_matrix) == diameter)).reshape(-1)
        area_center = 0.5 * (dataset.x[furthest_cities[0]] + dataset.x[furthest_cities[1]])
        circle = mpatches.Circle(area_center, diameter / 2, alpha=0.2)
        ax.add_patch(circle)
    plt.legend()
    if figname != None:
        plt.savefig(figname + '.png')
    plt.show()
    
def plot_series(costs, x=None, param_names=None, figname=None):
    '''
    Takes a set of algorithm runs and plot its cost function on one figure.
    Basically, it can be used for plotting individual terms of the plot function 
    or the number of cities.
    Parameters:
        costs: array-like
            The shape is (num_runs, num_iters)
        x: array-like
            The shape is (num_runs, num_iters)
        param_names: array-like
            The shape is (num_runs,) with str type
        figname: str
            A name for plot saving
    '''
    assert len(costs.shape) == 2, 'The shape of the argument must be (num_runs, num_iters)'
    for i, one_run in enumerate(costs):
        if param_names != None:
            label = param_names[i]
        else:
            label=str(i)
        if np.all(x == None):
            plt.plot(one_run, label=label)
        else:
            plt.plot(x[i], one_run, label=label)
    plt.grid(True)
    plt.legend()
    if figname != None:
        plt.savefig(figname + '.png')
    plt.show()