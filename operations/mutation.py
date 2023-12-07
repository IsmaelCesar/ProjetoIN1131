
import numpy as np
import copy


def _apply_swap(inidividual: np.ndarray) -> np.ndarray:
    """
    Effectively apply swapped mutaiton to the indiviudal
    """
    index_range = list(range(len(inidividual)))
    choices = np.random.choice(index_range, 2)
    
    #swap
    temp = inidividual[choices[0]]
    inidividual[choices[0]] = inidividual[choices[1]]
    inidividual[choices[1]] = temp

    return inidividual

def _apply_scramble(individual: np.ndarray) -> np.ndarray:
    """
    Effectively applies scramble mutation to individual
    """
    start = np.random.randint(0, len(individual)-2)
    end = np.random.randint(start+1, len(individual)-1)

    np.random.shuffle(individual[start: end + 1])

    return individual

def _apply_inverse(individual: np.ndarray) -> np.ndarray:
    """
    Effectively aplies inverse mutation
    """

    start = np.random.randint(0, len(individual)-2)
    end = np.random.randint(start+1, len(individual)-1)

    individual[start: end + 1] = list(reversed(individual[start: end + 1]))

    return individual

class SingleTravelerMut:

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def swap(self, individual: np.ndarray) -> np.ndarray: 
        rand_prob = np.random.rand()
        if rand_prob < self.probability: 
            _apply_swap(individual)
        return copy.deepcopy(individual)
    
    def scramble(self, individual: np.ndarray) -> np.ndarray:
        rand_prob = np.random.rand()
        if rand_prob < self.probability: 
            _apply_scramble(individual)
        return copy.deepcopy(individual)

    def inverse(self, individual: np.ndarray) -> np.ndarray:
        rand_prob = np.random.rand()
        if rand_prob < self.probability: 
            _apply_inverse(individual)
        return copy.deepcopy(individual)
