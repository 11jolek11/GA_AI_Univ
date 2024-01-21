from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np
from math import *


def get_function(content: str):
    def generated_function(x):
        return eval(content)
    return generated_function


boundries = (-8, 8)

def boundries_based_decode(bit_len,encoded):
    decoded = list()
    for i in range(len(boundries)):
        start,stop = i*bit_len,(i+1)*bit_len
        substring = encoded[start:stop]
        chars = ''.join([str(i) for i in substring])
        integer = int(chars,2)
        scaled = boundries[0]+(integer/((2**bit_len)-1))*(boundries[1]-boundries[0])
        decoded.append(scaled)
    return decoded

def roulette_wheel_selection(populacja,values, k_hipherparameter=3):
    selekcja_losowa = randint(len(populacja))
    for i in range(0,len(populacja),k_hipherparameter-1):
        if values[i] > values[selekcja_losowa]:
            selekcja_losowa = i
    return populacja[selekcja_losowa]

# def single_point_cross(parent_a, parent_b, krzyzowanie_hiperparametr):
#     child1, child2 = parent_a.copy(), parent_b.copy()
#     if rand() < krzyzowanie_hiperparametr:
#         cross_pointer = randint(1, len(parent_a) - 2)
#         child1 = parent_a[:cross_pointer] + parent_b[cross_pointer:]
#         child2 = parent_b[:cross_pointer] + parent_a[cross_pointer:]
#     return [child1,child2]

def single_point_cross(parent_a, parent_b):
    child1, child2 = parent_a.copy(), parent_b.copy()

    cross_pointer = randint(1, len(parent_a) - 1)
    child1 = parent_a[:cross_pointer] + parent_b[cross_pointer:]
    child2 = parent_b[:cross_pointer] + parent_a[cross_pointer:]

    return child1, child2

def double_point_cross(parent_a, parent_b):
    child1, child2 = parent_a.copy(), parent_b.copy()
    for _ in range(2):
        cross_pointer = randint(1, len(parent_a) - 1)
        child1 = parent_a[:cross_pointer] + parent_b[cross_pointer:]
        child2 = parent_b[:cross_pointer] + parent_a[cross_pointer:]

    return child1, child2

def mutate(encoded, mutation_probability=0.01):
    for i in range(len(encoded)):
        if rand() < mutation_probability:
            encoded[i] = 1 - encoded[i]

if __name__ == "__main__":
    b = input(">> ")
    poi = get_function(b)
    print(type(poi))
    print(poi(2))
