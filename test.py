from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np
from math import *

func = input("Podaj funkcję: ")
def input_function():
    # x = x[0]
    result = lambda x: eval(func)
    return result

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
    roulette_wheel_selection_losowa = randint(len(populacja))
    for i in range(0,len(populacja),k_hipherparameter-1):
        if values[i] > values[roulette_wheel_selection_losowa]:
            roulette_wheel_selection_losowa = i
    return populacja[roulette_wheel_selection_losowa]

# def single_point_cross(parent_a, parent_b, single_point_cross_hiperparametr):
#     child1, child2 = parent_a.copy(), parent_b.copy()
#     if rand() < single_point_cross_hiperparametr:
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

# if __name__ == "__main__":
#     b = input(">> ")
#     poi = get_function(b)
#     print(type(poi))
#     print(poi(2))

def algorytm_genetyczny(fun_operator ,ilosc_bitow,ilosc_iteracji,ilosc_populacji,single_point_cross_hiperparametr,mutate_hiperparametr):
    zadana_funkcja = fun_operator()
    
    populacja = [[randint(2) for i in range(len(boundries)*ilosc_bitow)] for j in range(ilosc_populacji)]
    print(populacja)
    najlepsze_wartosci, najlepsze_populacje = 0, zadana_funkcja(boundries_based_decode(ilosc_bitow,populacja[0]))
    for generacja in range(ilosc_iteracji):
        zdekodowana_populacja = [boundries_based_decode(ilosc_bitow,osobnik) for osobnik in populacja]
        wartosci = [zadana_funkcja(osobnik) for osobnik in zdekodowana_populacja]
        for i in range(ilosc_populacji):
            if wartosci[i] < najlepsze_populacje:
                najlepsze_wartosci, najlepsze_populacje = populacja[i], wartosci[i]
                print( f"Najlepsza wartość f({zdekodowana_populacja[i][0]}) = ", wartosci[i])
        wybrani_rodzice = [roulette_wheel_selection(populacja,wartosci) for i in range(ilosc_populacji)]
        potomstwo = list()
        for i in range(0,ilosc_populacji,2):
            rodzic1, rodzic2 = wybrani_rodzice[i], wybrani_rodzice[i+1]
            for dziecko in single_point_cross(rodzic1,rodzic2):
                mutate(dziecko,mutate_hiperparametr)
                potomstwo.append(dziecko)
        populacja = potomstwo
    plt.figure()
    plt.title(f"Algorytm genetyczny dla funkcji {func}")
  
    plt.plot(zdekodowana_populacja, wartosci, 'x', color='black')
    x_axis = np.arange(-8.0, 8.0, 0.1)

    # vfunc = np.vectorize(zadana_funkcja)

    # y_axis = vfunc(x_axis)
    y = []

    for x in x_axis:
        y.append(zadana_funkcja(x))


    plt.plot(x_axis, y)
    return najlepsze_wartosci, najlepsze_populacje


if __name__ == "__main__":
    # func = input("Podaj funkcję: ")
    ilosc_iteracji = 100
    ilosc_bitow = 16
    ilosc_populacji = 100
    krzyzowanie_hiperparametr = 0.9
    mutacja_hiperparametr = 1/(ilosc_bitow)
    najlepszy,wynik = algorytm_genetyczny(input_function,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
    print('Najlepszy wynik dla')
    decoded = boundries_based_decode(ilosc_bitow, najlepszy)
    print('f(%s) = %f' % (decoded[0], wynik))
