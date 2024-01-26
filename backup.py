from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np
from math import *
from statistics import *

func = input("Podaj funkcję: ")
def input_function(x):
    x = x[0]
    result = eval(func)
    return result

boundries = (0, 16)

def boundries_based_decode(bit_len: int,encoded: list):
    decoded = list()
    for i in range(len(boundries)):
        start,stop = i*bit_len,(i+1)*bit_len
        substring = encoded[start:stop]
        chars = ''.join([str(i) for i in substring])
        integer = int(chars,2)
        scaled = boundries[0]+(integer/((2**bit_len)-1))*(boundries[1]-boundries[0])
        decoded.append(scaled)
    return decoded

# NOTE: testuje kodowanie logarytmiczne
def log_based_decode(bit_len, chromosome):
    boundries = (-8.0, 8.0)
    min_value, max_value = boundries[0], boundries[1]
    decimal_value = int(''.join(map(str, chromosome)), 2)
    
    mapped_value = min_value * (max_value / min_value) ** (decimal_value / (2**len(chromosome) - 1))
    
    return mapped_value

def log_based_encode(value, min_value, max_value, num_bits):
    decimal_value = int((2**num_bits - 1) * (np.log(value / min_value) / np.log(max_value / min_value)))
    
    binary_representation = list(format(decimal_value, f'0{num_bits}b'))
    
    return binary_representation

def roulette_wheel_selection(populacja,values, k_hipherparameter=3):
    roulette_wheel_selection_losowa = randint(len(populacja))
    for i in range(0, len(populacja),k_hipherparameter-1):
        if values[i]/sum(values) > values[roulette_wheel_selection_losowa]:
            roulette_wheel_selection_losowa = i
    return populacja[roulette_wheel_selection_losowa]

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

def selekcja(populacja,wartosci,k_hipherparametr=3):
    selekcja_losowa =randint(len(populacja))
    for i in range(0,len(populacja),k_hipherparametr-1):
        if wartosci[i] > wartosci[selekcja_losowa]:
            selekcja_losowa = i
    return populacja[selekcja_losowa]

def stop_condition(evaluated, epsilon, old_population):
    print(f"{mean(old_population)} - {mean(evaluated)}")
    if abs(mean(evaluated) - mean(old_population)) <= epsilon:
        return True, mean(evaluated)

    return False, mean(evaluated)

def roulette_wheel_selection(populacja, values, k_hipherparameter=3):
    roulette_wheel_selection_losowa = randint(len(populacja))
    for i in range(0, len(populacja), k_hipherparameter-1):
        if values[i]/sum(values) > values[roulette_wheel_selection_losowa]:
            roulette_wheel_selection_losowa = i
    return populacja[roulette_wheel_selection_losowa]



def algorytm_genetyczny(zadana_funkcja,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,mutacja_hiperparametr):
    populacja = [[randint(2) for _ in range(len(granice)*ilosc_bitow)] for _ in range(ilosc_populacji)]

    sorted(populacja, key=lambda individual: log_based_decode(ilosc_bitow, individual) , reverse=True)

    najlepsze_wartosci = 0
    najlepsze_populacje = zadana_funkcja(log_based_decode(ilosc_bitow, populacja[0]))

    stare_wartoci = [0 for _ in range(ilosc_populacji)]
    
    for _ in range(ilosc_iteracji):
        print("iter")
        zdekodowana_populacja = [log_based_decode(ilosc_bitow,osobnik) for osobnik in populacja]
        wartosci = [zadana_funkcja(osobnik) for osobnik in zdekodowana_populacja]
        
        for i in range(ilosc_populacji):
            if wartosci[i] < najlepsze_populacje:
                najlepsze_wartosci, najlepsze_populacje = populacja[i], wartosci[i]
                print( f"Best value at {zdekodowana_populacja[i][0]} is {wartosci[i]} ")
        
        stop_signal, _ = stop_condition(wartosci, 1e-37, stare_wartoci)
        if stop_signal:
            pass
            print(">> STOP CONDITION")
            # break

        # wybrani_rodzice = [roulette_wheel_selection(populacja,wartosci) for i in range(ilosc_populacji)]
        wybrani_rodzice = [selekcja(populacja,wartosci) for i in range(ilosc_populacji)]
        
        potomstwo = list()
        
        for i in range(0,ilosc_populacji,2):
            rodzic1, rodzic2 = wybrani_rodzice[i], wybrani_rodzice[i+1]
            for dziecko in double_point_cross(rodzic1,rodzic2):
                mutate(dziecko, mutacja_hiperparametr)
                potomstwo.append(dziecko)
        populacja = potomstwo
        # zdekodowana_potomstwo = [log_based_decode(ilosc_bitow,osobnik) for osobnik in potomstwo]
        # stare_wartoci = [zadana_funkcja(osobnik) for osobnik in zdekodowana_potomstwo]
    
    
    plt.figure()
    plt.title(f"Funkcja {func}")
  
    x_axis = np.arange(granice[0],granice[1],0.1)

    # x = np.arange(granice[0],granice[1],0.1)
    # FIXME(11jolek11): eval is not ok
    full_func = eval("lambda x:" + func)
    vfunc = np.vectorize(full_func)
    plt.plot(x_axis, vfunc(x_axis))
    plt.plot(zdekodowana_populacja,wartosci,'1',color='red')
    # plt.savefig("./images/test.jpg")
    plt.show()
    return najlepsze_wartosci, najlepsze_populacje


granice = (0,16)
ilosc_iteracji = 100
ilosc_bitow = 64
ilosc_populacji = 100
krzyzowanie_hiperparametr = 0.9
mutacja_hiperparametr = 0.01
# mutacja_hiperparametr = 0.1
# mutacja_hiperparametr = 0.001

najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,mutacja_hiperparametr)
print('Najlepszy wynik dla')
decoded = log_based_decode(ilosc_bitow, najlepszy)
print('f(%s) = %f' % (decoded[0], wynik))
