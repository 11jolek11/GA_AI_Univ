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

#FIXME(11jolek11): No normalization of probab
def roulette_wheel_selection(populacja,values, k_hipherparameter=3):
    roulette_wheel_selection_losowa = randint(len(populacja))
    for i in range(0,len(populacja),k_hipherparameter-1):
        if values[i] > values[roulette_wheel_selection_losowa]:
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

# TODO(11jolek11): add stop condition to algorytm genetyczny
def stop_condition(evaluated, epsilon, old_population):
    if abs(mean(evaluated) - mean(old_population)) <= epsilon:
        return True, mean(evaluated)

    return False, mean(evaluated)

def algorytm_genetyczny(zadana_funkcja,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr):
    populacja = [[randint(2) for _ in range(len(granice)*ilosc_bitow)] for _ in range(ilosc_populacji)]

    print(populacja)

    najlepsze_wartosci = 0
    najlepsze_populacje = zadana_funkcja(boundries_based_decode(ilosc_bitow, populacja[0]))

    stare_wartoci = [0 for _ in range(ilosc_populacji)]
    
    for _ in range(ilosc_iteracji):
        zdekodowana_populacja = [boundries_based_decode(ilosc_bitow,osobnik) for osobnik in populacja]
        wartosci = [zadana_funkcja(osobnik) for osobnik in zdekodowana_populacja]
        
        for i in range(ilosc_populacji):
            if wartosci[i] < najlepsze_populacje:
                najlepsze_wartosci, najlepsze_populacje = populacja[i], wartosci[i]
                print( f"Best value at {zdekodowana_populacja[i][0]} is {wartosci[i]} ")
        
        stop_signal, _ = stop_condition(wartosci, 1e-6, stare_wartoci)
        if stop_signal:
            print(">> STOP CRITERION")
            break

        wybrani_rodzice = [roulette_wheel_selection(populacja,wartosci) for i in range(ilosc_populacji)]
        
        potomstwo = list()
        
        for i in range(0,ilosc_populacji,2):
            rodzic1, rodzic2 = wybrani_rodzice[i], wybrani_rodzice[i+1]
            for dziecko in single_point_cross(rodzic1,rodzic2):
                mutate(dziecko,mutacja_hiperparametr)
                potomstwo.append(dziecko)
        populacja = potomstwo
        zdekodowana_potomstwo = [boundries_based_decode(ilosc_bitow,osobnik) for osobnik in potomstwo]
        stare_wartoci = [zadana_funkcja(osobnik) for osobnik in zdekodowana_potomstwo]
    
    
    plt.figure()
    plt.title(f"Wybrana funkcja {func}")
  
    plt.plot(zdekodowana_populacja,wartosci,'x',color='green')
    x_axis = np.arange(granice[0],granice[1],0.1)

    x = np.arange(granice[0],granice[1],0.1)
    plt.plot(x_axis,eval(func))
    return najlepsze_wartosci, najlepsze_populacje


granice =[-8,8]
ilosc_iteracji = 100
ilosc_bitow = 16
ilosc_populacji = 100
krzyzowanie_hiperparametr = 0.9
mutacja_hiperparametr = 1/(ilosc_bitow)
najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
print('Najlepszy wynik dla')
decoded = boundries_based_decode(ilosc_bitow, najlepszy)
print('f(%s) = %f' % (decoded[0], wynik))




















# granice =[-8,8]
# ilosc_iteracji_tab = [10,50,100,200,500]
# ilosc_bitow = 16
# ilosc_populacji = 100
# krzyzowanie_hiperparametr = 0.9
# mutacja_hiperparametr = 1/(ilosc_bitow*len(granice))
# for ilosc_iteracji in ilosc_iteracji_tab:
#     najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
#     print(f'Najlepszy wynik dla iteracji {ilosc_iteracji}')
#     decoded = boundries_based_decode(granice, ilosc_bitow, najlepszy)
#     print('f(%s) = %f' % (decoded[0], wynik))

# granice =[-8,8]
# ilosc_iteracji = 100
# ilosc_bitow = 16
# ilosc_populacji = 100
# krzyzowanie_hiperparametr_tab = [0.1,0.3,0.5,0.7,0.9]
# mutacja_hiperparametr = 1/(ilosc_bitow*len(granice))
# for krzyzowanie_hiperparametr in krzyzowanie_hiperparametr_tab:
#     najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
#     print(f'Najlepszy wynik dla hiperparametru krzyzowania {krzyzowanie_hiperparametr}')
#     decoded = boundries_based_decode(granice, ilosc_bitow, najlepszy)
#     print('f(%s) = %f' % (decoded[0], wynik))

# granice =[-8,8]
# ilosc_iteracji = 100
# ilosc_bitow = 16
# ilosc_populacji_tab = [10,50,100,200,500]
# krzyzowanie_hiperparametr = 0.9
# mutacja_hiperparametr = 1/(ilosc_bitow*len(granice))
# for ilosc_populacji in ilosc_populacji_tab:
#     najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
#     print(f'Najlepszy wynik dla populacji {ilosc_populacji}')
#     decoded = boundries_based_decode(granice, ilosc_bitow, najlepszy)
#     print('f(%s) = %f' % (decoded[0], wynik))

# granice =[-8,8]
# ilosc_iteracji = 100
# ilosc_bitow = 16
# ilosc_populacji = 100
# krzyzowanie_hiperparametr = 0.9
# mutacja_hiperparametr_tab = [0.1,0.3,0.5,0.7,0.9]
# for mutacja_hiperparametr in mutacja_hiperparametr_tab:
#     najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
#     print(f'Najlepszy wynik dla wsp. mutacji {mutacja_hiperparametr}')
#     decoded = boundries_based_decode(granice, ilosc_bitow, najlepszy)
#     print('f(%s) = %f' % (decoded[0], wynik))

# granice =[-8,8]
# ilosc_iteracji_tab = 100
# ilosc_bitow_tab= [8,16,32,64]
# ilosc_populacji = 100
# krzyzowanie_hiperparametr = 0.9
# mutacja_hiperparametr = 1/(ilosc_bitow*len(granice))
# for ilosc_bitow in ilosc_bitow_tab:
#     najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
#     print(f'Najlepszy wynik dla chromosomu o długości {ilosc_bitow}')
#     decoded = boundries_based_decode(granice, ilosc_bitow, najlepszy)
#     print('f(%s) = %f' % (decoded[0], wynik))
