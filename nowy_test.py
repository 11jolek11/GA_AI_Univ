from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np
from math import *
from statistics import mean
from copy import copy



GRANICE = (-8 + 1e-12, 8 - 1e-12)

def get_function(sequence: str):
    return eval("lambda x:" + sequence)

# def dekodowanie(ilosc_bitow,zakodowana_wartosc):
#     zdekodowana = list()
#     for i in range(len(GRANICE)):
#         start,stop = i*ilosc_bitow,(i+1)*ilosc_bitow
#         substring = zakodowana_wartosc[start:stop]
#         chars = ''.join([str(i) for i in substring])
#         integer = int(chars,2)
#         zakres_wartosci_skalowanie =GRANICE[0]+(integer/((2**ilosc_bitow)-1))*(GRANICE[1]-GRANICE[0])
#         zdekodowana.append(zakres_wartosci_skalowanie)
#     return zdekodowana

def dekodowanie(ilosc_bitow, binary_list):
    L = len(binary_list)
    decimal_value = sum(bit * 2 ** (L - 1 - i) for i, bit in enumerate(binary_list))
    return GRANICE[0] + decimal_value * (GRANICE[1] - GRANICE[0]) / (2 ** L - 1)

# def selekcja(populacja, wartosci, k_hipherparametr=3):
#     selekcja_losowa = randint(len(populacja))
#     for i in range(0, len(populacja), k_hipherparametr-1):
#         if wartosci[i] > wartosci[selekcja_losowa]:
#             selekcja_losowa = i
#     return populacja[selekcja_losowa]

# def selekcja(populacja, wartosci, k_hipherparametr=3):
#     selekcja_losowa = randint(len(populacja))
#     for i in range(0, len(populacja), k_hipherparametr-1):
#         if wartosci[i] > wartosci[selekcja_losowa]:
#             selekcja_losowa = i
#     return populacja[selekcja_losowa]

def selekcja(population, fitness_values):
    population_and_fitness = list(zip(population, fitness_values))
    population_and_fitness.sort(key=lambda x: x[1])
    population_to_consider = [x[0] for x in population_and_fitness[:int(len(population) / 2)]]

    return population_to_consider[randint(0, len(population_to_consider))], population_to_consider[randint(0, len(population_to_consider))]

# def selekcja(population, fitness_values):
#     population_and_fitness = list(zip(population, fitness_values))
#     population_and_fitness.sort(key=lambda x: x[1])

def selekcja_r(populacja, wartosci_przystosowania, funkcja_oceny):
    suma_ocen = sum(funkcja_oceny(osobnik) for osobnik in populacja)
    prawdopodobienstwa = [funkcja_oceny(osobnik) / suma_ocen for osobnik in populacja]
    ruletka = [sum(prawdopodobienstwa[:i+1]) for i in range(len(prawdopodobienstwa))]
    nowa_populacja = []
    for _ in range(2):
        losowy = rand()
        for (i, osobnik) in enumerate(populacja):
            if losowy <= ruletka[i]:
                nowa_populacja.append(osobnik)
                break
    return nowa_populacja[0], nowa_populacja[1]

def selekcja_r(populacja, wartosci_przystosowania):
    suma_ocen = sum(wartosci_przystosowania)
    prawdopodobienstwa = [wartosci_przystosowania[osobnik] / suma_ocen for osobnik in range(len(populacja))]
    ruletka = [sum(prawdopodobienstwa[:i+1]) for i in range(len(prawdopodobienstwa))]
    nowa_populacja = []
    for _ in range(2):
        losowy = rand()
        for (i, osobnik) in enumerate(populacja):
            if losowy <= ruletka[i]:
                nowa_populacja.append(osobnik)
                break
    return nowa_populacja[0], nowa_populacja[1]


def krzyzowanie(rodzic1,rodzic2):
    dziecko1, dziecko2 = rodzic1.copy(), rodzic2.copy()
    for _ in range(2):
        punkt_krzyzowania = randint(1,len(rodzic1)-2)
        dziecko1 = rodzic1[:punkt_krzyzowania] + rodzic2[punkt_krzyzowania:]
        dziecko2 = rodzic2[:punkt_krzyzowania] + rodzic1[punkt_krzyzowania:]
    return dziecko1, dziecko2

def mutacja(zakodowana_wartosc, mutacja_hiperparametr=0.01):
    for i in range(len(zakodowana_wartosc)):
        if rand() < mutacja_hiperparametr:
            zakodowana_wartosc[i] = 1 - zakodowana_wartosc[i]

def algorytm_genetyczny(ilosc_bitow: int, licznosc_populacji: int, fitness: callable, generation_num: int, mut_score: float):

    # Generacja populacji
    populacja = [[randint(2) for _ in range(len(GRANICE)*ilosc_bitow)] for _ in range(licznosc_populacji)]
    assert(len(populacja) == licznosc_populacji)
    stara_populacja = copy(populacja)

    for gen in range(generation_num):
        print(f"Generation >> {gen} <<")

        # Sortowanie do oceny
        # sorted(populacja, key=lambda individual: fitness(dekodowanie(ilosc_bitow, individual)) , reverse=True)

        # Ocena "wstepna"
        # najlepiej_przystosowana = fitness(dekodowanie(ilosc_bitow, populacja[0]))

        # for individual in populacja:
        #     if individual is None:
        #         print(f"{individual} is NOne at gen {gen}")
        #     else:
        #         print("OK")

        wartosci_przystosowania = [fitness(dekodowanie(ilosc_bitow, individual)) for individual in populacja]
        old_wartosci_przystosowania = [fitness(dekodowanie(ilosc_bitow, individual)) for individual in stara_populacja]

        # FIXME(11jolek11): Doesn't break
        if abs(mean(wartosci_przystosowania) - mean(old_wartosci_przystosowania)) <= 1e-23 and gen > 0:
            print("BREAK")
            
            return populacja

        nowa_populacja = []

        for i in range(0, licznosc_populacji, 2):
            parent1, parent2 = selekcja_r(populacja, wartosci_przystosowania)
            # nowa_populacja.append(parent1)
            # nowa_populacja.append(parent2)

            # crossover
            cross_over_rate=1
            if rand() < cross_over_rate:
                child1, child2 = krzyzowanie(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            mutacja(child1, mut_score)
            mutacja(child2, mut_score)
            
            nowa_populacja.append(child1)
            nowa_populacja.append(child2)
        
        stara_populacja = copy(populacja)
        # assert(len(populacja) == licznosc_populacji)
        assert(len(populacja) == len(stara_populacja))
        populacja = nowa_populacja
    
    print('Standard return')
    return populacja



if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (5,5)
    # func_str = input("Give funct eq: ")

    # insider = "-0.5*x**3"
    # insider = "x**2"
    insider = "sin(0.25*x**2) - cos(0.125*x**2)"
    ilosc_bitow = 64
    f_string = get_function(insider)

    vfunc = np.vectorize(f_string)

    x_axis = np.arange(GRANICE[0], GRANICE[1], 0.0001)

    popu = algorytm_genetyczny(ilosc_bitow, 100, f_string, 100, mut_score=0.1)
    x_guess = [dekodowanie(ilosc_bitow, individual) for individual in popu]

    best_p = sorted(x_guess, key= lambda x: f_string(x))[0]

    print("Najlepszy osiągnięty wynik: {} dla {}".format(vfunc(best_p), best_p))

    plt.plot(best_p, vfunc(best_p), color="green")

    plt.title(f"Function {insider}")

    plt.plot(x_guess, vfunc(x_guess), '1', color='red')
    plt.plot(x_axis, vfunc(x_axis))

    plt.savefig("test.jpg")
