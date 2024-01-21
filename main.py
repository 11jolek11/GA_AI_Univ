from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np

func = input("Podaj funkcję: ")
def input_function(x):
    x = x[0]
    result = eval(func)
    return result

def dekodowanie(granice,ilosc_bitow,zakodowana_wartosc):
    zdekodowana = list()
    for i in range(len(granice)):
        start,stop = i*ilosc_bitow,(i+1)*ilosc_bitow
        substring = zakodowana_wartosc[start:stop]
        chars = ''.join([str(i) for i in substring])
        integer = int(chars,2)
        zakres_wartosci_skalowanie =granice[0]+(integer/((2**ilosc_bitow)-1))*(granice[1]-granice[0])
        zdekodowana.append(zakres_wartosci_skalowanie)
    return zdekodowana

def selekcja(populacja,wartosci,k_hipherparametr=3):
    selekcja_losowa =randint(len(populacja))
    for i in range(0,len(populacja),k_hipherparametr-1):
        if wartosci[i] > wartosci[selekcja_losowa]:
            selekcja_losowa = i
    return populacja[selekcja_losowa]

def krzyzowanie(rodzic1,rodzic2,krzyzowanie_hiperparametr):
    dziecko1, dziecko2 = rodzic1.copy(), rodzic2.copy()
    if rand() < krzyzowanie_hiperparametr:
        punkt_krzyzowania = randint(1,len(rodzic1)-2)
        dziecko1 = rodzic1[:punkt_krzyzowania] + rodzic2[punkt_krzyzowania:]
        dziecko2 = rodzic2[:punkt_krzyzowania] + rodzic1[punkt_krzyzowania:]
    return [dziecko1,dziecko2]

def mutacja(zakodowana_wartosc,mutacja_hiperparametr):
    for i in range(len(zakodowana_wartosc)):
        if rand() < mutacja_hiperparametr:
            zakodowana_wartosc[i] = 1 - zakodowana_wartosc[i]

def algorytm_genetyczny(zadana_funkcja,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr):
    populacja = [[randint(2) for i in range(len(granice)*ilosc_bitow)] for j in range(ilosc_populacji)]
    print(populacja)
    najlepsze_wartosci, najlepsze_populacje = 0, zadana_funkcja(dekodowanie(granice,ilosc_bitow,populacja[0]))
    for generacja in range(ilosc_iteracji):
        zdekodowana_populacja = [dekodowanie(granice,ilosc_bitow,osobnik) for osobnik in populacja]
        wartosci = [zadana_funkcja(osobnik) for osobnik in zdekodowana_populacja]
        for i in range(ilosc_populacji):
            if wartosci[i] < najlepsze_populacje:
                najlepsze_wartosci, najlepsze_populacje = populacja[i], wartosci[i]
                print( f"Najlepsza wartość f({zdekodowana_populacja[i][0]}) = ", wartosci[i])
        wybrani_rodzice = [selekcja(populacja,wartosci) for i in range(ilosc_populacji)]
        potomstwo = list()
        for i in range(0,ilosc_populacji,2):
            rodzic1, rodzic2 = wybrani_rodzice[i], wybrani_rodzice[i+1]
            for dziecko in krzyzowanie(rodzic1,rodzic2,krzyzowanie_hiperparametr):
                mutacja(dziecko,mutacja_hiperparametr)
                potomstwo.append(dziecko)
        populacja = potomstwo
    plt.figure()
    plt.title(f"Algorytm genetyczny dla funkcji {func}")
  
    plt.plot(zdekodowana_populacja,wartosci,'x',color='black')
    x_axis = np.arange(granice[0],granice[1],0.1)

    x = np.arange(granice[0],granice[1],0.1)
    plt.plot(x_axis,eval(func))
    return [najlepsze_wartosci, najlepsze_populacje]
                

granice =[-8,8]
ilosc_iteracji = 100
ilosc_bitow = 16
ilosc_populacji = 100
krzyzowanie_hiperparametr = 0.9
mutacja_hiperparametr = 1/(ilosc_bitow)
najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
print('Najlepszy wynik dla')
decoded = dekodowanie(granice, ilosc_bitow, najlepszy)
print('f(%s) = %f' % (decoded[0], wynik))

granice =[-8,8]
ilosc_iteracji_tab = [10,50,100,200,500]
ilosc_bitow = 16
ilosc_populacji = 100
krzyzowanie_hiperparametr = 0.9
mutacja_hiperparametr = 1/(ilosc_bitow*len(granice))
for ilosc_iteracji in ilosc_iteracji_tab:
    najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
    print(f'Najlepszy wynik dla iteracji {ilosc_iteracji}')
    decoded = dekodowanie(granice, ilosc_bitow, najlepszy)
    print('f(%s) = %f' % (decoded[0], wynik))

granice =[-8,8]
ilosc_iteracji = 100
ilosc_bitow = 16
ilosc_populacji = 100
krzyzowanie_hiperparametr_tab = [0.1,0.3,0.5,0.7,0.9]
mutacja_hiperparametr = 1/(ilosc_bitow*len(granice))
for krzyzowanie_hiperparametr in krzyzowanie_hiperparametr_tab:
    najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
    print(f'Najlepszy wynik dla hiperparametru krzyzowania {krzyzowanie_hiperparametr}')
    decoded = dekodowanie(granice, ilosc_bitow, najlepszy)
    print('f(%s) = %f' % (decoded[0], wynik))

granice =[-8,8]
ilosc_iteracji = 100
ilosc_bitow = 16
ilosc_populacji_tab = [10,50,100,200,500]
krzyzowanie_hiperparametr = 0.9
mutacja_hiperparametr = 1/(ilosc_bitow*len(granice))
for ilosc_populacji in ilosc_populacji_tab:
    najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
    print(f'Najlepszy wynik dla populacji {ilosc_populacji}')
    decoded = dekodowanie(granice, ilosc_bitow, najlepszy)
    print('f(%s) = %f' % (decoded[0], wynik))

granice =[-8,8]
ilosc_iteracji = 100
ilosc_bitow = 16
ilosc_populacji = 100
krzyzowanie_hiperparametr = 0.9
mutacja_hiperparametr_tab = [0.1,0.3,0.5,0.7,0.9]
for mutacja_hiperparametr in mutacja_hiperparametr_tab:
    najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
    print(f'Najlepszy wynik dla wsp. mutacji {mutacja_hiperparametr}')
    decoded = dekodowanie(granice, ilosc_bitow, najlepszy)
    print('f(%s) = %f' % (decoded[0], wynik))

granice =[-8,8]
ilosc_iteracji_tab = 100
ilosc_bitow_tab= [8,16,32,64]
ilosc_populacji = 100
krzyzowanie_hiperparametr = 0.9
mutacja_hiperparametr = 1/(ilosc_bitow*len(granice))
for ilosc_bitow in ilosc_bitow_tab:
    najlepszy,wynik = algorytm_genetyczny(input_function,granice,ilosc_bitow,ilosc_iteracji,ilosc_populacji,krzyzowanie_hiperparametr,mutacja_hiperparametr)
    print(f'Najlepszy wynik dla chromosomu o długości {ilosc_bitow}')
    decoded = dekodowanie(granice, ilosc_bitow, najlepszy)
    print('f(%s) = %f' % (decoded[0], wynik))
