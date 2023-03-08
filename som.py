# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 23:32:58 2023

@author: Osman VARIŞLI

Self Organizing Maps
"""
import numpy as np
import math
import csv
from sklearn import preprocessing
class SOM:

    def kazanan(self, weights, sample):

        A = 0
        B = 0

        for i in range(len(sample)):
            A = A + math.pow((sample[i] - weights[0][i]), 2)
            B = B + math.pow((sample[i] - weights[1][i]), 2)

        if A > B:  return 1
        else: return 0


    def agirlik_guncelle(self, weights, sample, J, n):

        for i in range(len(weights[0])):
            weights[J][i] = weights[J][i] + n * (sample[i] - weights[J][i])
        return weights

def main():
  
    with open("dataset/diabetes.csv", 'r') as x:
        diabet_data = list(csv.reader(x, delimiter=","))
    diabet_data = np.array(diabet_data)
    diabet_data = np.delete(diabet_data, 0, axis=0) #başlıkları siliyoruz
    
    diabet_data = np.delete(diabet_data, 6, axis=1) #Insulin siliyoruz
    diabet_data = np.delete(diabet_data, 5, axis=1) #BMI siliyoruz
    diabet_data = np.delete(diabet_data, 4, axis=1) #DiabetesPedigreeFunction siliyoruz

    diabet_data = diabet_data.astype(np.float)
    Xa=diabet_data[:,0:5] 

    X=preprocessing.scale(diabet_data)

    X_train=X[0:550,:]
    X_test=X[550:,:]
    

    T=X_train
    m, n = len(T), len(T[0])


    
    weights=[[0.67855669,0.6942807,0.41191634,0.82115273,0.20854013,0.85388117],
             [0.70454461,0.89058265,0.45115704,0.09250288,0.72520027,0.91155235]]

    #weights=np.random.rand(2,n)

    ob = SOM()

    epochs = 100
    n = 0.002

    for i in range(epochs):
        print('epok  :',i)
        for j in range(m):

            sample = T[j]
            
            J = ob.kazanan(weights, sample)

            weights = ob.agirlik_guncelle(weights, sample, J, n)

    
    tahmin=[]
    sonuc_t=[]
    for j in range(len(X_test)):
        J = ob.kazanan(weights, X_test[j])
        tahmin.append(J)
        sonuc_t.append(int(X_test[j][-1]))

        
    from sklearn.metrics import accuracy_score 
    print('-------------------------------')
    print('SOM modeli için başarı oranı : ', accuracy_score(sonuc_t, tahmin))
    print('-------------------------------')
    print('Not: Değer çok küçük çıkarsa sorun değil, sadece guruplama labelleri yanlış demektir. ')
    print('Not: Weight değeri saklanıp sonra ki datalar için de kullanılabilir.')


if __name__ == "__main__":
	main()
