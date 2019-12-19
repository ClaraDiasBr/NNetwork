'''
Este código faz:
- Cria um array aleaório do tipo M colunas por N linhas
- "Pega" cada linha do array e transforma para string
- Transforma o elemento "string-binario" em decimal
- Add numa lista
- E transforma a lista num tensor(list,[formato,tensor])

'''

import numpy as np
import tensorflow as tf
import keras
from pprint import pprint
import math
#matplotlib inline

# Set random seeds
from numpy.random import seed
seed(1)
tf.random.set_seed(3)

# Encoder Parameters
M = 3 # Nº de bits 128
N = 100 # Amostras 10000
#exp = 0
num = 0
num_decimal = list()
#print('M:',M, 'N:',N,)

#generating data of size N
label = np.random.randint(2,size=[N,M])
#label = np.array(label) desnecessario
print(label)

total = ""

for j in range(N):
	for element in label[j]:
		#turning line into string
		total += str(element)
	
	#print(total.encode())

	#turning string-bunary into decimal
	total_conv = int(total,2)
	#print(total_conv)
	
	#adding it to a list
	num_decimal.append(total_conv)
	total=""	

#print(num_decimal)	
#num_decimal = np.array(num_decimal)

#turning list to tensor
num_dec = tf.reshape(num_decimal, [10,10])
print(num_dec)	
