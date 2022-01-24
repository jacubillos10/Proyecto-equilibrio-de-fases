import numpy as np

#Cuando importe algo de un archivo csv, recuerde que la primera fila es ignorada, pero a la vez esta no puede estar vacía.
Interaccion=np.array(np.genfromtxt('ParametrosInteraccion.csv',delimiter=',',names=True))

ParametrosFin=[]
for i in range(len(Interaccion)):
	A=[]
	vecto=Interaccion[i]
	for j in range(len(vecto)):
		A.append(vecto[j])
	#END FOR 
	ParametrosFin.append(A)
#END FOR

ParametrosFin=np.array(ParametrosFin)

	
