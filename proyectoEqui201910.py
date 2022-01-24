import numpy as np
import matplotlib.pyplot as plt
import math
import datosSust
import ParInteraccion

#Mira (con tilde) aqui esta la forma de importar la base de datos: print(datosSust.Mat)
datosWilson=datosSust.Mat;
Int=ParInteraccion.ParametrosFin;
Componente1=25; #Etanol
Componente2=30; #Agua
P=101.325; #kPa

def ecuacionesWilson(VECT,x,P,Sustancia1,Sustancia2):
	T=np.asscalar(VECT[1]);
	y1=np.asscalar(VECT[0]);
	x1=x;
	x2=1-x1;
	y2=1-y1;
	R=8.314472; #J/mol K
	datosSustancia1=datosWilson[Sustancia1,:];
	datosSustancia2=datosWilson[Sustancia2,:];
	#++++++++++++++++++++++++++++++++++++INICIO PARAMETROS DE ENTRADA+++++++++++++++++++++++++++++++++++++++++++++++++
	#***Parametros de Wilson Para la sustancia 1:	
	W1=datosSustancia1[1]
	V25_1=W1[0];
	Vb1=W1[1];
	Delta25_1=W1[2];
	Tb1=W1[3];
	#***Parametros de Antoine Para la sustancia 1:
	Ant1=datosSustancia1[2]
	A1=Ant1[0];
	B1=Ant1[1];
	C1=-Ant1[2];
	
	#***Parametros de Wilson Para la sustancia 2:
	W2=datosSustancia2[1]
	V25_2=W2[0];
	Vb2=W2[1];
	Delta25_2=W2[2];
	Tb2=W2[3];
	#***Parametros de Antoine Para la sustancia 2:
	Ant2=datosSustancia2[2]
	A2=Ant2[0];
	B2=Ant2[1];
	C2=-Ant2[2];

	#+++++++++++++++++++++++++++++++++++FIN PARAMETROS DE ENTRADA++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	#*********Calculo de las presiones de saturacion***********
	P1sat=10**(A1-(B1/(C1+T))); #kPa
	P2sat=10**(A2-(B2/(C2+T))); #kPa
	#*********Calculo de otras variables de wilson*************
	Beta1=(Vb1-V25_1)/(Tb1-25);
	Beta2=(Vb2-V25_2)/(Tb2-25);
	v1=V25_1+(Beta1*((T-273.15)-25));
	v2=V25_2+(Beta2*((T-273.15)-25));
	Delta1=(V25_1/v1)*Delta25_1;
	Delta2=(V25_2/v2)*Delta25_2;
	#*******ENTRADAS INTERACCIONES WILSON+++++++++++++++
	Epsilon12=Int[Sustancia1,Sustancia2];
	Epsilon21=Int[Sustancia2,Sustancia1];
	#-------FIN ENTRADAS INTERACCIONES WILSON+++++++++++
	Z=2;
	Lambda11=-(2/Z)*v1*(Delta1**2);
	Lambda22=-(2/Z)*v2*(Delta2**2);
	Lambda12=-(1-Epsilon12)*(2/Z)*((v1*v2)**0.5)*Delta1*Delta2;
	Lambda21=-(1-Epsilon21)*(2/Z)*((v1*v2)**0.5)*Delta2*Delta1;
	A12=(v2/v1)*np.exp(-((Lambda12-Lambda11)/(R*T)));
	A21=(v1/v2)*np.exp(-((Lambda21-Lambda22)/(R*T)));
	#*****************************LOS GAMMAS********************************
	lnGamma1=-np.log(x1+(A12*x2))+x2*((A12/(x1+(A12*x2)))-(A21/((A21*x1)+x2)));
	lnGamma2=-np.log(x2+(A21*x1))-x1*((A12/(x1+(A12*x2)))-(A21/((A21*x1)+x2)));
	Gamma1=np.exp(lnGamma1);
	Gamma2=np.exp(lnGamma2);
	f1=y1*P-Gamma1*x1*P1sat;
	f2=y2*P-Gamma2*x2*P2sat;
	#OK AQUI PARECE QUE TERMINAN LAS ECUACIONES COMO TAL
	resp=np.array([[f1],[f2]])
	return resp
#FIN FUNCTION


def Jacobiano(varIni,x,P,Sus1,Sus2):
	var0=varIni
	nVar=len(varIni);
	h=10**(-8);
	Jalr=[]
	for i in range(nVar):
		Vh=np.zeros((nVar,1));
		Vh[i]=h;
		varH=var0+Vh;
		A=(ecuacionesWilson(varH,x,P,Sus1,Sus2)-ecuacionesWilson(var0,x,P,Sus1,Sus2))//h;
		Jalr.append(A);
	#FIN FOR
	Jalr=np.array(Jalr);
	J=np.transpose(Jalr);
	return J
#FIN FUNCTION
#v1=np.array([[320],[0.8]])
#Jaac=Jacobiano(v1,0.3,101.325,25,30)
#v2=np.dot(Jaac,v1)

def equilibrio(x,P,Sus1,Sus2):
	Error=100;
	tol=10**(-5);
	cIter=0;
	maxIter=100;
	datosSustancia1=datosWilson[Sus1,:];
	datosSustancia2=datosWilson[Sus2,:];
	Ant1=datosSustancia1[2];
	Ant2=datosSustancia2[2];
	A1=Ant1[0];
	B1=Ant1[1];
	C1=-Ant1[2];
	A2=Ant2[0];
	B2=Ant2[1];
	C2=-Ant2[2];
	T1sat=(B1/(A1-math.log10(P)))-C1;
	T2sat=(B2/(A2-math.log10(P)))-C2;
	TM=(T1sat+T2sat)/2;
	v0=np.array([[0.5],[TM]]); #valores supuestos
	while Error>tol and cIter<=maxIter:
		Jac=Jacobiano(v0, x,P, Sus1, Sus2);
		fx0=ecuacionesWilson(v0,x,P,Sus1,Sus2);
		Jacinv=np.linalg.inv(Jac);
		vnew=v0-np.dot(Jacinv,fx0);
		cIter=cIter+1;
		#por alguna razón se forman otros corchetes encima del vector
		vnuevo=vnew[0];
		Error=abs((np.linalg.norm(vnuevo)-np.linalg.norm(v0))/np.linalg.norm(v0));
		v0=vnuevo;
	#FIN WHILE
	if cIter>=maxIter:
		print("Advertencia: Máximas iteraciones alcanzadas en x= ",x);
	#FIN IF
	JacN=Jacobiano(vnuevo,x,P,Sus1,Sus2);
	return [vnuevo, JacN];
#FIN FUNCTION

#print("Prueba ",equilibrio(0,101.325,25,30))
#Finura=201;
#xL=np.linspace(0,1,Finura);
#y=np.zeros((Finura,1));
#T=np.zeros((Finura,1));
#for i in range(Finura):
#	Salidas=equilibrio(xL[i],P,Componente1,Componente2);
#	y[i,:]=Salidas[1];
#	T[i,:]=Salidas[0];
##FIN FOR

Valores=equilibrio(0.3,P,Componente1,Componente2)[0];
Jacobiano=equilibrio(0.3,P,Componente1,Componente2)[1];
print("Temperatura", Valores[0])
print("Composicion", Valores[1])
print("EcWil", ecuacionesWilson([Valores[0],Valores[1]], 0.3,P,Componente1,Componente2));
print("Jacobiano: ", Jacobiano[0]);
print("Jacobiano: ", Jacobiano[0][1,1]);
#fig = plt.figure()
#plt.title("Gráfica Txy")
#plt.xlabel("Composicion")
#plt.ylabel("Temperatura [K]")
#plt.plot(xL,T,y,T)
#plt.savefig("DiagramaTxy.png")

