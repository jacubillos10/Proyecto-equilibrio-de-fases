#include <iostream>
#include <fstream>
#include <cmath>
#include <armadillo>
#include <gtkmm-3.0/gtkmm.h>

// Compílelo así:
//g++ py1.cpp -larmadillo -llapack -lblas `pkg-config gtkmm-3.0 --cflags --libs`
//Hola hola
class Sustancia
{
    public:
        std::string nombre;
        int indiceS;
        double v25; //volumen a 25°C en cm³/mol
        double vb; // volumen en el punto de ebullición en cm³/mol
        double delta25; // delta25, ignoro que es eso
        double T_eb; // temperatura de ebullición a 1 bar
        double A;
        double B;
        double C; //constantes de Antoine en log base 10 y C en Kelvin
        arma::mat parametrosInteraccion;
};

//Bueno... ya sabes que puedes hacer funciones que devuelvan arma::mat directamente sin void.
//Sin embargo, se hizo el programa a punta de pointers... grande... 
void obtenerSustancia(int indice, arma::mat Datos, arma::mat Interaccion, Sustancia* dirSustancia);
void ecuacionesWilson(arma::mat vars, double x, double P, Sustancia sus1, Sustancia sus2, arma::mat* dirResp);
void jacobiano(arma::mat vars, double x, double P, Sustancia comp1, Sustancia comp2, double h, arma::mat* dirJac);
void equilibrio(double x, double P, Sustancia comp1, Sustancia comp2, arma::mat vecIni, arma::mat* dirResp);

int main(int argc, char *argv[])
{
    arma::mat losDatos;
    arma::mat Interacciones;
    losDatos.load("datosSustancias.csv");
    Interacciones.load("ParametrosInteraccion.csv");
    Sustancia *dirComp1 = new Sustancia;
    Sustancia *dirComp2 = new Sustancia;
    obtenerSustancia(25,losDatos,Interacciones,dirComp1);
    obtenerSustancia(30,losDatos,Interacciones,dirComp2);
    Sustancia laSustancia1=*dirComp1;
    Sustancia laSustancia2=*dirComp2;
    arma::mat variablesIni = arma::mat{0.4,353}.t();
    arma::mat *dirRespuesta = new arma::mat;
    equilibrio(0.3,101.325,laSustancia1,laSustancia2,variablesIni,dirRespuesta);
    std::cout<<"La respuesta es: \n"<<(*dirRespuesta)<<"\n";
    delete dirComp1;
    delete dirComp2;
    delete dirRespuesta;
    return 0;    
}

void obtenerSustancia(int indice, arma::mat Datos, arma::mat Interaccion, Sustancia* dirSustancia)
{
    /*
    Esta función devuelve una sustancia de la lista de los datos de las sustancias y de la lista de parámetros de interacción, 
    dado un número entero. Se devolverá la sustancia de la [número entero de entrada "índice"] fila del archivo de los datos de las sustancias
    */
    Sustancia compuesto;
    compuesto.v25=Datos(indice,1);
    compuesto.vb=Datos(indice,2);
    compuesto.delta25=Datos(indice,3);
    compuesto.T_eb=Datos(indice,4);
    compuesto.A=Datos(indice,5);
    compuesto.B=Datos(indice,6);
    compuesto.C=Datos(indice,7);
    compuesto.parametrosInteraccion=Interaccion.row(indice);
    compuesto.indiceS=indice;
    *dirSustancia=compuesto;
}

/*
Esta función determina el equilibrio de un sistema binario dada una composición líquida y la presión. Es decir, dada la composición líquida, detrmina
la composición de valor y la temperatura. 
*/
void equilibrio(double x, double P, Sustancia comp1, Sustancia comp2, arma::mat vecIni, arma::mat* dirResp)
{
    int iter, maxIter;
    double error, tol, h;
    arma::mat vnuevo;
    arma::mat v0;
    iter=0;
    maxIter=100;
    error=1000;
    tol=1e-6;
    h=1e-7;
    v0=vecIni;
    arma::mat *dirfv0 = new arma::mat;
    arma::mat *dirJacobiano = new arma::mat;
    while (error>tol && iter<=maxIter)
    {
        jacobiano(v0,x,P,comp1,comp2,h,dirJacobiano);
        ecuacionesWilson(v0,x,P,comp1,comp2,dirfv0);
        vnuevo=v0-arma::solve((*dirJacobiano),(*dirfv0));
        iter++;
        error=std::abs(arma::norm(vnuevo,1)-arma::norm(v0,1))/arma::norm(v0,1);
        v0=vnuevo;
    }
    if (iter>maxIter)
    {
        std::cout<<"OJO! Máximas iteraciones alcanzadas \n";
    }
    delete dirJacobiano;
    delete dirfv0;
    (*dirResp)=vnuevo;
}

void jacobiano(arma::mat vars, double x, double P, Sustancia comp1, Sustancia comp2, double h, arma::mat* dirJac)
{
    /*
    Esta función devuelve el jacobiano de la función ecuaciones Wilson en un punto
    */
    int nVars=vars.n_elem;
    arma::mat vec_h, vec_hmas, derivada;
    arma::mat Jac;
    Jac=arma::zeros<arma::mat>(nVars,nVars);
    arma::mat *dirEcu0 = new arma::mat;
    arma::mat *dirEcuh = new arma::mat;
    for (int i=0; i<nVars;i++)
    {
        vec_h.zeros(2); //Por defecto zeros y ones hace vectores columna
        vec_h(i)=h;
        vec_hmas=vars+vec_h;
        ecuacionesWilson(vars,x,P,comp1,comp2,dirEcu0);
        ecuacionesWilson(vec_hmas,x,P,comp1,comp2,dirEcuh);
        derivada=(1/h)*((*dirEcuh)-(*dirEcu0));
        Jac.col(i)=derivada;
    }
    delete dirEcu0;
    delete dirEcuh;
    (*dirJac)=Jac;
}
/*
Esta función iguala las ecuaciones de equilibrio a cero: 
dirResp(0)=y1*P-Gamma1*x1*P1sat;
dirResp(1)=y2*P-Gamma2*x2*P2sat;
El modelo usado es el modelo de Wilson
*/
void ecuacionesWilson(arma::mat vars, double x, double P, Sustancia sus1, Sustancia sus2, arma::mat* dirResp)
{
    double y1, y2, x1, x2, T;
    //parámetros
    double v[2], delta[2], beta[2], lambdas[2][2], parInter[2], Atlantis[2], logGamma[2], Gamma[2], Psat[2], resp[2];
    int z=2;
    double R=8.314;
    // este guarda la respuesta
    y1=vars(0);
    T=vars(1); // Esta debe entrar a esta función en Kelvin
    x1=x;
    y2=1-y1;
    x2=1-x1;
    beta[0]=(sus1.vb-sus1.v25)/(sus1.T_eb-25);
    beta[1]=(sus2.vb-sus2.v25)/(sus2.T_eb-25);
    v[0]=sus1.v25+beta[0]*((T-273.15)-25); //En el cálculo de los volúmenes la temperatura está en °C
    v[1]=sus2.v25+beta[1]*((T-273.15)-25); //En el cálculo de los volúmenes la temperatura está en °C
    delta[0]=(sus1.v25/v[0])*sus1.delta25;
    delta[1]=(sus2.v25/v[1])*sus2.delta25;
    parInter[0]=sus1.parametrosInteraccion(sus2.indiceS);
    parInter[1]=sus2.parametrosInteraccion(sus1.indiceS);
    lambdas[0][1]=-(1-parInter[0])*(2/z+0.0)*pow(v[0]*v[1],0.5)*delta[0]*delta[1];
    lambdas[1][0]=-(1-parInter[1])*(2/z+0.0)*pow(v[1]*v[0],0.5)*delta[1]*delta[0];
    lambdas[0][0]=-(2/z+0.0)*v[0]*pow(delta[0],2);
    lambdas[1][1]=-(2/z+0.0)*v[1]*pow(delta[1],2);
    Atlantis[0]=(v[1]/v[0])*exp(-(lambdas[0][1]-lambdas[0][0])/(R*T));
    Atlantis[1]=(v[0]/v[1])*exp(-(lambdas[1][0]-lambdas[1][1])/(R*T));
    logGamma[0]=-log(x1+Atlantis[0]*x2)+x2*((Atlantis[0]/(x1+Atlantis[0]*x2))-(Atlantis[1]/(Atlantis[1]*x1+x2)));
    logGamma[1]=-log(Atlantis[1]*x1+x2)-x1*((Atlantis[0]/(x1+Atlantis[0]*x2))-(Atlantis[1]/(Atlantis[1]*x1+x2)));
    Gamma[0]=exp(logGamma[0]);
    Gamma[1]=exp(logGamma[1]);
    Psat[0]=pow(10,sus1.A-(sus1.B/(T-sus1.C)));
    Psat[1]=pow(10,sus2.A-(sus2.B/(T-sus2.C)));
    resp[0]=y1*P-Gamma[0]*x1*Psat[0];
    resp[1]=y2*P-Gamma[1]*x2*Psat[1];
    (*dirResp)=arma::mat{resp[0],resp[1]}.t();
}  
