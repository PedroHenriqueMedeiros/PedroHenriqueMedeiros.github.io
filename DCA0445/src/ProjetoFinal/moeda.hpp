#ifndef MOEDA_H
#define MOEDA_H
 
/* Estrutura que define um círculo, que contornará a moeda. */
struct Circulo {
    Point2f centro;
    float raio;
};

/* Estrutura que representa a moeda e outras informações associadas, tais como 
 * seu contorno e o menor retângulo que a cerca. */
struct Moeda {

    Mat imagem;
    int valor = 0;
    double raio;
    double momentos[7];
    Point2f centro;
    Rect retangulo;
    vector<vector<Point> > contornos;
    vector<Vec4i> hierarquia;
    
    /* Calcula os momentos invariantes quando a moeda é criada. */
    Moeda(Mat _imagem) 
    {
        Mat moedaCinza;
        imagem = _imagem.clone();
        
        /* Calcula os momentos invariantes. */
        cvtColor(imagem, moedaCinza, CV_RGB2GRAY);
        Moments momento = moments(moedaCinza, false);
        HuMoments(momento, momentos);
    }
    
    void imprimirMomentos()
    {
        stringstream saida;
        saida << valor << " = {";
        for(int j = 0; j < 7; j++)
        {    
            saida << scientific << setprecision(7) << momentos[j];
            if (j < 6)
            {
                saida << ", ";
            }
        }
        saida << "};";
        cout << saida.str() << endl;
    }
};

#endif
