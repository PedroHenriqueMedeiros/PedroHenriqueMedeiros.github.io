#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iomanip>
#include "opencv2/legacy/legacy.hpp"

using namespace std;
using namespace cv;


#define PONTOS_CIRCULO 50
#define LIMIAR_RELACAO_AREAS 0.70
#define LIMIAR_RAIO 10
#define MAX_FECHAMENTO 32 // Cresce a partir de 1 em potências de 2.
#define RESULTADO_SEPARADO true


/* Momentos invariantes das moedas. */

const double M1001H[] = {2.0047524e-03, 7.6195900e-09, 6.6652229e-12, 3.6035503e-11, -2.8540688e-22, 2.7119486e-15, -4.8003802e-22};
const double M1001S[] = {2.0047524e-03, 7.6195900e-09, 6.6652229e-12, 3.6035503e-11, -2.8540688e-22, 2.7119486e-15, -4.8003802e-22};
const double M1001V[] = {2.0047524e-03, 7.6195900e-09, 6.6652229e-12, 3.6035503e-11, -2.8540688e-22, 2.7119486e-15, -4.8003802e-22};
const double M1002H[] = {1.8209878e-03, 5.7594182e-09, 9.6894857e-12, 3.4945343e-12, -2.0186826e-23, -2.5762759e-16, 2.4465278e-24};
const double M1002S[] = {1.8209878e-03, 5.7594182e-09, 9.6894857e-12, 3.4945343e-12, -2.0186826e-23, -2.5762759e-16, 2.4465278e-24};
const double M1002V[] = {1.8209878e-03, 5.7594182e-09, 9.6894857e-12, 3.4945343e-12, -2.0186826e-23, -2.5762759e-16, 2.4465278e-24};
const double M501H[] = {1.8995101e-03, 3.5026479e-09, 9.9531483e-13, 5.4271398e-12, -8.6628792e-24, 2.9890887e-16, 9.1681920e-24};
const double M501S[] = {1.8995101e-03, 3.5026479e-09, 9.9531483e-13, 5.4271398e-12, -8.6628792e-24, 2.9890887e-16, 9.1681920e-24};
const double M501V[] = {1.8995101e-03, 3.5026479e-09, 9.9531483e-13, 5.4271398e-12, -8.6628792e-24, 2.9890887e-16, 9.1681920e-24};
const double M502H[] = {1.8440455e-03, 3.4546870e-09, 1.8010588e-12, 9.3299591e-12, -8.5691110e-24, -5.4746165e-16, 3.7273452e-23};
const double M502S[] = {1.8440455e-03, 3.4546870e-09, 1.8010588e-12, 9.3299591e-12, -8.5691110e-24, -5.4746165e-16, 3.7273452e-23};
const double M502V[] = {1.8440455e-03, 3.4546870e-09, 1.8010588e-12, 9.3299591e-12, -8.5691110e-24, -5.4746165e-16, 3.7273452e-23};
const double M251H[] = {1.7283154e-03, 2.9919907e-09, 7.3535902e-12, 5.9363725e-13, -8.5496023e-25, 2.0423030e-20, 8.9856571e-25};
const double M251S[] = {1.7283154e-03, 2.9919907e-09, 7.3535902e-12, 5.9363725e-13, -8.5496023e-25, 2.0423030e-20, 8.9856571e-25};
const double M251V[] = {1.7283154e-03, 2.9919907e-09, 7.3535902e-12, 5.9363725e-13, -8.5496023e-25, 2.0423030e-20, 8.9856571e-25};
const double M252H[] = {1.6736123e-03, 1.3628862e-09, 1.4371021e-12, 1.5141660e-11, -7.0630446e-23, 4.2231771e-16, -5.2690557e-25};
const double M252S[] = {1.6736123e-03, 1.3628862e-09, 1.4371021e-12, 1.5141660e-11, -7.0630446e-23, 4.2231771e-16, -5.2690557e-25};
const double M252V[] = {1.6736123e-03, 1.3628862e-09, 1.4371021e-12, 1.5141660e-11, -7.0630446e-23, 4.2231771e-16, -5.2690557e-25};
const double M101H[] = {1.9511730e-03, 2.5675459e-09, 3.7804369e-14, 3.6840869e-11, 5.1570068e-24, 1.8436298e-15, -4.3170697e-23};
const double M101S[] = {1.9511730e-03, 2.5675459e-09, 3.7804369e-14, 3.6840869e-11, 5.1570068e-24, 1.8436298e-15, -4.3170697e-23};
const double M101V[] = {1.9511730e-03, 2.5675459e-09, 3.7804369e-14, 3.6840869e-11, 5.1570068e-24, 1.8436298e-15, -4.3170697e-23};
const double M102H[] = {1.7295513e-03, 1.5927668e-09, 5.9847518e-13, 1.3172711e-11, 3.4195390e-23, 3.8134996e-16, 1.4093610e-23};
const double M102S[] = {1.7295513e-03, 1.5927668e-09, 5.9847518e-13, 1.3172711e-11, 3.4195390e-23, 3.8134996e-16, 1.4093610e-23};
const double M102V[] = {1.7295513e-03, 1.5927668e-09, 5.9847518e-13, 1.3172711e-11, 3.4195390e-23, 3.8134996e-16, 1.4093610e-23};

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
        for(int i = 0; i < 7; i++)
        {    
            saida << scientific << setprecision(7) << momentos[i];
            if (i < 6)
            {
                saida << ", ";
            }
        }
        saida << "};";
        cout << saida.str() << endl;
    }
};

/* Detecta a quantidade máxima de moedas utilizando a função detectarMoedas. */
vector<Moeda> detectarTodasMoedas(Mat &imagem);

/* Função que detecta moedas em uma imagem, utilizando um determinado nível na
 * operação morfológica do fechamento. */
vector<Moeda> detectarMoedas(Mat &imagem, int fechamento);

/* Função que desenha na imagem ou individualmente as moedas localizadas. */
void exibirMoedas(const Mat &imagem, const vector<Moeda> &moedas);

/* Função que remova moedas da borda da imagem. */
void removeMoedasBorda(Mat &imagem);

/* Função que remove buracos dentro da moeda durante a sua segmentação. */
void removeBuracos(Mat &imagem);

/* Realiza a contagem do dinheiro. */
double contarDinheiro(vector<Moeda> &moedas);

/* Calcula o valor com momento invariante mais similar. */
double calcularValor(Moeda moeda);

struct MomentosInvariantes
{
    double h[7];
    double s[7];
    double v[7];
};

MomentosInvariantes calcularMomentosInvariantes(Mat &imagem)
{
    Mat hsv;
    vector<Mat> planos;
    MomentosInvariantes mi;
    
    cvtColor(imagem, hsv, CV_BGR2HSV);
    split(hsv, planos);
    
    Moments momentoH = moments(planos[0], false);
    Moments momentoS = moments(planos[1], false);
    Moments momentoV = moments(planos[2], false);
    
    HuMoments(momentoH, mi.h);
    HuMoments(momentoS, mi.s);
    HuMoments(momentoV, mi.v);

    return mi;
}

/* Função main. */
int main(int argc, char** argv) 
{
    Mat imagemColorida;
    vector<Moeda> moedas;
    double dinheiro;
    vector<Mat> moedasPreparadas;
    
    /* Verifica o número de argumentos.  */
    if (argc != 2)
    {
        cout << "A lista de argumentos deve ser: "
            "./preparador <imagem>"<< endl;
        return -1;
    }

    /* Checa se a imagem pode ser aberta. */
    imagemColorida = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if (!imagemColorida.data)
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }

    cout << "[main] Realizando deteção pelo algoritmo padrão." << endl;
    moedas = detectarTodasMoedas(imagemColorida);

    cout << "[main] Exibindo " << moedas.size() << " moedas encontradas. " << endl;
    exibirMoedas(imagemColorida, moedas);
    
    //cout << "[main] Realizando a contagem do dinheiro." << endl;
    
    
    dinheiro = contarDinheiro(moedas);
    
    cout << "[main] O dinheiro total contado foi de " << dinheiro << " reais." << endl;  
    
    waitKey(0);
    return(0);
}


void removeMoedasBorda(Mat &imagem)
{
    Point p;
    
    for (int i = 0; i <= imagem.rows; i += imagem.rows - 1)
    {
        for (int j = 0; j < imagem.cols; j++)
        {
            if (imagem.at<uchar>(i, j) == 255)
            {
                p.x = j;
                p.y = i;
             
                floodFill(imagem, p, 0);
            }
        }
    }
    
    for (int j = 0; j < imagem.cols; j += imagem.cols - 1)
    {
        for (int i = 0; i < imagem.rows; i++)
        {
            if (imagem.at<uchar>(i, j) == 255)
            {
                p.x = j;
                p.y = i;
                
                floodFill(imagem, p, 0);
            }
        }
    }
}

void removeBuracos(Mat &imagem)
{
    /* Altera a cor de fundo padrão para uma nova cor. */
    floodFill(imagem, Point(0,0), 127);
    
    /* Remove todas as bolhas. */
    for (int i = 0; i < imagem.rows; i++)
    {
        for (int j = 0; j < imagem.cols; j++)
        {
            if (imagem.at<uchar>(i, j) == 0)
            {
                floodFill(imagem, Point(j,i), 255);
            }
        }
    }
    
    /* Retorna a cor de fundo par aa original. */
    floodFill(imagem, Point(0,0), 0);
}

vector<Moeda> detectarMoedas(Mat &imagem,  int fechamento)
{

    Mat imagemColorida, imagemCinza, imagemBinaria, imagemBinariaClone, imagemDelimitada;
    vector<Moeda> moedas;
    vector<vector<Point> > contornos;
    vector<Vec4i> hierarquia;
    vector<Circulo> circulos;
    
    imagemColorida = imagem.clone();

     /* Suaviza a imagem lida. */
    blur(imagemColorida, imagemColorida, Size(3, 3));
    
    /* Mantenha a imagem original em três formatos: binária, em escala de cinza 
     * e colorida. A em escala de cinza será usada na detecção de bordas. */
    cvtColor(imagemColorida, imagemCinza, CV_BGR2GRAY);
    imagemBinaria = Mat(imagemCinza.size(), imagemCinza.type());
    imagemDelimitada = imagemColorida.clone();
    
    /* Usando threshold fixo. */
    //threshold(imagemCinza, imagemBinaria, 0, 255, THRESH_BINARY_INV|THRESH_OTSU);
    
    /* Usando threshold adaptativo (resultado melhor). */
    //adaptiveThreshold(imagemCinza, imagemBinaria, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 7, 5);
    adaptiveThreshold(imagemCinza, imagemBinaria, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 5);
    
    /* Remove moedas da borda. */
    removeMoedasBorda(imagemBinaria);

    /* Faz o fechameto da imagem binária. */
    dilate(imagemBinaria, imagemBinaria, Mat(), Point(-1, -1), fechamento, 1, 1);
    erode(imagemBinaria, imagemBinaria, Mat(), Point(-1, -1), fechamento, 1, 1); 
        
    /* Remove buracos dentro das moedas (só a borda interessa). */
    removeBuracos(imagemBinaria);
    
    /* Encontra contornos iniciais para o algoritmo de active contours. */
    imagemBinariaClone = imagemBinaria.clone();
    findContours(imagemBinariaClone, contornos, hierarquia, RETR_TREE, CV_CHAIN_APPROX_NONE);
          
    circulos = vector<Circulo>(contornos.size());
      
    for(uint i = 0; i < contornos.size(); i++ )
    { 
        if( contornos[i].size() > PONTOS_CIRCULO)
        { 
            minEnclosingCircle(contornos[i], circulos[i].centro, circulos[i].raio); 
        }
    }
      
    
    for( int i = 0; i < (int) contornos.size(); i++)
    { 
        Point2f centro = circulos[i].centro;
        double raio = circulos[i].raio;
        double areaContorno = contourArea(contornos[i]);
        double areaCirculo = M_PI * raio * raio;
        double relacaoAreas = areaContorno/areaCirculo;

        if(raio > LIMIAR_RAIO && relacaoAreas > LIMIAR_RELACAO_AREAS)
        {
            
            Rect ret = boundingRect(contornos[i]);
            
            /* Calcula o negativo da região retangular formada pelos dois pontos. */
            for (int m = ret.y; m < ret.y + ret.height; m++)
            {
                for (int n = ret.x; n < ret.x + ret.width; n++)
                {
                    // Muda o fundo para branco, caso o ponto esteia fora do círculo.
                    if(pow(m - centro.y, 2) + pow(n - centro.x, 2) > pow(raio, 2))
                    {
                         imagemColorida.at<Vec3b>(m,n)[0] = 255; 
                         imagemColorida.at<Vec3b>(m,n)[1] = 255;
                         imagemColorida.at<Vec3b>(m,n)[2] = 255;
                    }
                     
                }
            }
            

            /* Cria as moedas identificadas. */
            Moeda moeda(Mat(imagemColorida, ret));
            moeda.raio = raio;
            moeda.centro = centro;
            moeda.retangulo = ret;
            moeda.contornos.push_back(contornos[i]);
            moeda.hierarquia = hierarquia;
            moedas.push_back(moeda);
            
        }
    }
     
     
    /* Exibe resultado da imagem segmentada. */
    
    imshow("binaria", imagemBinaria);
    waitKey(500);
     
    return moedas;
}

void exibirMoedas(const Mat &imagem, const vector<Moeda> &moedas)
{
    Mat imagemDelimitada = imagem.clone();
    
    if(RESULTADO_SEPARADO)
    {
        for(int i = 0; i < (int) moedas.size(); i++)
        {
            imshow("Moeda " + to_string(i), moedas[i].imagem);
        }
    }
    else
    {
        for(int i = 0; i < (int) moedas.size(); i++)
        {
            /* Desenha o retângulo e o círculo delimitador. */
            circle(imagemDelimitada, moedas[i].centro, moedas[i].raio, Scalar(0, 0, 255), 2);
            rectangle(imagemDelimitada, moedas[i].retangulo, Scalar(255, 0, 0), 2);
            putText(imagemDelimitada, "Moeda" + to_string(i), Point(moedas[i].retangulo.x-5, moedas[i].retangulo.y-5), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0), 1);
            drawContours(imagemDelimitada, moedas[i].contornos, 0, Scalar(0,255,255), 2);
        }
        
        imshow("delimitada", imagemDelimitada);
    }
    
    waitKey(500);

}


vector<Moeda> detectarTodasMoedas(Mat &imagem)
{
    Mat imagemColorida;
    vector<Moeda> moedas, moedasMax;
    int fechamento = 1;
    
    imagemColorida = imagem.clone();
    
    moedas = detectarMoedas(imagemColorida, fechamento);
    moedasMax = moedas;

    /* Tenta encontrar mais moedas e só para as tentativas se o resultado piorar. */
    while(fechamento <= MAX_FECHAMENTO && moedas.size() >= moedasMax.size())
    {   
        moedas = detectarMoedas(imagemColorida, fechamento);
        cout << "[detectar] Usando fechamento de tamanho " << fechamento << endl;
        cout << "[detectar] Foram encontradas " << moedas.size() << " moedas." << endl;
        
        if(moedas.size() > moedasMax.size())
        {
            moedasMax = moedas;
        }
        
        fechamento *= 2;
        
    }
    
    return moedasMax;
}


double calcularValor(Moeda moeda)
{
    
    Mat moedaPreparada = moeda.imagem.clone();
    
    /* Redimensiona e equaliza as moedas. */
    Mat ycrcb, resultado;
    cvtColor(moedaPreparada, ycrcb, CV_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, resultado);
    cvtColor(resultado, resultado, CV_YCrCb2BGR);
    resize(resultado, resultado, Size(400, 400), 0, 0, INTER_LINEAR);
    
    MomentosInvariantes mi = calcularMomentosInvariantes(resultado);
    
    double erro10f = 0.0, erro10n = 0.0;
    double erro25f = 0.0, erro25n = 0.0;
    double erro50f = 0.0, erro50n = 0.0;;
    double erro100f = 0.0, erro100n = 0.0;

    double valor = -1.0;
    
    for(int i = 0; i < 7; i++)
    {
        erro10f += pow((mi.h[i] - M101H[i])/mi.h[i],2) + pow((mi.s[i] - M101S[i])/mi.s[i],2) + pow((mi.v[i] - M101V[i])/mi.v[i],2);
        erro10n += pow((mi.h[i] - M102H[i])/mi.h[i],2) + pow((mi.s[i] - M102S[i])/mi.s[i],2) + pow((mi.v[i] - M102V[i])/mi.v[i],2);
        
        erro25f += pow((mi.h[i] - M251H[i])/mi.h[i],2) + pow((mi.s[i] - M251S[i])/mi.s[i],2) + pow((mi.v[i] - M251V[i])/mi.v[i],2);
        erro25n += pow((mi.h[i] - M252H[i])/mi.h[i],2) + pow((mi.s[i] - M252S[i])/mi.s[i],2) + pow((mi.v[i] - M252V[i])/mi.v[i],2);
        
        erro50f += pow((mi.h[i] - M501H[i])/mi.h[i],2) + pow((mi.s[i] - M501S[i])/mi.s[i],2) + pow((mi.v[i] - M501V[i])/mi.v[i],2);
        erro50n += pow((mi.h[i] - M502H[i])/mi.h[i],2) + pow((mi.s[i] - M502S[i])/mi.s[i],2) + pow((mi.v[i] - M502V[i])/mi.v[i],2);
        
        erro100f += pow((mi.h[i] - M1001H[i])/mi.h[i],2) + pow((mi.s[i] - M1001S[i])/mi.s[i],2) + pow((mi.v[i] - M1001V[i])/mi.v[i],2);
        erro100n += pow((mi.h[i] - M1002H[i])/mi.h[i],2) + pow((mi.s[i] - M1002S[i]/mi.s[i]),2) + pow((mi.v[i] - M1002V[i])/mi.v[i],2);
        
    }
    
    cout << endl;
    
    cout << "erro10f = " << erro10f << endl;
    cout << "erro25f = " << erro25f  << endl;
    cout << "erro50f = " << erro50f  << endl;
    cout << "erro100f = " << erro100f  << endl;
    
    cout << "erro10n = " << erro10n << endl;
    cout << "erro25n = " << erro25n  << endl;
    cout << "erro50n = " << erro50n  << endl;
    cout << "erro100n = " << erro100n  << endl;
    
    
    /*
    for(int i = 0; i < 7; i++)
    {
        erro10f += fabs((m[i] - M10_FACE[i])/M10_FACE[i]);
        erro10n += fabs((m[i] - M10_NUMERO[i])/M10_NUMERO[i]);
        
        erro25f += fabs((m[i] - M25_FACE[i])/M25_FACE[i]);
        erro25n += fabs((m[i] - M25_NUMERO[i])/M25_NUMERO[i]);
        
        erro50f += fabs((m[i] - M50_FACE[i])/M50_FACE[i]);
        erro50n += fabs((m[i] - M50_NUMERO[i])/M50_NUMERO[i]);
        
        erro100f += fabs((m[i] - M100_FACE[i])/M100_FACE[i]);
        erro100n += fabs((m[i] - M100_NUMERO[i])/M100_NUMERO[i]);
    }
    
    if (erro10f < 0.01 || erro10n < 0.01)
    {
        valor = 0.10;
    }
    else if (erro25f < 0.01 || erro25n < 0.01)
    {
        valor = 0.25;
    }
    else if (erro50f < 0.01 || erro50n < 0.01)
    {
        valor = 0.50;
    }
    else if (erro100f < 0.01 || erro100n < 0.01)
    {
        valor = 1.0;
    }
    
    cout << "erro 10f = " << erro10f << endl;
    cout << "erro 10n = " << erro10n << endl;
    
    cout << "erro 25f = " << erro25f << endl;
    cout << "erro 25n = " << erro25n << endl;
    
    cout << "erro 50f = " << erro50f << endl;
    cout << "erro 50n = " << erro50n << endl;
    
    cout << "erro 100f = " << erro100f << endl;
    cout << "erro 100n = " << erro100n << endl;
    * 
    * */
    
    cout << "-------------------" << endl;
    
    return valor;

}


double contarDinheiro(vector<Moeda> &moedas)
{
    double dinheiro = 0.0;
    
    for(int i = 0; i < (int) moedas.size(); i++)
    {
        cout << "Contando valor na moeda " << i;
        //imshow("finalMoeda" + to_string(i+100), moedas[i].imagem);
        
        dinheiro += calcularValor(moedas[i]);
    }
    
    return dinheiro;

}


/*
vector<Mat> detectarMoedasHough(vector<Mat> moedas)
{
    
    for(int i = 0; i < (int) moedas.size(); i++)
    {
        vector<Vec3f> circulos;
        cvtColor(moedas[i], moedas[i], CV_BGR2GRAY);
        
        int a = 50;
        int b = 50;
        
        while(circulos.size() != 1 && a < 800)
        {
            while(circulos.size() != 1 && b < 800)
            {
                HoughCircles(moedas[i], circulos, CV_HOUGH_GRADIENT, 1, moedas[i].rows/2, a, b, 0.9*(moedas[i].rows/2.0), moedas[i].rows);
                b += 50;
            }
            
            b = 0;
            a += 50;
        }
        
        for(int i = 0; i < (int) circulos.size(); i++)
        {
            Point centro(cvRound(circulos[i][0]), cvRound(circulos[i][1]));
            int raio = cvRound(circulos[i][2]);
            circle(moedas[i], centro, 3, Scalar(0,0,255), -1, 8, 0 );
            circle(moedas[i], centro, raio, Scalar(255,0,0), 3, 8, 0 );
            imshow("moeda" + to_string(i), moedas[i]);
            
        } 

    }

    return moedas;
}
*/
