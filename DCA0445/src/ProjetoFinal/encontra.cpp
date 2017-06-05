#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include "opencv2/legacy/legacy.hpp"

using namespace std;
using namespace cv;

#define PONTOS_CIRCULO 50
#define LIMIAR_RELACAO_AREAS 0.70
#define LIMIAR_RAIO 10


struct Circulo {
    Point2f centro;
    float raio;
};

void removeMoedasBorda(Mat &imagem);
void removeBuracos(Mat &imagem);

int main(int argc, char** argv) 
{
    Mat imagemColorida, imagemCinza, imagemBinaria, imagemBinariaClone;
    Mat imagemDelimitada;
    vector<Mat> moedas;
    vector<vector<Point> > contornos;
    vector<Vec4i> hierarquia;
    vector<Circulo> circulos;
    
     /* Verifica o número de argumentos.  */
    if (argc != 2)
    {
        cout << "A lista de argumentos deve ser: "
            "./encontra <imagem>"<< endl;
        return -1;
    }

    /* Checa se a imagem pode ser aberta. */
    imagemColorida = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if (!imagemColorida.data)
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }
    
    /* Suaviza a imagem lida. */
    blur(imagemColorida, imagemColorida, Size(3, 3));
    
    /* Mantenha a imagem original em três formatos: binária, em escala de cinza 
     * e colorida. A em escala de cinza será usada na detecção de bordas. */
    cvtColor(imagemColorida, imagemCinza, CV_BGR2GRAY);
    imagemBinaria = Mat(imagemCinza.size(), imagemCinza.type());
    imagemDelimitada = Mat::zeros(imagemColorida.size(), CV_8UC1);
    
    /* Usando threshold fixo. */
    //threshold(imagemCinza, imagemBinaria, 0, 255, THRESH_BINARY_INV|THRESH_OTSU);
    
    /* Usando threshold adaptativo (resultado melhor). */
    //adaptiveThreshold(imagemCinza, imagemBinaria, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 7, 5);
    adaptiveThreshold(imagemCinza, imagemBinaria, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 5);
    
    /* Remove moedas da borda. */
    removeMoedasBorda(imagemBinaria);

    /* Faz o fechameto da imagem binária. */
    dilate(imagemBinaria, imagemBinaria, Mat(), Point(-1, -1), 8, 1, 1);
    erode(imagemBinaria, imagemBinaria, Mat(), Point(-1, -1), 8, 1, 1); 
        
    /* Remove buracos dentro das moedas (só a borda interessa). */
    removeBuracos(imagemBinaria);
    
    /* Encontra contornos iniciais para o algoritmo de active contours. */
    imagemBinariaClone = imagemBinaria.clone();
    findContours(imagemBinariaClone, contornos, hierarquia, RETR_TREE, CV_CHAIN_APPROX_NONE);
      
    /* Desenha os contornos detectado pela função anterior. */
    for(uint i = 0; i < contornos.size(); i++ )
    {
        Scalar color = Scalar(63, 63, 63);
        drawContours(imagemDelimitada, contornos, i, color, 2, 8, hierarquia, 0, Point() );
    }
          
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
                    // Muda o fundo para branco, caso o ponto esteja fora do círculo.
                    if(pow(m - centro.y, 2) + pow(n - centro.x, 2) > pow(raio, 2))
                    {
                         imagemColorida.at<Vec3b>(m,n)[0] = 255; 
                         imagemColorida.at<Vec3b>(m,n)[1] = 255;
                         imagemColorida.at<Vec3b>(m,n)[2] = 255;
                    }
                     
                }
            }
            
            /* Cria as moedas identificadas. */
           Mat moeda(imagemColorida, ret);
           moedas.push_back(moeda);
           
           // Desenha o retângulo e o círculo delimitador.
           circle(imagemDelimitada, centro, raio, Scalar(191, 191, 191));
           rectangle(imagemDelimitada, ret, Scalar(255, 255, 255));
           
        }
     }
     
     /* Exibindo moedas identificadas e calculando os momentos invariantes. */
     for( int i = 0; i < (int) moedas.size(); i++)
     {
        Mat moedaCinza;
        double momentos[7];
        
        cvtColor(moedas[i], moedaCinza,CV_RGB2GRAY);
        Moments momento = moments(moedaCinza, false);
        HuMoments(momento, momentos);
         
         
         string titulo = "Moeda " + to_string(i);
         imshow(titulo, moedas[i]);
         cout << "--------------------------" << endl;
         cout << "Momentos invariantes da moeda " << i << ": " << endl;
         cout << "hu[0] = " << momentos[0] << endl;
         cout << "hu[1] = " << momentos[1] << endl;
         cout << "hu[2] = " << momentos[2] << endl;
         cout << "hu[3] = " << momentos[3] << endl;
         cout << "hu[4] = " << momentos[4] << endl;
         cout << "hu[5] = " << momentos[5] << endl;
         cout << "hu[6] = " << momentos[6] << endl;
         cout << "--------------------------" << endl;
         
     }

     
    imshow("binaria", imagemBinaria);
    imshow("delimitada", imagemDelimitada);
    
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


