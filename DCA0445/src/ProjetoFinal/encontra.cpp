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


struct Circulo {
    Point2f centro;
    float raio;
};

vector<Mat> detectaMoedas(Mat &imagem, bool exibirDelimitada = true);
vector<Mat> detectaMoedasHough(Mat &imagem, bool exibirDelimitada = true);
void removeMoedasBorda(Mat &imagem);
void removeBuracos(Mat &imagem);

int main(int argc, char** argv) 
{
    Mat imagemColorida;
    vector<Mat> moedas;
    stringstream saida;
    
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
    
    //moedas = detectaMoedas(imagemColorida);
    
    moedas = detectaMoedasHough(imagemColorida);

     /* Exibe as moedas identificadas. */
    for( int i = 0; i < (int) moedas.size(); i++)
    {
        string titulo = "Moeda " + to_string(i);
        imshow(titulo, moedas[i]);
        waitKey(100);
    }
     
     
    /* Calculando os momentos invariantes e associando-os aos valores. */
    for( int i = 0; i < (int) moedas.size(); i++)
    {
        Mat moedaCinza;
        double momentos[7];
        unsigned int valor;

        cvtColor(moedas[i], moedaCinza,CV_RGB2GRAY);
        Moments momento = moments(moedaCinza, false);
        HuMoments(momento, momentos);

        /* Salvando resultado em arquivo. */
        cout << "Digite o valor da moeda " << i << ": ";
        cin >> valor;

        /* Armazena o valor da moeda e seus momentos invariantes. */

        saida << valor << ",";

        for(int j = 0; j < 7; j++)
        {    
            saida << scientific << setprecision(7) << momentos[j];
            if (j < 6)
            {
                saida << ",";
            }
        }

        saida << endl;
        
    }
     
     /* Exibe o resultado final. */
     
    cout << endl << "Resultado final (valor da moeda seguido dos momentos invariantes): " << endl << endl;
    cout << "valor,monento0,momento1,momento2,momento3,momento4,momento5,momento6" << endl;
    cout << saida.str() << endl;


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

vector<Mat> detectaMoedas(Mat &imagemColorida, bool exibirDelimitada)
{

    Mat imagemCinza, imagemBinaria, imagemBinariaClone, imagemDelimitada;
    vector<Mat> moedas;
    vector<vector<Point> > contornos;
    vector<Vec4i> hierarquia;
    vector<Circulo> circulos;

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
        Scalar color = Scalar(0, 255, 255);
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
           circle(imagemDelimitada, centro, raio, Scalar(0, 0, 255), 2);
           rectangle(imagemDelimitada, ret, Scalar(255, 0, 0), 2);
           
        }
    }
     
     
    /* Exibe resultado da delimitação. */
    if(exibirDelimitada)
    {
        imshow("delimitada", imagemDelimitada);
    }
     
    return moedas;
}


vector<Mat> detectaMoedasHough(Mat &imagemColorida, bool exibirDelimitada)
{
    Mat imagemCinza, imagemDelimitada;
    vector<Vec3f> circulos;
    vector<Mat> moedas;
    
    cvtColor(imagemColorida, imagemCinza, CV_BGR2GRAY);
    imagemDelimitada = imagemColorida.clone();

    HoughCircles(imagemCinza, circulos, CV_HOUGH_GRADIENT, 1, imagemCinza.rows/8, 50, 400, 0, 0);

    for(int i = 0; i < (int) circulos.size(); i++ )
    {
        Point centro(cvRound(circulos[i][0]), cvRound(circulos[i][1]));
        int raio = cvRound(circulos[i][2]);
        circle(imagemDelimitada, centro, 3, Scalar(0,0,255), -1, 8, 0 );
        circle(imagemDelimitada, centro, raio, Scalar(255,0,0), 3, 8, 0 );
    } 
    
    /* Exibe resultado da delimitação. */
    if(exibirDelimitada)
    {
        imshow("delimitada", imagemDelimitada);
    }

    imshow("delimitada", imagemDelimitada);
    waitKey(0);
    
    return moedas;
}
     
