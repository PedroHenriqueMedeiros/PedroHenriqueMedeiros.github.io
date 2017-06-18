#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace cv::ocl;

#define PONTOS_CIRCULO 50
#define LIMIAR_RELACAO_AREAS 0.70
#define LIMIAR_RAIO 10
#define MAX_FECHAMENTO 32 // Cresce a partir de 1 em potências de 2.
#define RESULTADO_SEPARADO false

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
        Mat canny;
        imagem = _imagem.clone();
        
        /* Calcula os momentos invariantes. */
        cvtColor(imagem, moedaCinza, CV_RGB2GRAY);
        Canny(imagem, canny, 20, 60, 3);
        
        SIFT s;
        
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
vector<Moeda> detectarTodasMoedas(Mat imagem, bool adaptativo);

/* Função que detecta moedas em uma imagem, utilizando um determinado nível na
 * operação morfológica do fechamento. */
vector<Moeda> detectarMoedas(Mat imagem, int fechamento, bool adaptativo);

/* Função que desenha na imagem ou individualmente as moedas localizadas. */
void exibirMoedas(const Mat &imagem, const vector<Moeda> &moedas);

/* Função que remova moedas da borda da imagem. */
void removeMoedasBorda(Mat &imagem);

/* Função que remove buracos dentro da moeda durante a sua segmentação. */
void removeBuracos(Mat &imagem);

 

/* Função main. */
int main(int argc, char** argv) 
{
    Mat imagemColorida;
    vector<Moeda> moedas;
    
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
    moedas = detectarTodasMoedas(imagemColorida, false);
    
    if(moedas.size() < 1)
    {
		moedas = detectarTodasMoedas(imagemColorida, true);
	}
	
	Mat saida = moedas[0].imagem.clone();
	resize(saida, saida, Size(400, 400), 0, 0, INTER_LINEAR);
	imwrite(argv[1], saida);

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

vector<Moeda> detectarMoedas(Mat imagem,  int fechamento, bool adaptativo)
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
    
    
    if(adaptativo)
    {
    
		/* Usando threshold adaptativo (resultado melhor). */
		//adaptiveThreshold(imagemCinza, imagemBinaria, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 7, 5);
		adaptiveThreshold(imagemCinza, imagemBinaria, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 5);
	}
	else
	{
		/* Usando threshold fixo. */
		threshold(imagemCinza, imagemBinaria, 0, 255, THRESH_BINARY_INV|THRESH_OTSU);	
	}
    
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
            
            /* Mantém apenas os píxels da moeda e apaga os externos. 
            for (int m = ret.y; m < ret.y + ret.height; m++)
            {
                for (int n = ret.x; n < ret.x + ret.width; n++)
                {
                    // Muda o fundo para branco, caso o ponto esteia fora do círculo.
                    if(pow(m - centro.y, 2) + pow(n - centro.x, 2) > pow(raio, 2))
                    {
                         imagemColorida.at<Vec3b>(m,n)[0] = 0; 
                         imagemColorida.at<Vec3b>(m,n)[1] = 0;
                         imagemColorida.at<Vec3b>(m,n)[2] = 0;
                    }
                     
                }
            }
            * */
            

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
    
    //imshow("binaria", imagemBinaria);
    //waitKey(500);
     
    return moedas;
}

vector<Moeda> detectarTodasMoedas(Mat imagem, bool adaptativo)
{
    vector<Moeda> moedas, moedasMax;
    int fechamento = 1;
    
    moedas = detectarMoedas(imagem, fechamento, adaptativo);
    moedasMax = moedas;

    /* Tenta encontrar mais moedas e só para as tentativas se o resultado piorar. */
    while(fechamento <= MAX_FECHAMENTO && moedas.size() >= moedasMax.size())
    {   
        moedas = detectarMoedas(imagem, fechamento, adaptativo);
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
