#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>          
          
using namespace cv;
using namespace std;

// Parâmetros da MLP

#define TIPOS_MOEDAS 3 // 25, 50 e 1 real.
#define NUM_AMOSTRAS 20 // 24 fotos de cada moeda.

// Parâmetros do histograma
#define NUM_NIVEIS_MATIZ 32
#define NUM_NIVEIS_SATURACAO 128
#define HIST_UNIFORME true
#define HIST_ACUMULADO false

// Momentos invariantes
#define NUM_MOMENTOS 3

void equalizarHistograma(Mat& imagem)
{
    if(imagem.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(imagem, ycrcb, CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb, channels);

        equalizeHist(channels[0], channels[0]);

        merge(channels, ycrcb);

        cvtColor(ycrcb, imagem, CV_YCrCb2BGR);
    }
}

int main(int argc, char** argv)
{
	initModule_nonfree();
    
    Mat entrada(1, NUM_NIVEIS_MATIZ + 2 + NUM_MOMENTOS, CV_32FC1);
    Mat saida(1, TIPOS_MOEDAS, CV_32FC1);
	
    Mat imagem;
    
    /* Verifica o número de argumentos.  */
    if (argc != 2)
    {
        cout << "A lista de argumentos deve ser: "
            "./testador <imagem>"<< endl;
        return -1;
    }

    /* Checa se a imagem pode ser aberta. */
    imagem = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    equalizarHistograma(imagem);
    
    Mat moedaCinza;;
    cvtColor(imagem, moedaCinza, CV_BGR2GRAY);
    
    Moments m = moments(moedaCinza, false);
    double hu[7];
    HuMoments(m, hu);
    
    cvtColor(imagem, imagem, CV_BGR2HSV);
    
    int histSize[] = {NUM_NIVEIS_MATIZ};
    float hranges[] = {0, 180}; // hue varies from 0 to 179, see cvtColor
    const float* ranges[] = {hranges};
    int channels[] = {0};
    
    Mat hist;
    calcHist(&imagem, 1, channels, Mat(), hist, 1, histSize, ranges, HIST_UNIFORME, HIST_ACUMULADO);
    
    for(int i = 0; i < NUM_NIVEIS_MATIZ; i++)
    {
        entrada.at<float>(0,i) = hist.at<float>(i,0);
    }
    
    entrada.at<float>(0,NUM_NIVEIS_MATIZ) = imagem.rows;
    entrada.at<float>(0,NUM_NIVEIS_MATIZ+1) = imagem.cols;
    
    for(int i = NUM_NIVEIS_MATIZ + 2; i < NUM_NIVEIS_MATIZ + 2 + NUM_MOMENTOS; i++)
    {
        entrada.at<float>(0,i) = hu[i-NUM_NIVEIS_MATIZ+2];
    }
    
    
	CvANN_MLP mlp;
	mlp.load("mlp.yml", "mlp");
    

	mlp.predict(entrada, saida);

    cout << "25 = " << saida.at<float>(0, 0) << endl;
    cout << "50 = " << saida.at<float>(0, 1) << endl;
    cout << "100 = " << saida.at<float>(0, 2) << endl;
    
    return 0;
	
}
