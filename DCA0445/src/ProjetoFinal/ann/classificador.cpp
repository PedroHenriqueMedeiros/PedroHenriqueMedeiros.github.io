#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>          
          
using namespace cv;
using namespace std;

// Parâmetros da MLP

#define TIPOS_MOEDAS 8
#define NUM_AMOSTRAS 60

#define NUM_MI 3

int main()
{
	initModule_nonfree();
    
    Mat entrada(1, NUM_MI, CV_32FC1);
    Mat saida(1, TIPOS_MOEDAS, CV_32FC1);
	
	Mat imagem = imread("teste.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    resize(imagem, imagem, Size(400, 400));
    equalizeHist(imagem, imagem);
    blur(imagem, imagem, Size(3,3));
    Canny(imagem, imagem, 100, 200, 3);
    
    double momentos[7];
    Moments m = moments(imagem, false);
	HuMoments(m, momentos);
    
    for(int i = 0; i < NUM_MI; i++)
    {
        entrada.at<float>(0, i) = momentos[i];
    }

	
	CvANN_MLP mlp;
	mlp.load("mlp.yml", "mlp");
    

	mlp.predict(entrada, saida);

    cout << "10f = " << saida.at<float>(0, 0) << endl;
    cout << "10n = " << saida.at<float>(0, 1) << endl;
    cout << "25f = " << saida.at<float>(0, 2) << endl;
    cout << "25n = " << saida.at<float>(0, 3) << endl;
    cout << "50f = " << saida.at<float>(0, 4) << endl;
    cout << "50n = " << saida.at<float>(0, 5) << endl;
    cout << "100f = " << saida.at<float>(0, 6) << endl;
    cout << "100n = " << saida.at<float>(0, 7) << endl;
    
    return 0;
	
}
