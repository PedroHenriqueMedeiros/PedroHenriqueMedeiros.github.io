#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat imagem, bordas; 
Mat imagemBorrada1, imagemBorrada3;

vector<Vec4i> hierarquia;
vector<vector<Point> > contornos;
double momentos[7];

/** @function main */
int main( int argc, char** argv )
{
    /* Verifica o número de argumentos.  */
    if (argc != 2) 
    {
        cout << "A lista de argumentos deve ser: "
        "./calculadora <imagem>" << endl;
        return -1;
    }

    /* Checa se a imagem pode ser aberta. */
    imagem = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    if (!imagem.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }
    
    /* Suaviza a imagem. */
    blur(imagem, imagemBorrada1, Size(12, 12));
    blur(imagem, imagemBorrada3, Size(5, 5));
    
    
    Mat diferenca = imagemBorrada1 - imagemBorrada3;
    bitwise_not(diferenca, diferenca);
    normalize(diferenca, diferenca, 0, 255, NORM_MINMAX);
    
    vector<Vec3f> circulos;
    
    int minDist = 120;
    
    cout << "Rows: " << imagem.rows << endl;
    cout << "Mindist: " << minDist << endl;

    HoughCircles(diferenca, circulos, CV_HOUGH_GRADIENT, 1, imagem.rows/2, minDist, 100, 0, 0);
    
    /// Draw the circles detected
    for( size_t i = 0; i < circulos.size(); i++ )
    {
        
        
      Point center(cvRound(circulos[i][0]), cvRound(circulos[i][1]));
      int radius = cvRound(circulos[i][2]);
      // circle center
      circle( diferenca, center, 3, Scalar(0,0,0), -1, 8, 0 );
      // circle outline
      circle( diferenca, center, radius, Scalar(0,0,0), 3, 8, 0 );
    }
    
    
    Moments momento = moments(imagem, false);
    HuMoments(momento, momentos);
    
    for (int i = 0; i<7; i++)
        cout << momentos[i] << endl;
    
    /*
    findContours(bordas, contornos, hierarquia, 
        CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        
    vector<Moments> mu(contornos.size());
    
    for(int i = 0; i < contornos.size(); i++)
    { 
        mu[i] = moments(contornos[i], false ); 
    }
    
    imshow("canny", bordas);
    */
    
    imshow("imagem", diferenca);
    
    //imwrite("imagem.jpg", imagem);

  waitKey(0);
  return(0);
}
