#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <cstdlib>

using namespace std;
using namespace cv;

int top_slider = 10;
int top_slider_max = 200;

#define STEP 2
#define JITTER 3
#define RAIO 2

/* image é a imagem original; border é a imagem com apenas as bordas detectadas
 * pelo algoritmo de Canny. */
Mat image, border, points, resultado;

void on_trackbar_canny(int, void*){
  Canny(image, border, top_slider, 3*top_slider);
  uchar gray;
  resultado = points.clone();
  
  /* Recria o efeito o pointilhismo, modificando as bordas. */
  for(int i = 0; i < border.rows; i++)
  {
    for(int j = 0; j < border.cols; j++)
    {
        
        if(border.at<uchar>(i,j) == 255)
        {
            gray = resultado.at<uchar>(i,j);
            circle(resultado,
             cv::Point(j,i),
             1,
             CV_RGB(gray,gray,gray),
             -1,
             CV_AA);
        }
        
    }
   }
  
  imshow("resultado", resultado);
  imshow("canny", border);
}

int main(int argc, char**argv)
{

  vector<int> yrange;
  vector<int> xrange;

    /* Matriz points guarda o resultado dos circulos. */

  int width, height, gray;
  int x, y;

    /* Verifica o número de argumentos.  */
    if (argc != 2) 
    {
        cout << "A lista de argumentos deve ser: "
        "./cannypoints <imagem>" << endl;
        return -1;
    }
    
    /* Checa se a imagem pode ser aberta. */
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    if (!image.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }
    
    namedWindow("canny", 1);
    
    Canny(image, border, top_slider, 3*top_slider);

    createTrackbar("Treshold inferior", "canny",
                &top_slider,
                top_slider_max,
                on_trackbar_canny );

   
   /* aplica o efeito do pointilhismo. */
   
   srand(time(0));
   
   width = image.cols;
   height = image.rows;

   xrange.resize(height/STEP);
   yrange.resize(width/STEP);
    
   /* Preenche ambos os vetores com 0's. */
   iota(xrange.begin(), xrange.end(), 0);
   iota(yrange.begin(), yrange.end(), 0);

  for(uint i = 0; i < xrange.size(); i++){
    xrange[i]= xrange[i] *STEP + STEP/2;
  }

  for(uint i = 0; i < yrange.size(); i++){
    yrange[i]= yrange[i]*STEP+STEP/2;
  }

  points = Mat(height, width, CV_8U, Scalar(255));

  random_shuffle(xrange.begin(), xrange.end());

  for(auto i : xrange){
    random_shuffle(yrange.begin(), yrange.end());
    for(auto j : yrange){
      x = i+rand()%(2*JITTER)-JITTER+1;
      y = j+rand()%(2*JITTER)-JITTER+1;
      gray = image.at<uchar>(x,y);
      circle(points,
             cv::Point(y,x),
             RAIO,
             CV_RGB(gray,gray,gray),
             -1,
             CV_AA);
    }
  }
    
  on_trackbar_canny(top_slider, 0 );  
  waitKey();
  return 0;
}

