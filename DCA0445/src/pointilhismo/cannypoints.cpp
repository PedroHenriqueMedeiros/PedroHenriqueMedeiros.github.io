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

int threshold_slider = 10;
int threshold_slider_max = 200;

int step_slider = 1;
int step_slider_max = 10;

int jitter_slider = 3;
int jitter_slider_max = 10;

int raio_slider = 2;
int raio_slider_max = 10;

#define STEP 2
#define JITTER 3
#define RAIO 2

/* image é a imagem original; border é a imagem com apenas as bordas detectadas
 * pelo algoritmo de Canny. */
Mat image, border, points, resultado;

void alterarSliderCanny(int, void*)
{
  Canny(image, border, threshold_slider, 3*threshold_slider);
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
  
  imshow("canny", border);
  imshow("resultado", resultado);
  
}



void alterarSliderPointilhismo(int, void*)
{
    vector<int> yrange;
    vector<int> xrange;
    int width, height, gray;
    int x, y;
    srand(time(0));
   
       width = image.cols;
       height = image.rows;

       xrange.resize(height/step_slider);
       yrange.resize(width/step_slider);
        
       /* Preenche ambos os vetores com 0's. */
       iota(xrange.begin(), xrange.end(), 0);
       iota(yrange.begin(), yrange.end(), 0);

      for(uint i = 0; i < xrange.size(); i++){
        xrange[i]= xrange[i] *step_slider + step_slider/2;
      }

      for(uint i = 0; i < yrange.size(); i++){
        yrange[i]= yrange[i]*step_slider+step_slider/2;
      }

      points = Mat(height, width, CV_8U, Scalar(255));

      random_shuffle(xrange.begin(), xrange.end());

      for(auto i : xrange){
        random_shuffle(yrange.begin(), yrange.end());
        for(auto j : yrange){
          x = i+rand()%(2*jitter_slider)-jitter_slider+1;
          y = j+rand()%(2*jitter_slider)-jitter_slider+1;
          gray = image.at<uchar>(x,y);
          circle(points,
                 cv::Point(y,x),
                 raio_slider,
                 CV_RGB(gray,gray,gray),
                 -1,
                 CV_AA);
        }
      }
    
    
    imshow("pointilhismo", points);
    
    alterarSliderCanny(0, 0);
    imshow("resultado", resultado);
}



int main(int argc, char**argv)
{



    /* Matriz points guarda o resultado dos circulos. */



    /* Verifica o número de argumentos.  */
    if (argc != 2) 
    {
        cout << "A lista de argumentos deve ser: "
        "./cannypoints <imagem>" << endl;
        return -1;
    }
    
    /* Checa se a imagem pode ser aberta. */
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    resultado = image.clone();
    points = image.clone();
    border = image.clone();
    
    if (!image.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }
    
    namedWindow("canny", 1);
    namedWindow("pointilhismo", 1);
    namedWindow("resultado", 1);
    
    Canny(image, border, threshold_slider, 3*threshold_slider);

    createTrackbar("Treshold inferior", "canny",
                &threshold_slider,
                threshold_slider_max,
                alterarSliderCanny);
                
    createTrackbar("Step", "pointilhismo",
                &step_slider,
                step_slider_max,
                alterarSliderPointilhismo);
                
    createTrackbar("Jitter", "pointilhismo",
            &jitter_slider,
            jitter_slider_max,
            alterarSliderPointilhismo);
            
    createTrackbar("Raio", "pointilhismo",
            &raio_slider,
            raio_slider_max,
            alterarSliderPointilhismo);

   
   /* aplica o efeito do pointilhismo. */
   
  
    
    
  alterarSliderPointilhismo(0, 0);
  alterarSliderCanny(threshold_slider, 0 );
  
  waitKey();
  return 0;
}

