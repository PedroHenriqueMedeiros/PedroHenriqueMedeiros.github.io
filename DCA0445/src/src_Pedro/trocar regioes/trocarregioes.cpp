#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
// a imagem é de tamanho 512x512
int main(int, char**){
  Mat image;
  Mat image2;
  Mat image3;
  Mat image4;
  Mat image5;
  Vec3i val;
  //int p1_x, p1_y, p2_x, p2_y, p3_x, p3_y;
  
  namedWindow("janela",WINDOW_AUTOSIZE);

  image= imread("lena.png",CV_LOAD_IMAGE_COLOR);
  image2= (image.rowRange(0,256),image.colRange(0,256));
  image3= (image.rowRange(0,256),image.colRange(257,512));
  image2= (image.rowRange(257,512),image.colRange(0,256));
  image2= (image.rowRange(257,512),image.colRange(257,512));
  //image2= Rect(0,0,100,100);
  cout<<image.diag();
  imshow("janela", image);  
  waitKey();
  imshow("janela1",image2);
  waitKey();
  imshow("janela2",image3);
  waitKey();
  imshow("janela3",image4);
  waitKey();
  imshow("janela4",image5);
  waitKey();
  return 0;
}
