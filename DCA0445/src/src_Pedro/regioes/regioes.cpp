#include <iostream>
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;
// a imagem é de tamanho 512x512
int main(int, char**){
  Mat image;
  Vec3b val;
  int p1_x;
  int p1_y;
  int p2_x;
  int p2_y;
  cout<< "Coloque a coordenada X do ponto P1\n";
  cin>> p1_x;
  cout<< "\n Coloque a coordenada Y do ponto P1\n";
  cin>> p1_y;
  cout<< "\n Coloque a coordenada X do ponto P2\n";
  cin>> p2_x;
  cout<< "\nColoque a coordenada Y do ponto P2\n";
  cin>> p2_y;
  
  namedWindow("janela",WINDOW_AUTOSIZE);

  image= imread("lena.png",CV_LOAD_IMAGE_COLOR);

  val[0] = 255;   //B
  val[1] = 255;   //G
  val[2] = 255; //R
  
  for(int i=p1_x;i<p2_x;i++){
    for(int j=p1_y;j<p2_y;j++){
      image.at<Vec3b>(i,j)=val-image.at<Vec3b>(i,j);
    }
  }

  imshow("janela", image);  
  waitKey();
  return 0;
}
