#include <iostream>
#include <opencv2/opencv.hpp>
#include  <time.h>  // time()
#include <math.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  Mat image;
  int width, height;
  VideoCapture cap;
  vector<Mat> planes;
  Mat histR,histRef;
  int nbins = 64;
  float range[] = {0, 256};
  const float *histrange = { range };
  bool uniform = true;
  bool acummulate = false;

  bool catchFrame = true;
  bool alarm = false;
  double clock = 0;
  double clockTemp = 0;
  double alarmTime = 0;
  float calibration = 45;
  float changes = 0;

  cap.open(0);
  
  if(!cap.isOpened()){
    cout << "cameras indisponiveis";
    return -1;
  }
  
  width  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

  cout << "Entre com o tempo do alarme em segundos: ";
  cin >> alarmTime;

  cout << "largura = " << width << endl;
  cout << "altura  = " << height << endl;

  int histw = nbins, histh = nbins/2;
  Mat histImgR(histh, histw, CV_8UC3, Scalar(0,0,0));

  while(1){
    cap >> image;
    split (image, planes);
    calcHist(&planes[0], 1, 0, Mat(), histR, 1,
             &nbins, &histrange,
             uniform, acummulate);

    normalize(histR, histR, 0, histImgR.rows, NORM_MINMAX, -1, Mat());

    if(catchFrame){
	histRef = histR.clone(); // Captura o histograma do frame de referência;
	catchFrame = false;
    	}

    histImgR.setTo(Scalar(0));
    
    for(int i=0; i<nbins; i++){

      changes += abs(histRef.at<float>(i)-histR.at<float>(i)); //acumula para calcular diferenca media

      line(histImgR, Point(i, histh),
           Point(i, cvRound(histR.at<float>(i))),
           Scalar(0, 0, 255), 1, 8, 0);
    	}

    if(changes >= calibration) { //se a diferenca for igual ou acima do limite permitido
	alarm = true;
	clock = 0;
	clockTemp = (double)getTickCount();
    	}

   if(alarm) {  //Caso a alarme seja ativado;
	catchFrame = true;
	circle(image,Point(image.cols/2,image.rows/2),200, Scalar(0,0,255),-200);
	clock = ((double)getTickCount() - clockTemp)/getTickFrequency();
   	}

   if(clock >= alarmTime){  //Desativar o alarme após um certo tempo;
	alarm = false;
        }

   changes = 0;

   histImgR.copyTo(image(Rect(0,0 ,nbins, histh)));
   imshow("image", image);
   if(waitKey(30) >= 0) break;

   }
  return 0;
}

