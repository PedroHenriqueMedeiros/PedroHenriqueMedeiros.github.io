// Neste programa vamos mostrar um método bem simples de detecção de movimento envolvendo dois frames capturados pela câmera: um de base e outro sendo o frame atual
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cv.h>
#include <time.h>
using namespace cv;
using namespace std;

int main(int argc, char** argv){
  /* Matriz para imagem atual*/
  Mat image;
  /*Matriz para a imagem base*/
  Mat base;
  /*Variável acumulador que é responsável por verificar a passagem do tempo utilzando de funções de para medir o número de ciclos e as frequências dos mesmos(do sistema operacional)*/
  double acumulador=0;
  
  int width, height;
  VideoCapture cap;
  /*Vetor para separar os planos RGB para os frames base e atual*/
  vector<Mat> planes, planes2;
  /*Cálculos dos histogramas das componentes Red, Green e Blue de cada frame.*/ 
  Mat histR, histR2, histG, histG2, histB, histB2;
  int nbins = 64;
  /*Faixa de valores do histograma*/
  float range[] = {0, 256};
  const float *histrange = { range };
  const float *histrange2 = { range };
  bool uniform = true;
  bool acummulate = false;
 
  cap.open(0);
  /*Verifica se a câmera do computador está disponível*/
  if(!cap.isOpened()){
    cout << "cameras indisponiveis";
    return -1;
  }
  
  /*Parâmetros de largura e altura do frame capturado pela câmera*/ 
  width  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

  cout << "largura = " << width << endl;
  cout << "altura  = " << height << endl;

  int histw = nbins, histh = nbins/2;
  

  /*Define-se a largura e altura das imagens que serão usadas para desenhar os histogramas de cada uma das componentes de cor. As imagens são criadas com o tipo CV_8UC3, ou seja, com 8 bits por pixel, com tipo de dados unsigned char contendo 3 canais de cor. A cor, nesse caso, servirá apenas para que o histograma seja desenhado na cor respectiva de sua componente.*/

  
  Mat histImgR(histh, histw, CV_8UC3, Scalar(0,0,0));
  Mat histImgG(histh, histw, CV_8UC3, Scalar(0,0,0));
  Mat histImgB(histh, histw, CV_8UC3, Scalar(0,0,0));

  Mat histImgR2(histh, histw, CV_8UC3, Scalar(0,0,0));
  Mat histImgG2(histh, histw, CV_8UC3, Scalar(0,0,0));
  Mat histImgB2(histh, histw, CV_8UC3, Scalar(0,0,0));
  double t;
  /*A função getTickCount retorna o número de ticks que a CPU emitiu  desde um determinado evento e a função getTickFrequency diz quantas vezes a CPU emitiu o tick em um segundo. Dessa forma calculamos o tempo.*/
   t = (double)getTickCount();

   t = ((double)getTickCount() - t)/getTickFrequency();
   cap >> base;
   
  while(1){
    acumulador= acumulador+t;
    cap >> image;
    split (base, planes2);
    /*Cálculo de histogramas*/
    calcHist(&planes2[0], 1, 0, Mat(), histR2, 1,
             &nbins, &histrange2,
             uniform, acummulate);
    calcHist(&planes2[1], 1, 0, Mat(), histG2, 1,
             &nbins, &histrange2,
             uniform, acummulate);
    calcHist(&planes2[2], 1, 0, Mat(), histB2, 1,
             &nbins, &histrange2,
             uniform, acummulate);
   
    /*Dividir a imagem nos planos RGB*/
    split (image, planes);
    
   
    calcHist(&planes[0], 1, 0, Mat(), histR, 1,
             &nbins, &histrange,
             uniform, acummulate);
    calcHist(&planes[1], 1, 0, Mat(), histG, 1,
             &nbins, &histrange,
             uniform, acummulate);
    calcHist(&planes[2], 1, 0, Mat(), histB, 1,
             &nbins, &histrange,
             uniform, acummulate);
    
    
    /*Normalizar os histogramas*/
    normalize(histR, histR, 0,1, NORM_MINMAX, -1, Mat());
    normalize(histG, histG, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(histB, histB, 0, 1, NORM_MINMAX, -1, Mat());

    normalize(histR2, histR2, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(histG2, histG2, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(histB2, histB2, 0, 1, NORM_MINMAX, -1, Mat());
    
    histImgR.setTo(Scalar(0));
    histImgG.setTo(Scalar(0));
    histImgB.setTo(Scalar(0));

    histImgR2.setTo(Scalar(0));
    histImgG2.setTo(Scalar(0));
    histImgB2.setTo(Scalar(0));

    /*Desenhar os histogramas no frame*/
     for(int i=0; i<nbins; i++){
      line(histImgR,
           Point(i, histh),
           Point(i, histh-cvRound(histR.at<float>(i))),
           Scalar(0, 0, 255), 1, 8, 0);
      line(histImgG,
           Point(i, histh),
           Point(i, histh-cvRound(histG.at<float>(i))),
           Scalar(0, 255, 0), 1, 8, 0);
      line(histImgB,
           Point(i, histh),
           Point(i, histh-cvRound(histB.at<float>(i))),
           Scalar(255, 0, 0), 1, 8, 0);
    }

     
     for(int i=0; i<nbins; i++){
      line(histImgR2,
           Point(i, histh),
           Point(i, histh-cvRound(histR.at<float>(i))),
           Scalar(0, 0, 255), 1, 8, 0);
      line(histImgG2,
           Point(i, histh),
           Point(i, histh-cvRound(histG.at<float>(i))),
           Scalar(0, 255, 0), 1, 8, 0);
      line(histImgB2,
           Point(i, histh),
           Point(i, histh-cvRound(histB.at<float>(i))),
           Scalar(255, 0, 0), 1, 8, 0);
    }
      
    histImgR.copyTo(image(Rect(0, 0       ,nbins, histh)));
    histImgG.copyTo(image(Rect(0, histh   ,nbins, histh)));
    histImgB.copyTo(image(Rect(0, 2*histh ,nbins, histh)));

    histImgR2.copyTo(base(Rect(0, 0       ,nbins, histh)));
    histImgG2.copyTo(base(Rect(0, histh   ,nbins, histh)));
    histImgB2.copyTo(base(Rect(0, 2*histh ,nbins, histh)));
    
   
    /*Utiliza o método da correlação para comparar os histogramas RED de cada frame*/
    
     double base_test1 = compareHist(histR2, histR, 1 );
    

     cout<<"Resultado do método da correlação"<<base_test1<<"\n";
     
      t = (double)getTickCount();

   t = ((double)getTickCount() - t)/getTickFrequency();
   /*Verifica se a variável acumulador chegou ao valor preestabelicido pelo programador*/
   if(acumulador>0.001){
     cap>>base;
   }
   if(acumulador>0.1){
     acumulador=0;
   }
   acumulador=acumulador*100;
   cout<<acumulador<<"\n";
    imshow("base",base);
    imshow("image", image);
   
   
    if(waitKey(30)>=0)break;
   
  }
  return 0;
}
