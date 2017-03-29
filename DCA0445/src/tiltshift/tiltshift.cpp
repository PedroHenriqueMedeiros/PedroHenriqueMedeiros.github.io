#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

double altura_regiao_central;
double forca_decaimento;
double posicao_vertical_centro; 
double alfa;

int slide_altura_regiao_central = 1;
int slide_altura_regiao_central_max = 100;

int slide_forca_decaimento = 1;
int slide_forca_decaimento_max = 100;

int slide_posicao_vertical_centro = 1;
int slide_posicao_vertical_centro_max = 100;

Mat imagem, imagemBorrada, imagemFinal;
char TrackbarName[50];

void calcularImagemFinal() 
{
   for(int i = 0; i < imagem.rows; i++)
   {

      alfa = 0.5 * (tanh((i + altura_regiao_central/2)/forca_decaimento) - 
        tanh((i - altura_regiao_central/2)/forca_decaimento));

      cout << altura_regiao_central << ", " << forca_decaimento << ", " << posicao_vertical_centro << ", " << alfa << endl;

      for(int j = 0; j < imagem.cols; j++)
      {
        imagemFinal.at<uchar>(i, j) = alfa * imagem.at<uchar>(i, j) + (1-alfa)*imagemBorrada.at<uchar>(i, j);
      }
   }
}

void alterar_slide_altura_regiao_central(int, void*)
{
    altura_regiao_central = (double) slide_altura_regiao_central/slide_altura_regiao_central_max;
    altura_regiao_central *= imagem.rows;

    calcularImagemFinal();
    imshow("resultado", imagemFinal);
}

void alterar_slide_forca_decaimento(int, void*)
{  
  forca_decaimento = (double) slide_forca_decaimento/slide_forca_decaimento_max;
  forca_decaimento *= 5;

  if(forca_decaimento == 0)
  {
    forca_decaimento = 0.05;
  }
  
  calcularImagemFinal();
  imshow("resultado", imagemFinal);
}

void alterar_slide_posicao_vertical_centro(int, void*)
{
  posicao_vertical_centro = (double) slide_posicao_vertical_centro/slide_posicao_vertical_centro_max;
  posicao_vertical_centro *= imagem.rows;

  calcularImagemFinal();
  imshow("resultado", imagemFinal);
}

int main(int argc, char* argv[]){


    /* Verifica o número de argumentos.  */
    if (argc != 2) 
    {
        cout << "A lista de argumentos deve ser: "
                                            "./tiltshift <imagem>" << endl;
        return -1;
    }
    
    /* Checa se a imagem pode ser aberta. */
    //imagem = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    imagem = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    imagemFinal = imagem.clone();

    if (!imagem.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }

    namedWindow("resultado", 1);
    //imshow("resultado", imagem);

    blur(imagem, imagemBorrada, Size(5, 5), Point(-1,-1));
    blur(imagemBorrada, imagemBorrada, Size(5, 5), Point(-1,-1));
    blur(imagemBorrada, imagemBorrada, Size(5, 5), Point(-1,-1));

    
    
    createTrackbar("Altura", "resultado",
            &slide_altura_regiao_central,
            slide_altura_regiao_central_max,
            alterar_slide_altura_regiao_central);
    alterar_slide_altura_regiao_central(slide_altura_regiao_central, 0);
    
    
    createTrackbar("Decaimento", "resultado",
            &slide_forca_decaimento,
            slide_forca_decaimento_max,
            alterar_slide_forca_decaimento );
    alterar_slide_forca_decaimento(slide_forca_decaimento, 0);

    createTrackbar( "Centro", "resultado",
            &slide_posicao_vertical_centro,
            slide_posicao_vertical_centro_max,
            alterar_slide_posicao_vertical_centro );
    alterar_slide_posicao_vertical_centro(slide_posicao_vertical_centro, 0);

    while(1)
    {
      if( waitKey(30) == 27 ) break; // esc pressed!
    }

  return 0;
}
