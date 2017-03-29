#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

float altura;

int alfa_slider = 0;
int alfa_slider_max = 100;

int top_slider = 0;
int top_slider_max = 100;

int altura_regiao_central = 0;
int altura_regiao_central_max = 100;

int forca_decaimento = 0;
int forca_decaimento_max = 100;

int posicao_vertical_centro = 0;
int posicao_vertical_centro_max = 100;

Mat imagem, imagemFiltrada, imagemFinal;

char TrackbarName[50];

void alterar_altura_regiao_central(int, void*)
{
   //alfa = (double) alfa_slider/alfa_slider_max ;
   //addWeighted( image1, alfa, imageTop, 1-alfa, 0.0, blended);

  altura = (double) altura_regiao_central/altura_regiao_central_max ;
  addWeighted(imagem, altura, imagemFiltrada, 1-altura, 0.0, imagemFinal);
  imshow("addweighted", imagemFinal);

}

void alterar_forca_decaimento(int, void*)
{
  imshow("addweighted", imagem);
}

void alterar_posicao_vertical_centro(int, void*)
{
imshow("addweighted", imagem);
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
    imagem = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (!imagem.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }

  //imagemFiltrada = imagem.clone();
    imagemFiltrada = imread("copacabana2.png", CV_LOAD_IMAGE_COLOR);



  namedWindow("addweighted", 1);
  
  createTrackbar("Altura", "addweighted",
				  &altura_regiao_central,
				  altura_regiao_central_max,
				  alterar_altura_regiao_central);
  alterar_altura_regiao_central(altura_regiao_central, 0);
  
  
  createTrackbar("Decaimento", "addweighted",
          &forca_decaimento,
          forca_decaimento_max,
          alterar_forca_decaimento );
  alterar_forca_decaimento(forca_decaimento, 0 );

  createTrackbar( "Centro", "addweighted",
          &posicao_vertical_centro,
          posicao_vertical_centro_max,
          alterar_posicao_vertical_centro );
  alterar_posicao_vertical_centro(posicao_vertical_centro, 0 );


  waitKey(0);
  return 0;
}
