#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

double alturaRegiaoCentral;
double forcaDecaimento;
double posicaoVerticalCentro; 
double alfa;

int slideAlturaRegiaoCentral;
int slideAlturaRegiaoCentralMax ;
int slideForcaDecaimento;
int slideForcaDecaimentoMax;
int slidePosicaoVerticalCentro;
int slidePosicaoVerticalCentroMax;

Mat imagem, imagemBorrada, imagemFinal;
char trackbarName[50];

/* Calcula a imagem final a partir da imagem original e de sua versão borrada,
com as devidas ponderações escolhidas pelo usuário. */
void calcularImagemFinal() 
{
    double xDeslocado;

    for(int i = 0; i < imagem.rows; i++)
    {
        xDeslocado = i - (posicaoVerticalCentro + alturaRegiaoCentral/2.0);
        
        alfa = 0.5 * (tanh((xDeslocado + alturaRegiaoCentral/2)/forcaDecaimento) 
            - tanh((xDeslocado - alturaRegiaoCentral/2)/forcaDecaimento));

        for(int j = 0; j < imagem.cols; j++)
        {
            imagemFinal.at<Vec3b>(i, j) = alfa * imagem.at<Vec3b>(i, j) 
                + (1-alfa)*imagemBorrada.at<Vec3b>(i, j);
        }
    }
}

void alterarSlideAlturaRegiaoCentral(int, void*)
{
    alturaRegiaoCentral = slideAlturaRegiaoCentral;
    calcularImagemFinal();
    imshow("resultado", imagemFinal);
}

void alterarSlideForcaDecaimento(int, void*)
{  
    forcaDecaimento = slideForcaDecaimento;
    if(forcaDecaimento == 0)
    {
        forcaDecaimento = 1;
    }
    calcularImagemFinal();
    imshow("resultado", imagemFinal);
}

void alterarSlidePosicaoVerticalCentro(int, void*)
{
    posicaoVerticalCentro = slidePosicaoVerticalCentro;
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
    imagem = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if (!imagem.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }

    imagemFinal = imagem.clone();
    namedWindow("resultado", 1);

    /* Primeiramente faz o borramento da imagem original. */
    blur(imagem, imagemBorrada, Size(5, 5), Point(-1,-1));
    blur(imagemBorrada, imagemBorrada, Size(5, 5), Point(-1,-1));
    blur(imagemBorrada, imagemBorrada, Size(5, 5), Point(-1,-1));

    slideAlturaRegiaoCentral = 1;
    slideForcaDecaimento = 1;
    slidePosicaoVerticalCentro = 1;

    slideAlturaRegiaoCentralMax = imagemFinal.rows;
    slidePosicaoVerticalCentroMax = imagemFinal.rows;
    slideForcaDecaimentoMax = 100;

    /* Cria as barras de rolagem. */
    createTrackbar("Altura", "resultado",
        &slideAlturaRegiaoCentral,
        slideAlturaRegiaoCentralMax,
        alterarSlideAlturaRegiaoCentral);
    alterarSlideAlturaRegiaoCentral(slideAlturaRegiaoCentral, 0);
    
    
    createTrackbar("Decaimento", "resultado",
        &slideForcaDecaimento,
        slideForcaDecaimentoMax,
        alterarSlideForcaDecaimento );
    alterarSlideForcaDecaimento(slideForcaDecaimento, 0);

    createTrackbar( "Centro", "resultado",
        &slidePosicaoVerticalCentro,
        slidePosicaoVerticalCentroMax,
        alterarSlidePosicaoVerticalCentro );
    alterarSlidePosicaoVerticalCentro(slidePosicaoVerticalCentro, 0);

    /* Fecha o programa quando o usuário digita ESC. */
    while(1)
    {
      if( waitKey(30) == 27 ) break; 
  }

  return 0;
}
