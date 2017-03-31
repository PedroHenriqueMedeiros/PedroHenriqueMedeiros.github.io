#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

double alturaRegiaoCentral;
double forcaDecaimento;
double posicaoVerticalCentro; 
double alfa;

int sliderAlturaRegiaoCentral;
int sliderAlturaRegiaoCentralMax ;
int sliderForcaDecaimento;
int sliderForcaDecaimentoMax;
int sliderPosicaoVerticalCentro;
int sliderPosicaoVerticalCentroMax;

Mat imagem, imagemBorrada, imagemFinal;
char nomeTrackbar[50];

/* Faz o borramento da imagem utilizando filtro da média. */
void borrarImagem()
{
    blur(imagem, imagemBorrada, Size(3, 3), Point(-1,-1));
    blur(imagemBorrada, imagemBorrada, Size(3, 3), Point(-1,-1));
    blur(imagemBorrada, imagemBorrada, Size(3, 3), Point(-1,-1));
    blur(imagemBorrada, imagemBorrada, Size(3, 3), Point(-1,-1));
    blur(imagemBorrada, imagemBorrada, Size(3, 3), Point(-1,-1));
    blur(imagemBorrada, imagemBorrada, Size(3, 3), Point(-1,-1));
}

/* Aumenta a saturação do quadro atual. */
void aumentarSaturacao()
{   
    Mat imagemHSV;
    Mat hsv[3];
   
    cvtColor(imagemFinal, imagemHSV, COLOR_BGR2HSV);

    split(imagemHSV, hsv);
    hsv[1] = hsv[1] * 1.6;
        
    merge(hsv, 3, imagemHSV);
    
    cvtColor(imagemHSV, imagemFinal, COLOR_HSV2BGR);
}

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

void alterarSliderAlturaRegiaoCentral(int, void*)
{
    alturaRegiaoCentral = sliderAlturaRegiaoCentral;
    calcularImagemFinal();
    aumentarSaturacao();
    imshow("resultado", imagemFinal);
}

void alterarSliderForcaDecaimento(int, void*)
{  
    forcaDecaimento = sliderForcaDecaimento;
    if(forcaDecaimento == 0)
    {
        forcaDecaimento = 1;
    }
    calcularImagemFinal();
    aumentarSaturacao();
    imshow("resultado", imagemFinal);
}

void alterarSliderPosicaoVerticalCentro(int, void*)
{
    posicaoVerticalCentro = sliderPosicaoVerticalCentro;
    calcularImagemFinal();
    aumentarSaturacao();
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
    borrarImagem();

    sliderAlturaRegiaoCentral = 1;
    sliderForcaDecaimento = 1;
    sliderPosicaoVerticalCentro = 1;

    sliderAlturaRegiaoCentralMax = imagemFinal.rows;
    sliderPosicaoVerticalCentroMax = imagemFinal.rows;
    sliderForcaDecaimentoMax = 100;

    /* Cria as barras de rolagem. */
    createTrackbar("Altura", "resultado",
        &sliderAlturaRegiaoCentral,
        sliderAlturaRegiaoCentralMax,
        alterarSliderAlturaRegiaoCentral);
    alterarSliderAlturaRegiaoCentral(sliderAlturaRegiaoCentral, 0);
    
    createTrackbar("Decaimento", "resultado",
        &sliderForcaDecaimento,
        sliderForcaDecaimentoMax,
        alterarSliderForcaDecaimento );
    alterarSliderForcaDecaimento(sliderForcaDecaimento, 0);

    createTrackbar( "Centro", "resultado",
        &sliderPosicaoVerticalCentro,
        sliderPosicaoVerticalCentroMax,
        alterarSliderPosicaoVerticalCentro );
    alterarSliderPosicaoVerticalCentro(sliderPosicaoVerticalCentro, 0);

    /* Fecha o programa quando o usuário digita ESC. */
    while(1)
    {
      if(waitKey(30) >= 0) 
      {
        break;
      } 
    }
    
    /* Salva a imagem processada. */
    imwrite("saida.png", imagemFinal);
    

  return 0;
}
