#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

VideoCapture cap;
//VideoWriter saida;

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

int counter = 0;

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

int main(int argc, char* argv[]){

    /* Verifica o número de argumentos.  */
    if (argc != 2) 
    {
        cout << "A lista de argumentos deve ser: "
        "./tiltshiftvideo <video>" << endl;
        return -1;
    }
    
    /* Checa se o video pode ser aberto. */

    cap.open(argv[1]);
    if(!cap.isOpened()){
        cout << "O vídeo não pôde ser aberta." << endl;
        return -2;
    }

    int largura = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int altura = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int fourcc = cap.get(CV_CAP_PROP_FOURCC);
    double fps = cap.get(CV_CAP_PROP_FPS);

    VideoWriter saida("saida.mpg", CV_FOURCC('P','I','M','1'), fps, Size(largura, altura));

    //namedWindow("resultado", 1);

    alturaRegiaoCentral = 100;
    forcaDecaimento = 30;
    posicaoVerticalCentro = 0.5*altura;

    while(1)
    {
        cap >> imagem; 
        imagemFinal = imagem.clone();
        if (imagem.empty()) break;

        blur(imagem, imagemBorrada, Size(3, 3), Point(-1,-1));
        blur(imagemBorrada, imagemBorrada, Size(3, 3), Point(-1,-1));
        blur(imagemBorrada, imagemBorrada, Size(3, 3), Point(-1,-1));
        blur(imagemBorrada, imagemBorrada, Size(3, 3), Point(-1,-1));

        calcularImagemFinal();
        saida << imagemFinal;
        //imshow("resultado", imagemFinal);
        //if(waitKey(30) >= 0) break; 
    }

    cap.release();
    saida.release();
    
    cout << "Processamento do vídeo concluído." << endl;

    /*
    slideAlturaRegiaoCentral = 1;
    slideForcaDecaimento = 1;
    slidePosicaoVerticalCentro = 1;

    slideAlturaRegiaoCentralMax = imagemFinal.rows;
    slidePosicaoVerticalCentroMax = imagemFinal.rows;
    slideForcaDecaimentoMax = 100;

     */

    /* Cria as barras de rolagem. 
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

    */

    /* Fecha o programa quando o usuário digita ESC. */

  return 0;
}
