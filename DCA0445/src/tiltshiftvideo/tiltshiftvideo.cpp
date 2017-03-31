#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

VideoCapture cap;
<<<<<<< HEAD
//VideoWriter saida;
=======
VideoWriter saida;
>>>>>>> 675d11c127f56246aed30cf4bba07a1a3a95cc71

double alturaRegiaoCentral;
double forcaDecaimento;
double posicaoVerticalCentro; 
double alfa;

<<<<<<< HEAD
int slideAlturaRegiaoCentral;
int slideAlturaRegiaoCentralMax ;
int slideForcaDecaimento;
int slideForcaDecaimentoMax;
int slidePosicaoVerticalCentro;
int slidePosicaoVerticalCentroMax;

int counter = 0;

Mat imagem, imagemBorrada, imagemFinal;
char trackbarName[50];
=======
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
>>>>>>> 675d11c127f56246aed30cf4bba07a1a3a95cc71

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

<<<<<<< HEAD
int main(int argc, char* argv[]){
=======
void alterarSliderAlturaRegiaoCentral(int, void*)
{
    alturaRegiaoCentral = sliderAlturaRegiaoCentral;
    calcularImagemFinal();
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
    imshow("resultado", imagemFinal);
}

void alterarSliderPosicaoVerticalCentro(int, void*)
{
    posicaoVerticalCentro = sliderPosicaoVerticalCentro;
    calcularImagemFinal();
    imshow("resultado", imagemFinal);
}

int main(int argc, char* argv[])
{
    int largura, altura, fourcc, contador;
    double fps, quantidadeTotalQuadros, quadroAtual, percentagem;
>>>>>>> 675d11c127f56246aed30cf4bba07a1a3a95cc71

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
<<<<<<< HEAD

    int largura = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int altura = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    cout << largura << " " << altura << endl;

    VideoWriter saida("saida.mpg", CV_FOURCC('M','J','P','G'), 20.0, Size(largura, altura));

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
=======
    
    /* Obtém algumas propriedades do vídeo carregado. */
    largura = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    altura = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    fourcc = cap.get(CV_CAP_PROP_FOURCC);
    fps = cap.get(CV_CAP_PROP_FPS);
    quantidadeTotalQuadros = cap.get(CV_CAP_PROP_FRAME_COUNT);
    
    namedWindow("resultado");
    
    /* Utiliza o primeiro frame como a imagem base para ajustar a região do 
     * efeito do tiltshift. */
    cap >> imagem; 
    imagemFinal = imagem.clone();
    borrarImagem();
    calcularImagemFinal();
    aumentarSaturacao();
        
    sliderAlturaRegiaoCentralMax = altura;
    sliderPosicaoVerticalCentroMax = altura;
    sliderForcaDecaimentoMax = 100;

    /* Cria as barras de rolagem.  */
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
    
    cout << "[1] Selecionando regiões para efeito de tilt-shift..." << endl ;
    
    /* Abre a janela para o usuário conseguir selecionar a região de efeito
     * do tilt-shift. */
    while(1)
    {
        imshow("resultado", imagemFinal);
        /* If the values are set, then start video conversion. */
        if(waitKey(30) >= 0) 
        {
            destroyWindow("resultado");
            
            /* Única forma da janela fechar. */
            waitKey(1);
            waitKey(1);
            waitKey(1);
            waitKey(1);
            break; 
        }
    }
    
    /* Sobrescrevendo por falta de suporte ao MP4. */
    fourcc = CV_FOURCC('P','I','M','1');
    saida = VideoWriter("saida.mpg", fourcc, fps, Size(largura, altura));   
    
    /* Escreve o primeiro quadro. */
    saida << imagemFinal;
    
    contador = 0;
    while(1)
    {
        cap >> imagem; 
        
        /* Simula o efeito de stop-motion. */
        if(contador < 3)
        {
            saida << imagemFinal;
            contador++;
            continue;
        }
        
        /* Verifica se a imagem foi toda processada. */
        if (imagem.empty())
        {
            break;
        }
        
        /* Faz o borramento da imagem. */
        borrarImagem();
        
        /* Gera o quadro com o efeito de tiltshift. */
        calcularImagemFinal();
        aumentarSaturacao();
        saida << imagemFinal;
        quadroAtual = cap.get(CV_CAP_PROP_POS_FRAMES);
        percentagem = 100*quadroAtual/quantidadeTotalQuadros;
        
        /* Exibe o progresso atual do processamento. */
        cout << "[2] Processando vídeo... " << round(percentagem) << " % \r";
        cout.flush();
        
        contador = 0;
    }
    
    cap.release();
    saida.release();
    
    cout << endl << "[3] Processamento do vídeo concluído." << endl;

    return 0;
>>>>>>> 675d11c127f56246aed30cf4bba07a1a3a95cc71
}
