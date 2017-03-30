#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

VideoCapture cap;
VideoWriter saida;

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
    int largura, altura, fourcc;
    double fps, quantidadeTotalQuadros, quadroAtual, percentagem;

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
    
    string nomeArquivoCompleto(argv[1]);
    size_t indicePonto = nomeArquivoCompleto.find_last_of("."); 
    string nomeArquivo = nomeArquivoCompleto.substr(0, indicePonto); 
    
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
    saida = VideoWriter(nomeArquivo + ".mpg", fourcc, fps, Size(largura, altura));   
    
    /* Escreve o primeiro quadro. */
    saida << imagemFinal;
        
    while(1)
    {
        cap >> imagem; 
        
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
        
        cout << "[2] Processando vídeo... " << round(percentagem) << " % \r";
        cout.flush();
    }
    
    cap.release();
    saida.release();
    
    cout << endl << "[3] Processamento do vídeo concluído." << endl;

    return 0;
}
