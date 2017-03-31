#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

/* Define o tempo de captura de cada imagem base. */
#define TEMPO_NOVA_CAPTURA_MS 50

/* Controla a sensibilidade da detecção de movimento.  */
#define LIMIAR  0.99 

using namespace cv;
using namespace std;
using namespace chrono;

int main(int argc, char** argv){
    /* Imagem atual. */
    Mat imagem;
    /* Matriz para a imagem base (capturada mementos antes da atual). */
    Mat imagemBase;

    VideoCapture cap;

    /* Variável acumulador que é responsável por verificar a passagem do tempo 
     * utilzando de funções de para medir o número de ciclos e as frequências 
     * dos mesmos(do sistema operacional). */
    double resultadoCorrelacaoR, resultadoCorrelacaoG, resultadoCorrelacaoB;
    
    steady_clock::time_point inicio, fim;
    int diferencaTempo;

    /*Vetor para separar os planos RGB para os frames base e atual*/
    vector<Mat> planos, planosBase;
    
    /*Cálculos dos histogramas das componentes RGB de cada frame.*/ 
    Mat histR, histG, histB, histRBase, histGBase, histBBase;
        
    /* Níveis de matiz */
    int hbins = 128; 
    /* Níveis de saturação */
    int sbins = 64;
    /* Tamanho do histograma */
    int histSize[] = {hbins, sbins};

    /* Matiz varia de 0 a 179 */
    float hranges[] = {0, 180};
    /* Saturação varia de 0 a 255 */
    float sranges[] = {0, 256};
    /* Faixas de valores do histograma */
    const float* ranges[] = { hranges, sranges };
    
    bool uniform = true;
    bool acummulate = false;
    
    int histw = hbins, histh = sbins/2;

    cap.open(0);
    /* Verifica se a câmera do computador está disponível. */
    if(!cap.isOpened()){
        cout << "cameras indisponiveis";
        return -1;
    }

    /* Define-se a largura e altura das imagens que serão usadas para desenhar 
     * os histogramas de cada uma das componentes de cor. As imagens são criadas 
     * com o tipo CV_8UC3, ou seja, com 8 bits por pixel, com tipo de dados 
     * unsigned char contendo 3 canais de cor. A cor, nesse caso, servirá 
     * apenas para que o histograma seja desenhado na cor respectiva de 
     * sua componente. */
    Mat histImgR(histh, histw, CV_8UC3, Scalar(0,0,0));
    Mat histImgG(histh, histw, CV_8UC3, Scalar(0,0,0));
    Mat histImgB(histh, histw, CV_8UC3, Scalar(0,0,0));

    Mat histImgR2(histh, histw, CV_8UC3, Scalar(0,0,0));
    Mat histImgG2(histh, histw, CV_8UC3, Scalar(0,0,0));
    Mat histImgB2(histh, histw, CV_8UC3, Scalar(0,0,0));
        
    /* Captura a imagem base para servir como comparação. */
    cap >> imagemBase;
    
    inicio = chrono::steady_clock::now();

    while(1){
    
        cap >> imagem;
        
        fim = chrono::steady_clock::now();
        diferencaTempo = duration_cast<milliseconds>(fim - inicio).count();
        
        if(diferencaTempo > TEMPO_NOVA_CAPTURA_MS)
        {
            cap >> imagemBase;
            inicio = chrono::steady_clock::now();
        }
        
        /* Separa os planos R, G e B da imagem que serve como base. */
        split (imagemBase, planosBase);
        
        /* Calcula o histograma para os 3 canais de cores da imagem base. */
        calcHist(&planosBase[0], 1, 0, Mat(), histRBase, 1, histSize, ranges, 
                uniform, acummulate);
        calcHist(&planosBase[1], 1, 0, Mat(), histGBase, 1, histSize, ranges,
                uniform, acummulate);
        calcHist(&planosBase[2], 1, 0, Mat(), histBBase, 1, histSize, ranges,
                uniform, acummulate);

         /* Separa os planos R, G e B da imagem atual. */
        split (imagem, planos);
        
        /* Calcula o histograma para os 3 canais de cores da imagem atual. */
        calcHist(&planos[0], 1, 0, Mat(), histR, 1, histSize, ranges,
                    uniform, acummulate);
        calcHist(&planos[1], 1, 0, Mat(), histG, 1, histSize, ranges,
                uniform, acummulate);
        calcHist(&planos[2], 1, 0, Mat(), histB, 1, histSize, ranges,
                uniform, acummulate);

        /* Faz a normalização dos histogramas de cada canal de cor. */
        normalize(histR, histR, 0,1, NORM_MINMAX, -1, Mat());
        normalize(histG, histG, 0, 1, NORM_MINMAX, -1, Mat());
        normalize(histB, histB, 0, 1, NORM_MINMAX, -1, Mat());

        normalize(histRBase, histRBase, 0, 1, NORM_MINMAX, -1, Mat());
        normalize(histGBase, histGBase, 0, 1, NORM_MINMAX, -1, Mat());
        normalize(histBBase, histBBase, 0, 1, NORM_MINMAX, -1, Mat());

        /* Utiliza o método da correlação para comparar os histogramas de todos
         * os canais de cores. */
        resultadoCorrelacaoR = compareHist(histRBase, histR, CV_COMP_CORREL);
        resultadoCorrelacaoG = compareHist(histGBase, histG, CV_COMP_CORREL);
        resultadoCorrelacaoB = compareHist(histBBase, histB, CV_COMP_CORREL);
        
        /*
        cout <<  "Resultado da método da comparação R: "<< resultadoCorrelacaoR 
             << endl;
        cout <<  "Resultado da método da comparação G: "<< resultadoCorrelacaoG 
             << endl;
        cout <<  "Resultado da método da comparação B: "<< resultadoCorrelacaoB 
             << endl;
        cout << "--------------------------------------------------" << endl;
        */
        
        /* Verifica se pelo menos um dos resultados está abaixo do limiar. Caso
         * esteja, é porque houve variação no histograma e, por consguinte,
         * houve algum movimento.
         */
        if (resultadoCorrelacaoR < LIMIAR || resultadoCorrelacaoG < LIMIAR || 
            resultadoCorrelacaoB < LIMIAR)
        {
            putText(imagem, "Movimento Detectado", Point(10, 40), 
                            FONT_HERSHEY_TRIPLEX, 1.0, CV_RGB(255,255,0), 2.0);
        }
        
        /* Verifica se a variável acumulador chegou ao valor preestabelicido 
         * pelo programador. */

        //imshow("base", imagemBase);
        imshow("imagem", imagem);

        if(waitKey(30) >= 0)
        {
            break;
        }
    }
    return 0;
}
