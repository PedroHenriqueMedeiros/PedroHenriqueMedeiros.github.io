#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/* Definição de todas as máscaras dos filtros */
float media[] = {
    1,  1,  1,
    1,  1,  1,
    1,  1,  1
};
float gauss[] = {
    1,  2,  1,
    2,  4,  2,
    1,  2,  1
};
float horizontal[]={
    -1,  0,  1,
    -2,  0,  2,
    -1,  0,  1
};
float vertical[]={
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};
float laplacian[]={
     0, -1,  0,
    -1,  4, -1,
     0, -1,  0
};

/* Resultado da convolução digital dos filtros gauss e laplacian acima.
 * Ela pode ser feita à mão ou utilizando alguma ferramenta como Simulink
 *   ou Python + numpy+ scipy (exemplo em anexo). 
 */
float laplgauss[]={
    0, -1, -2, -1,  0,
   -1,  0,  2,  0, -1,
   -2,  2,  8,  2, -2,
   -1,  0,  2,  0, -1,
    0, -1, -2, -1,  0,
};

/* Exibe a máscara na saída padrão */
void printMask(Mat &m){
    for(int i = 0; i < m.size().height; i++)
    {
        for(int j = 0; j < m.size().width; j++)
        {
            cout << m.at<float>(i,j) << ",";
        }
        cout << endl;
    }
}

/* Exibe o menu principal */
void menu(){
    cout << "\nPressione a tecla para ativar o filtro: \n"
    "a - Calcular módulo\n"
    "m - Média\n"
    "g - Gauss\n"
    "v - Vertical\n"
    "h - Horizontal\n"
    "l - Laplaciano\n"
    "p - Laplaciano do Gaussiano\n"
    "ESC - Sair\n";
}

int main(int argvc, char** argv){
    VideoCapture video;
    Mat cap, frame, frame32f, frameFiltered;
    Mat mask(3,3,CV_32F), mask1;
    Mat result, result1;
    double width, height;
    int absolut;
    char key;

    /* Tenta abrir o vídeo. */
    video.open(0); 
    if(!video.isOpened()) 
    {
        return -1;
    }
    
    width = video.get(CV_CAP_PROP_FRAME_WIDTH);
    height = video.get(CV_CAP_PROP_FRAME_HEIGHT);
    cout << "largura=" << width << "\n";;
    cout << "altura =" << height<< "\n";;

    /* Cria as duas janelas */
    namedWindow("original",1);
    namedWindow("filtroespacial",1);

    mask = Mat(3, 3, CV_32F, media); 
    scaleAdd(mask, 1/9.0, Mat::zeros(3,3,CV_32F), mask1);
    swap(mask, mask1);
    absolut=1; // calcs abs of the image

    menu();

    for(;;)
    {
        /* Faz a captura do vídeo e converte para escala de cinza */
        video >> cap; 
        cvtColor(cap, frame, CV_BGR2GRAY);

        /* Gira a imagem horizontalmente */
        flip(frame, frame, 1);
        imshow("original", frame);
        frame.convertTo(frame32f, CV_32F);

        /* Aplica o filtro selecionado (filtro inicial: média) */
        filter2D(frame32f, frameFiltered, frame32f.depth(), mask, Point(1,1), 0);

        if(absolut)
        {
            frameFiltered=abs(frameFiltered);
        }
        frameFiltered.convertTo(result, CV_8U);
        imshow("filtroespacial", result);
        key = (char) waitKey(10);
        if( key == 27 ) break; // esc pressed!
	
        switch(key)
        {
            case 'a':
                menu();
                absolut=!absolut;
                break;
            case 'm':
                menu();
                mask = Mat(3, 3, CV_32F, media);
                scaleAdd(mask, 1/9.0, Mat::zeros(3,3,CV_32F), mask1);
                mask = mask1;
                printMask(mask);
                break;
            case 'g':
                menu();
                mask = Mat(3, 3, CV_32F, gauss);
                scaleAdd(mask, 1/16.0, Mat::zeros(3,3,CV_32F), mask1);
                mask = mask1;
                printMask(mask);
                break;
            case 'h':
                menu();
                mask = Mat(3, 3, CV_32F, horizontal);
                printMask(mask);
                break;
            case 'v':
                menu();
                mask = Mat(3, 3, CV_32F, vertical);
                printMask(mask);
                break;
            case 'l':
                menu();
                mask = Mat(3, 3, CV_32F, laplacian);
                printMask(mask);
                break;
            case 'p':
                menu();
                /* Aplica o filtro 5 x 5, que representa o gaussinado + laplaciano */
                mask = Mat(5, 5, CV_32F, laplgauss);
                printMask(mask);
                break;
            default:
                break;
        }
    }
	
    // imwrite("Imagem_Filtro_Mediana.png",frameFiltered);
    // imwrite("Imagem_Filtro_Vertical.png",frameFiltered);
    //imwrite("Imagem_Filtro_Horizontal.png",frameFiltered);
    //imwrite("Imagem_Filtro_Gauss.png",frameFiltered);
    //imwrite("Imagem_Filtro_Laplaciano.png",frameFiltered);
    imwrite("Imagem_Filtro_LaplacianoGaussiano.png",frameFiltered);
    

    return 0;
}

