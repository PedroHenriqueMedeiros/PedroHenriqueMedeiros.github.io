#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap;
    Mat imagemOriginal, imagemEscalaCinza, imagemEqualizada, resultado;

    /* Tenta abrir a câmera */
    cap.open(0);
    if(!cap.isOpened()){
        cout << "A captura não pôde ser aberta." << endl;
        return -2;
    }

    while(1)
    {
        /* Obtem um frame da imagem (colorida) */
        cap >> imagemOriginal;
        /* Converte o frame para escala de cinza */
        cvtColor(imagemOriginal, imagemEscalaCinza, CV_BGR2GRAY);
        /* Faz a equalização do histograma da imagem original. */
        equalizeHist(imagemEscalaCinza, imagemEqualizada);

        /* Põe as duas imagens lado a lado e exibe o resultado final */
        putText(imagemEscalaCinza, "Imagem Original", Point(5, 50), FONT_HERSHEY_TRIPLEX, 1.0, CV_RGB(255,255,0), 2.0);
        putText(imagemEqualizada, "Imagem Equalizada", Point(5, 50), FONT_HERSHEY_TRIPLEX, 1.0, CV_RGB(255,255,0), 2.0);
        hconcat(imagemEscalaCinza, imagemEqualizada, resultado);
        imshow("resultado",resultado);

        /* Fehca o programa ao apertar ESC */
        if(waitKey(30) >= 0) break;
    }
    return 0;
}
