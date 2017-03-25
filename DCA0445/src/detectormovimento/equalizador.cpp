#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap;
    Mat imagemOriginal, imageEscalaCinza, imagemEqualizada;

    /* Níveis de matiz */
    int hbins = 30; 
    /* Níveis de saturação */
    int sbins = 32;
    /* Tamanho do histograma */
    int histSize[] = {hbins, sbins};

    /* Matiz varia de 0 a 179 */
    float hranges[] = { 0, 180 };
    /* Saturação varia de 0 a 255 */
    float sranges[] = {0,256};
    /* Faixas de valores do histograma */
    const float* ranges[] = { hranges, sranges };


    Mat histogramaOriginal, histogramaEqualizado;

    int largura, altura;
    int nbins = 64;
    float range[] = {0, 256};
    const float *histrange = { range };
    bool uniform = true;
    bool acummulate = false;

    /* Definem-se as imagens que irão armazenar os histogramas da imagem original */
    //int histw = nbins, histh = nbins/2;
    //Mat histImgOriginal(histh, histw, CV_8UC1, Scalar(0,0,0));
    //Mat histImgEqualizado(histh, histw, CV_8UC1, Scalar(0,0,0));

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
        cvtColor(imagemOriginal, imageEscalaCinza, CV_BGR2GRAY);
        /* Faz a equalização do histograma */
        equalizeHist(imageEscalaCinza, imagemEqualizada);

        imshow("imagemOriginal", imageEscalaCinza);
        imshow("imagemEqualizada",imagemEqualizada);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}
