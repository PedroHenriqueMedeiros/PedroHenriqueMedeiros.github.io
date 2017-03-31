#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap;
    Mat imagemOriginal, imagemEscalaCinza, imagemEqualizada, resultado;
    Mat histImagemEscalaCinza, histImagemEqualizada;
    //Mat histImagemEscalaCinzaImg, histImagemEqualizadaImg;

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
    Mat histImagemEscalaCinzaImg(histh, histw, CV_8UC1, Scalar(0, 0, 0));
    Mat histImagemEqualizadaImg(histh, histw, CV_8UC1, Scalar(0, 0, 0));

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

        calcHist(&imagemEscalaCinza, 1, 0, Mat(), histImagemEscalaCinza, 1,
             histSize, ranges,
             uniform, acummulate);

        calcHist(&imagemEqualizada, 1, 0, Mat(), histImagemEqualizada, 1,
             histSize, ranges,
             uniform, acummulate);


        normalize(histImagemEscalaCinza, histImagemEscalaCinza, 0, histImagemEscalaCinzaImg.rows, NORM_MINMAX, -1, Mat());
        normalize(histImagemEqualizada, histImagemEqualizada, 0, histImagemEqualizadaImg.rows, NORM_MINMAX, -1, Mat());

        histImagemEscalaCinzaImg.setTo(Scalar(255));
        histImagemEqualizadaImg.setTo(Scalar(255));

        for(int i=0; i<hbins; i++)
        {
          line(histImagemEscalaCinzaImg,
               Point(i, histh),
               Point(i, histh-cvRound(histImagemEscalaCinza.at<float>(i))),
               Scalar(0), 1, 8, 0);
          line(histImagemEqualizadaImg,
               Point(i, histh),
               Point(i, histh-cvRound(histImagemEqualizada.at<float>(i))),
               Scalar(0), 1, 8, 0);
        }

        histImagemEscalaCinzaImg.copyTo(imagemEscalaCinza(Rect(10, 40, hbins, histh)));
        histImagemEqualizadaImg.copyTo(imagemEqualizada(Rect(10, 40, hbins, histh)));

        /* Põe as duas imagens lado a lado e exibe o resultado final */
        putText(imagemEscalaCinza, "Imagem Original", Point(10, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 2.0);
        putText(imagemEqualizada, "Imagem Equalizada", Point(10, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 2.0);
        hconcat(imagemEscalaCinza, imagemEqualizada, resultado);
        imshow("resultado",resultado);

        /* Fehca o programa ao apertar ESC */
        if(waitKey(30) >= 0) break;
    }
    imwrite("ImagemOriginalEscalaCinza.png",imagemEscalaCinza);
    imwrite("ImagemEqualizada.png",imagemEqualizada);
    return 0;
}
