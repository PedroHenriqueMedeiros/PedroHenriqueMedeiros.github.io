#include <iostream>
#include <opencv2/opencv.hpp>

#define COR_FUNDO_PADRAO 0
#define COR_BOLHA 255
#define COR_FUNDO_ALTERADA 192
#define COR_BURACO 96

using namespace cv;
using namespace std;

/* 
 * argv[1] = imagem 
 */

int main(int argc, char **argv)
{
    Mat imagem;
    Point p;
    uint bolhasEncontradas = 0;

    /* Verifica a quantidade de argumentos. */
    if (argc != 2) 
    {
        cout << "A lista de argumentos deve ser: ./rotulacao <imagem>" << endl;
        return -1;
    }

    /* Verifica se a imagem pode ser aberta. */
    imagem = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (!imagem.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }

    /* Primeiramente remove as bolhas das bordas */
    for (int i = 0; i <= imagem.rows; i += imagem.rows - 1)
    {
        for (int j = 0; j < imagem.cols; j++)
        {
            if (imagem.at<uchar>(i, j) == COR_BOLHA)
            {
                p.x = j;
                p.y = i;
                floodFill(imagem, p, COR_FUNDO_PADRAO);
            }
        }
    }
    for (int j = 0; j < imagem.cols; j += imagem.cols - 1)
    {
        for (int i = 0; i < imagem.rows; i++)
        {
            if (imagem.at<uchar>(i, j) == 255)
            {
                p.x = j;
                p.y = i;
                floodFill(imagem, p, COR_FUNDO_PADRAO);
            }
        }
    }

    /* Altera a cor de fundo padrão para uma nova cor */
    p.x = 0;
    p.y = 0;
    floodFill(imagem, p, COR_FUNDO_ALTERADA);

    /* Conta os elementos apenas que contém buraco na cor de fundo original */
    for (int i = 0; i < imagem.rows; i++)
    {
        for (int j = 0; j < imagem.cols; j++)
        {
            if (imagem.at<uchar>(i, j) == COR_FUNDO_PADRAO)
            {
                bolhasEncontradas++;
                p.x = j;
                p.y = i;
                floodFill(imagem, p, COR_BURACO);
            }
        }
    }

    /* Informa ao usuário a contagem final  */
    cout << "Há " << bolhasEncontradas << " bolhas nesa imagem." << endl;
    imshow("imagem", imagem);
    imwrite("saida.png", imagem);
    waitKey(5000);
    return 0;
}
