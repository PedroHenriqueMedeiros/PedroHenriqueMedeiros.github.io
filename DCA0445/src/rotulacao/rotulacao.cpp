#include <iostream>
#include <opencv2/opencv.hpp>

#define COR_FUNDO_PADRAO 0
#define COR_BOLHA 255
#define COR_FUNDO_ALTERADA 192
#define COR_BURACO 96

using namespace cv;
using namespace std;

/* O algoritmo padrão de labeling fornecido pelo professor contem um problema, 
* pois utiliza a própria cor do pixel rotulado para armazenar a contagem de elementos. 
* No caso de uma imagem em escala de cinza, em que cada pixel é representado por
* 1 byte, logo poder-se-ia contar apenas 255 elementos. 
* Uma abordagem diferente seria utilizar a cor do pixel como um tipo de classificação
* para os objetos contados, de modo que os pixels de uma mesma categoria assumam apenas
* um label específico, sendo reservado um contador à parte para contar quantos elementos
* dessa categoria foram encontrados.
*/

int main(int argc, char **argv)
{
    Mat imagem;
    Point p;
    uint bolhasEncontradas = 0;

    imagem = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (!imagem.data)
    {
        cout << "Imagem nao carregou corretamente.\n";
        return -1;
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
    for (int i = 0; i < imagem.rows; i++)
    {
        for (int j = 0; j < imagem.cols; j += imagem.cols - 1)
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

     /* Finalmente, conta os elementos apenas que contém buraco na cor de fundo
      * original. 
      */
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

    /* Informa ao usuário a contagem final */
    cout << "Há " << bolhasEncontradas << " bolhas nesa imagem." << endl;
    imshow("imagem", imagem);
    imwrite("saida.png", imagem);
    waitKey(5000);
    return 0;
}
