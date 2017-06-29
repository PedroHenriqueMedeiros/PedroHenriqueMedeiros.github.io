#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/* 
 * argv[1] = imagem 
 * argv[2] = p1.x
 * argv[3] = p1.y
 * argv[4] = p2.x
 * argv[5] = p2.y
 */

int main(int argc, char** argv)
{
    Mat imagem;  
    Vec3b atual, novo;
    Point p1, p2;
    Rect ret;
    
    /* Verifica a quantidade de argumentos.  */
    if (argc != 6) 
    {
        cout << "A lista de argumentos deve ser: ./regioes <imagem> <ponto1_x> "
                                    "<ponto1_y> <ponto2_x> <ponto2_y>" << endl;
        return -1;
    }
    
    /* Verifica se a imagem pode ser aberta. */
    imagem = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (!imagem.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }
    
    /* Verifica se todas as coordenadas fornecidas são números. */
    try 
    {
        p1.x = stoi(argv[2]);
        p1.y = stoi(argv[3]);
        p2.x = stoi(argv[4]);
        p2.y = stoi(argv[5]);       
    }
    catch (const std::exception& e) 
    {
        cout << "Uma ou mais coordenadas não é válida." << endl;
        return -3;
    }
    
    /* Vericfica os pontos estão dentro da imagem. */
    if (p1.x > imagem.rows || p1.y > imagem.cols)
    {
        cout << "Ponto 1 está fora da imagem.";
        return -4;
    }
    if (p2.x > imagem.rows || p2.y > imagem.cols)
    {
        cout << "Ponto 2 está fora da imagem..";
        return -4;
    }
    
    ret = Rect(p1, p2);
    
    /* Calcula o negativo da região retangular formada pelos dois pontos. */
    for (int i = ret.x; i < ret.height; i++)
    {
        for (int j = ret.y; j < ret.width; j++)
        {
            imagem.at<uchar>(i, j) = 255 - imagem.at<uchar>(i, j) ;
        }
    }

    imwrite("saida.png", imagem);
    imshow("resultado", imagem);  
    waitKey(10000);
    
    return 0;
}
