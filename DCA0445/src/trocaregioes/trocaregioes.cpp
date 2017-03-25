#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/* 
 * argv[1] = imagem 
 *
 *  Esquema dos quadrantes.
 * 
 *    q1  |   q2
 *   ____________
 *    q3  |   q4
 *
 */

int main(int argc, char** argv)
{
    Mat imagem, resultado, q1, q2, q3, q4, temp1, temp2;   
    
    /* Verifica o número de argumentos.  */
    if (argc != 2) 
    {
        cout << "A lista de argumentos deve ser: "
                                            "./trocaregioes <imagem>" << endl;
        return -1;
    }
    
    /* Checa se a imagem pode ser aberta. */
    imagem = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (!imagem.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }
    
    /* Cria sub-matrizes representando cada quadrante da figura. */
    q1 = Mat(imagem, Rect(0, 0, imagem.cols/2, imagem.rows/2));
    q2 = Mat(imagem, Rect(imagem.rows/2, 0, imagem.cols/2, imagem.rows/2));
    q3 = Mat(imagem, Rect(0, imagem.cols/2, imagem.cols/2, imagem.rows/2));
    q4 = Mat(imagem, Rect(imagem.rows/2, imagem.cols/2, 
                imagem.cols/2, imagem.rows/2));
    
    /* Une os quadrantes q4 e q3 horizontalmente para construir a metade 
     * superior da imagem. Este resultado parcial é armazenado em temp1.
     */
    hconcat(q4, q3, temp1);
    
    /* Une os quadrantes q2 e q1 horizontalmente para construir a metade 
     * inferior da imagem. Este resultado parcial é armazenado em temp2.
     */
    hconcat(q2, q1, temp2);
    
    /* Finalmente, une verticialmente os dois resultados parciais anteriores
     * para montar o resultado final.
     */
    vconcat(temp1, temp2, resultado);
    
    imwrite("saida.png", resultado);
    imshow("resultado", resultado);  
    waitKey(10000);

    return 0;
}
