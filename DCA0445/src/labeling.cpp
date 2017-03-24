#include <iostream>
#include <opencv2/opencv.hpp>

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

#define DEFAULT_BACKGROUND 0
#define OBJECT 255
#define NEW_BACKGROUND 192
#define HOLE 96

int main(int argc, char **argv)
{
    Mat image;
    Point p;
    uint bubble_count = 0;

    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
    {
        cout << "imagem nao carregou corretamente\n";
        return -1;
    }

    // Remove bubles from the borders.
    for (int i = 0; i <= image.rows; i += image.rows - 1)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uchar>(i, j) == OBJECT)
            {
                p.x = j;
                p.y = i;
                floodFill(image, p, DEFAULT_BACKGROUND);
            }
        }
    }
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j += image.cols - 1)
        {
            if (image.at<uchar>(i, j) == 255)
            {
                p.x = j;
                p.y = i;
                floodFill(image, p, DEFAULT_BACKGROUND);
            }
        }
    }

    // Change the background color.
    p.x = 0;
    p.y = 0;
    floodFill(image, p, NEW_BACKGROUND);

     // Finally, count only the elements with a black hole (bubbles).
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uchar>(i, j) == DEFAULT_BACKGROUND)
            {
                bubble_count++;
                p.x = j;
                p.y = i;
                floodFill(image, p, HOLE);
            }
        }
    }

    // Show the final result.
    cout << "There are " << bubble_count << " bubbles in this image." << endl;
    imshow("image", image);
    imwrite("labeling.png", image);
    waitKey(5000);
    return 0;
}