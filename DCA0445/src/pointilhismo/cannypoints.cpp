#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <ctime>

using namespace std;
using namespace cv;

/* Slider usado para configurar o limiar do algoritmo de Canny. */
int threshold_slider = 10;
int threshold_slider_max = 200;

/* Sliders usado para configurar o efeito de pointilhismo. */
int step_slider = 1;
int step_slider_max = 10;

int jitter_slider = 3;
int jitter_slider_max = 10;

int raio_slider = 2;
int raio_slider_max = 10;

int raio_borda = 4;

/* Altera o raio do pointilhismo na borda da imagem de acordo com o limiar
 * do algoritmo de Canny. */
void mudar_raio_borda()
{
    if (threshold_slider < 50)
    {
        raio_borda = 4;
    }
    else if (threshold_slider >= 50 && threshold_slider < 100)
    {
        raio_borda = 3;
    }
    else if (threshold_slider >= 100 && threshold_slider < 150)
    {
        raio_borda = 2;
    }
    else
    {
        raio_borda = 1;
    }
}


Mat imageGray, imageColor, border, points, resultado;

void alterarSliderCanny(int, void*)
{
    Canny(imageGray, border, threshold_slider, 3 * threshold_slider);
    resultado = points.clone();
    uchar r, g, b;
    Vec3b colors;
    
    mudar_raio_borda();

    /* Recria o efeito o pointilhismo, modificando as bordas. */
    for (int i = 0; i < border.rows; i++)
    {
        for (int j = 0; j < border.cols; j++)
        {
            if (border.at<uchar>(i, j) == 255)
            {
                colors = imageColor.at<Vec3b>(i, j);
                b = colors[0];
                g = colors[1];
                r = colors[2];

                circle(resultado, cv::Point(j, i), raio_borda, CV_RGB(r, g, b), 
                    -1, CV_AA);
            }
        }
    }

    imshow("canny", border);
    imshow("resultado", resultado);
}

void alterarSliderPointilhismo(int, void*)
{
    vector<int> yrange;
    vector<int> xrange;
    int width, height;
    int x, y;
    uchar r, g, b;
    Vec3b colors;

    if (step_slider < 1)
    {
        step_slider = 1;
    }

    srand(time(0));

    width = imageColor.cols;
    height = imageColor.rows;

    xrange.resize(height / step_slider);
    yrange.resize(width / step_slider);

    /* Preenche ambos os vetores com 0's. */
    iota(xrange.begin(), xrange.end(), 0);
    iota(yrange.begin(), yrange.end(), 0);

    for (uint i = 0; i < xrange.size(); i++)
    {
        xrange[i] = xrange[i] * step_slider + step_slider / 2;
    }

    for (uint i = 0; i < yrange.size(); i++)
    {
        yrange[i] = yrange[i] * step_slider + step_slider / 2;
    }

    points = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));

    random_shuffle(xrange.begin(), xrange.end());

    for (auto i : xrange)
    {
        random_shuffle(yrange.begin(), yrange.end());
        for (auto j : yrange)
        {
            x = i + rand() % (2 * jitter_slider) - jitter_slider + 1;
            y = j + rand() % (2 * jitter_slider) - jitter_slider + 1;

            colors = imageColor.at<Vec3b>(i, j);
            b = colors[0];
            g = colors[1];
            r = colors[2];

            circle(points, cv::Point(y, x), raio_slider, CV_RGB(r, g, b), 
            -1, CV_AA);
        }
    }

    imshow("pointilhismo", points);
    alterarSliderCanny(0, 0);
}

int main(int argc, char** argv)
{

    /* Verifica o número de argumentos.  */
    if (argc != 2)
    {
        cout << "A lista de argumentos deve ser: "
            "./cannypoints <imagem>"<< endl;
        return -1;
    }

    /* Checa se a imagem pode ser aberta. */
    imageColor = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if (!imageColor.data)
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }
    
    cvtColor(imageColor, imageGray, CV_BGR2GRAY);

    points = imageColor.clone();
    border = imageGray.clone();

    namedWindow("canny", 1);
    namedWindow("pointilhismo", 1);
    namedWindow("resultado", 1);

    Canny(imageGray, border, threshold_slider, 3 * threshold_slider);

    createTrackbar(
    "Treshold inferior", "canny", &threshold_slider, threshold_slider_max, alterarSliderCanny);

    createTrackbar(
    "Step", "pointilhismo", &step_slider, step_slider_max, alterarSliderPointilhismo);

    createTrackbar(
    "Jitter", "pointilhismo", &jitter_slider, jitter_slider_max, alterarSliderPointilhismo);

    createTrackbar(
    "Raio", "pointilhismo", &raio_slider, raio_slider_max, alterarSliderPointilhismo);

    /* aplica o efeito do pointilhismo. */

    alterarSliderPointilhismo(0, 0);
    alterarSliderCanny(threshold_slider, 0);

    waitKey();
    return 0;
}
