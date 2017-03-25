#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

int main(int, char**) {
   
    Mat image;
    namedWindow("janela",WINDOW_AUTOSIZE);

    image= imread("lena.png",CV_LOAD_IMAGE_COLOR);
   
    int width = image.size().width;
    int height = image.size().height;

    
    Mat A(image, Rect(0, 0, width/2, width/2));
    Mat B(image, Rect(width/2, 0, width/2, height/2));
    Mat C(image, Rect(0, height/2, width/2, height/2));
    Mat D(image, Rect(width/2,height/2,width/2,height/2));

    //Cria uma matrix de zeros de mesmo tamanho da original
    Mat saida = Mat::ones(image.size(), image.type());
    Mat auxiliar;

    //Utiliza uma variavel auxiliar para mapear a nova região da matriz de saída 
    auxiliar = saida.colRange(0, width/2).rowRange(0, height/2);
    //Copia o conteúdo em D na região mapeada pela variável auxiliar
    C.clone().copyTo(auxiliar);

    auxiliar = saida.colRange(width/2, width).rowRange(0, height/2);
    A.copyTo(auxiliar);

    auxiliar = saida.colRange(0, width/2).rowRange(height/2, height);
    D.copyTo(auxiliar);

    auxiliar = saida.colRange(width/2, width).rowRange(height/2, height);
    B.copyTo(auxiliar);

    if(!image.data)
        cout << "nao abriu a imagem" << endl;

    namedWindow("janela", WINDOW_AUTOSIZE);
    imshow("janela", saida);
    waitKey();
    return 0;

}
