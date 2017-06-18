#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace cv::ocl;

#define PONTOS_CIRCULO 50
#define LIMIAR_RELACAO_AREAS 0.70
#define LIMIAR_RAIO 10
#define MAX_FECHAMENTO 32 // Cresce a partir de 1 em potências de 2.
#define RESULTADO_SEPARADO true

/* Estrutura que define um círculo, que contornará a moeda. */
struct Circulo {
    Point2f centro;
    float raio;
};

/* Estrutura que representa a moeda e outras informações associadas, tais como 
 * seu contorno e o menor retângulo que a cerca. */
struct Moeda {

    Mat imagem;
    int valor = 0;
    double raio;
    double momentos[7];
    Point2f centro;
    Rect retangulo;
    vector<vector<Point> > contornos;
    vector<Vec4i> hierarquia;
    
    /* Calcula os momentos invariantes quando a moeda é criada. */
    Moeda(Mat _imagem) 
    {
        Mat moedaCinza;
        Mat canny;
        imagem = _imagem.clone();
        
        /* Calcula os momentos invariantes. */
        cvtColor(imagem, moedaCinza, CV_RGB2GRAY);
        Canny(imagem, canny, 20, 60, 3);
        
        SIFT s;
        
        Moments momento = moments(moedaCinza, false);
        HuMoments(momento, momentos);
    }
    
    void imprimirMomentos()
    {
        stringstream saida;
        saida << valor << " = {";
        for(int i = 0; i < 7; i++)
        {    
            saida << scientific << setprecision(7) << momentos[i];
            if (i < 6)
            {
                saida << ", ";
            }
        }
        saida << "};";
        cout << saida.str() << endl;
    }
};

/* Detecta a quantidade máxima de moedas utilizando a função detectarMoedas. */
vector<Moeda> detectarTodasMoedas(Mat imagem);

/* Função que detecta moedas em uma imagem, utilizando um determinado nível na
 * operação morfológica do fechamento. */
vector<Moeda> detectarMoedas(Mat imagem, int fechamento);

/* Função que desenha na imagem ou individualmente as moedas localizadas. */
void exibirMoedas(const Mat &imagem, const vector<Moeda> &moedas);

/* Função que remova moedas da borda da imagem. */
void removeMoedasBorda(Mat &imagem);

/* Função que remove buracos dentro da moeda durante a sua segmentação. */
void removeBuracos(Mat &imagem);



/**
 * Extract local features for an image
 */
 
Mat getDescriptors(const Mat& imagem)
{
	Mat src = imagem;
	Mat ycrcb, resultado;
	cvtColor(src, ycrcb, CV_BGR2YCrCb);
	vector<Mat> channels;
	split(ycrcb, channels);
	equalizeHist(channels[0], channels[0]);
	merge(channels, resultado);
	cvtColor(resultado, src, CV_YCrCb2BGR);
	resize(src, src, Size(100, 100), 0, 0, INTER_LINEAR);
	
	return src;
	
	/*

	Mat dst(src.size(), src.type()); // yes, you need to preallocate here

	IplImage ipsrc = src; // new header, points to the same pixels
	IplImage ipdst = dst;

	// CV_WARP_INVERSE_MAP
	cvLogPolar( &ipsrc, &ipdst, cvPoint2D32f(src.cols/2, src.rows/2), 50, CV_INTER_CUBIC);	
	
	return dst;
	* 
	* */
}

void mlp(cv::Mat& trainingData, cv::Mat& trainingClasses) {


	// Cria a MLP.

    Mat layers = cv::Mat(3, 1, CV_32SC1);
    layers.row(0) = Scalar(10000);
    layers.row(1) = Scalar(10);
    layers.row(2) = Scalar(1);

    CvANN_MLP mlp;
    CvANN_MLP_TrainParams params;
    CvTermCriteria criteria;
    criteria.max_iter = 1000;
    criteria.epsilon = 0.00001f;
    criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
    params.train_method = CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale = 0.05f;
    params.bp_moment_scale = 0.05f;
    params.term_crit = criteria;

    mlp.create(layers);

    // Realiza o treinamento.
    mlp.train(trainingData, trainingClasses, cv::Mat(), cv::Mat(), params);
    Mat response(1, 1, CV_32FC1);
    
    
    FileStorage fs("./treinamento/mlp.yml", FileStorage::WRITE); // or xml
	mlp.write(*fs, "mlp"); 
    
    //mlp.predict(sample, response);

}



vector<float> mat2vec(Mat &mat)
{
	std::vector<float> array;
	if (mat.isContinuous()) {
	  array.assign((float*)mat.datastart, (float*)mat.dataend);
	} else {
	  for (int i = 0; i < mat.rows; ++i) {
		array.insert(array.end(), (float*)mat.ptr<uchar>(i), (float*)mat.ptr<uchar>(i)+mat.cols);
	  }
	}
	
	return array;
}

/* Função main. */
int main(int argc, char** argv) 
{
    Mat imagemColorida;
    vector<Moeda> moedas;
    
    /* Verifica o número de argumentos.  */
    if (argc != 2)
    {
        cout << "A lista de argumentos deve ser: "
            "./preparador <imagem>"<< endl;
        return -1;
    }

    /* Checa se a imagem pode ser aberta. */
    imagemColorida = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if (!imagemColorida.data)
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }

    cout << "[main] Realizando deteção pelo algoritmo padrão." << endl;
    moedas = detectarTodasMoedas(imagemColorida);
    
	// Definindo conjunto treinamento.
    Mat trainingData(8, 10000, CV_32FC1);
    Mat trainingClasses(8, 1, CV_32FC1);
    
    Mat m10f = imread("./treinamento/10f.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v10f = mat2vec(m10f);
    Mat m10n = imread("./treinamento/10n.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v10n = mat2vec(m10n);
    Mat m25f = imread("./treinamento/25f.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v25f = mat2vec(m25f);
    Mat m25n = imread("./treinamento/25n.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v25n = mat2vec(m25n);
    Mat m50f = imread("./treinamento/50f.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v50f = mat2vec(m50f);
    Mat m50n = imread("./treinamento/50n.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v50n = mat2vec(m50n);
    Mat m100f = imread("./treinamento/100f.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v100f = mat2vec(m100f);
    Mat m100n = imread("./treinamento/100n.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v100n = mat2vec(m100n);
    
    for(int i = 0; i < 10000; i++)
    {
		trainingData.at<float>(0, i) = v10f[i];
		trainingData.at<float>(1, i) = v10n[i];
		trainingData.at<float>(2, i) = v25f[i];
		trainingData.at<float>(3, i) = v25n[i];
		trainingData.at<float>(4, i) = v50f[i];
		trainingData.at<float>(5, i) = v50n[i];
		trainingData.at<float>(6, i) = v100f[i];
		trainingData.at<float>(7, i) = v100n[i];
		
	}
    
    trainingClasses.at<float>(0,0) = 9;
    trainingClasses.at<float>(1,0) = 11;
    trainingClasses.at<float>(2,0) = 99;
    trainingClasses.at<float>(3,0) = 101;
    trainingClasses.at<float>(4,0) = 999;
    trainingClasses.at<float>(5,0) = 1001;
    trainingClasses.at<float>(6,0) = 9999;
    trainingClasses.at<float>(7,0) = 10001; 
    
    
    mlp(trainingData, trainingClasses);


    cout << "[main] Exibindo " << moedas.size() << " moedas encontradas. " << endl;
    exibirMoedas(imagemColorida, moedas);   

    cout << "[main] Associando valor às moedas." << endl;
    /* Associando um valor (financeiro) a cada moeda. */
    for( int i = 0; i < (int) moedas.size(); i++)
    {
        /* Salvando resultado em arquivo. */
        cout << "[main] Digite o valor da moeda " << i << ": ";
        cin >> moedas[i].valor; 
    }
    
    /* Exibe o resultado final. */
    cout << "[main] Resultado final (valor da moeda seguido dos momentos invariantes): " << endl << endl;
    for( int i = 0; i < (int) moedas.size(); i++)
    {
        moedas[i].imprimirMomentos();
    }

    return(0);
}


void removeMoedasBorda(Mat &imagem)
{
    Point p;
    
    for (int i = 0; i <= imagem.rows; i += imagem.rows - 1)
    {
        for (int j = 0; j < imagem.cols; j++)
        {
            if (imagem.at<uchar>(i, j) == 255)
            {
                p.x = j;
                p.y = i;
                              
                floodFill(imagem, p, 0);
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
                floodFill(imagem, p, 0);
            }
        }
    }
}

void removeBuracos(Mat &imagem)
{
    /* Altera a cor de fundo padrão para uma nova cor. */
    floodFill(imagem, Point(0,0), 127);
    
    /* Remove todas as bolhas. */
    for (int i = 0; i < imagem.rows; i++)
    {
        for (int j = 0; j < imagem.cols; j++)
        {
            if (imagem.at<uchar>(i, j) == 0)
            {
                floodFill(imagem, Point(j,i), 255);
            }
        }
    }
    
    /* Retorna a cor de fundo par aa original. */
    floodFill(imagem, Point(0,0), 0);
}

vector<Moeda> detectarMoedas(Mat imagem,  int fechamento)
{

    Mat imagemColorida, imagemCinza, imagemBinaria, imagemBinariaClone, imagemDelimitada;
    vector<Moeda> moedas;
    vector<vector<Point> > contornos;
    vector<Vec4i> hierarquia;
    vector<Circulo> circulos;
    
    imagemColorida = imagem.clone();

     /* Suaviza a imagem lida. */
    blur(imagemColorida, imagemColorida, Size(3, 3));
    
    /* Mantenha a imagem original em três formatos: binária, em escala de cinza 
     * e colorida. A em escala de cinza será usada na detecção de bordas. */
    cvtColor(imagemColorida, imagemCinza, CV_BGR2GRAY);
    imagemBinaria = Mat(imagemCinza.size(), imagemCinza.type());
    imagemDelimitada = imagemColorida.clone();
    
    /* Usando threshold fixo. */
    //threshold(imagemCinza, imagemBinaria, 0, 255, THRESH_BINARY_INV|THRESH_OTSU);
    
    /* Usando threshold adaptativo (resultado melhor). */
    //adaptiveThreshold(imagemCinza, imagemBinaria, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 7, 5);
    adaptiveThreshold(imagemCinza, imagemBinaria, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 5);
    
    /* Remove moedas da borda. */
    removeMoedasBorda(imagemBinaria);

    /* Faz o fechameto da imagem binária. */
    dilate(imagemBinaria, imagemBinaria, Mat(), Point(-1, -1), fechamento, 1, 1);
    erode(imagemBinaria, imagemBinaria, Mat(), Point(-1, -1), fechamento, 1, 1); 
        
    /* Remove buracos dentro das moedas (só a borda interessa). */
    removeBuracos(imagemBinaria);
    
    /* Encontra contornos iniciais para o algoritmo de active contours. */
    imagemBinariaClone = imagemBinaria.clone();
    findContours(imagemBinariaClone, contornos, hierarquia, RETR_TREE, CV_CHAIN_APPROX_NONE);
          
    circulos = vector<Circulo>(contornos.size());
      
    for(uint i = 0; i < contornos.size(); i++ )
    { 
        if( contornos[i].size() > PONTOS_CIRCULO)
        { 
            minEnclosingCircle(contornos[i], circulos[i].centro, circulos[i].raio); 
        }
    }
      
    
    for( int i = 0; i < (int) contornos.size(); i++)
    { 
        Point2f centro = circulos[i].centro;
        double raio = circulos[i].raio;
        double areaContorno = contourArea(contornos[i]);
        double areaCirculo = M_PI * raio * raio;
        double relacaoAreas = areaContorno/areaCirculo;

        if(raio > LIMIAR_RAIO && relacaoAreas > LIMIAR_RELACAO_AREAS)
        {
            
            Rect ret = boundingRect(contornos[i]);
            
            /* Mantém apenas os píxels da moeda e apaga os externos. */
            for (int m = ret.y; m < ret.y + ret.height; m++)
            {
                for (int n = ret.x; n < ret.x + ret.width; n++)
                {
                    // Muda o fundo para branco, caso o ponto esteia fora do círculo.
                    if(pow(m - centro.y, 2) + pow(n - centro.x, 2) > pow(raio, 2))
                    {
                         imagemColorida.at<Vec3b>(m,n)[0] = 0; 
                         imagemColorida.at<Vec3b>(m,n)[1] = 0;
                         imagemColorida.at<Vec3b>(m,n)[2] = 0;
                    }
                     
                }
            }
            

            /* Cria as moedas identificadas. */
            Moeda moeda(Mat(imagemColorida, ret));
            moeda.raio = raio;
            moeda.centro = centro;
            moeda.retangulo = ret;
            moeda.contornos.push_back(contornos[i]);
            moeda.hierarquia = hierarquia;
            moedas.push_back(moeda);
            
        }
    }
     
     
    /* Exibe resultado da imagem segmentada. */
    
    //imshow("binaria", imagemBinaria);
    //waitKey(500);
     
    return moedas;
}

void exibirMoedas(const Mat &imagem, const vector<Moeda> &moedas)
{
    Mat imagemDelimitada = imagem.clone();
    
    if(RESULTADO_SEPARADO)
    {
        for(int i = 0; i < (int) moedas.size(); i++)
        {
            imshow("Moeda " + to_string(i), moedas[i].imagem);
        }
    }
    else
    {
        for(int i = 0; i < (int) moedas.size(); i++)
        {
            /* Desenha o retângulo e o círculo delimitador. */
            circle(imagemDelimitada, moedas[i].centro, moedas[i].raio, Scalar(0, 0, 255), 2);
            rectangle(imagemDelimitada, moedas[i].retangulo, Scalar(255, 0, 0), 2);
            putText(imagemDelimitada, "Moeda" + to_string(i), Point(moedas[i].retangulo.x-5, moedas[i].retangulo.y-5), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0), 1);
            drawContours(imagemDelimitada, moedas[i].contornos, 0, Scalar(0,255,255), 2);
        }
        
        imshow("delimitada", imagemDelimitada);
    }
    
    waitKey(500);

}


vector<Moeda> detectarTodasMoedas(Mat imagem)
{
    vector<Moeda> moedas, moedasMax;
    int fechamento = 1;
    
    moedas = detectarMoedas(imagem, fechamento);
    moedasMax = moedas;

    /* Tenta encontrar mais moedas e só para as tentativas se o resultado piorar. */
    while(fechamento <= MAX_FECHAMENTO && moedas.size() >= moedasMax.size())
    {   
        moedas = detectarMoedas(imagem, fechamento);
        cout << "[detectar] Usando fechamento de tamanho " << fechamento << endl;
        cout << "[detectar] Foram encontradas " << moedas.size() << " moedas." << endl;
        
        if(moedas.size() > moedasMax.size())
        {
            moedasMax = moedas;
        }
        
        fechamento *= 2;
        
    }
    
    return moedasMax;
}



/*
vector<Mat> detectarMoedasHough(vector<Mat> moedas)
{
    
    for(int i = 0; i < (int) moedas.size(); i++)
    {
        vector<Vec3f> circulos;
        cvtColor(moedas[i], moedas[i], CV_BGR2GRAY);
        
        int a = 50;
        int b = 50;
        
        while(circulos.size() != 1 && a < 800)
        {
            while(circulos.size() != 1 && b < 800)
            {
                HoughCircles(moedas[i], circulos, CV_HOUGH_GRADIENT, 1, moedas[i].rows/2, a, b, 0.9*(moedas[i].rows/2.0), moedas[i].rows);
                b += 50;
            }
            
            b = 0;
            a += 50;
        }
        
        for(int i = 0; i < (int) circulos.size(); i++)
        {
            Point centro(cvRound(circulos[i][0]), cvRound(circulos[i][1]));
            int raio = cvRound(circulos[i][2]);
            circle(moedas[i], centro, 3, Scalar(0,0,255), -1, 8, 0 );
            circle(moedas[i], centro, raio, Scalar(255,0,0), 3, 8, 0 );
            imshow("moeda" + to_string(i), moedas[i]);
            
        } 

    }

    return moedas;
}
*/
