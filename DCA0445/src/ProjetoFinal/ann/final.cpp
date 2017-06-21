#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>          
    
using namespace cv;
using namespace std;

#define TIPOS_MOEDAS 8
#define NUM_AMOSTRAS 5

// Parâmetros da MLP

#define NUM_MAX_ITER 1000
#define EPSILON 1e-4f
#define LEARNING_RATE 0.1f
#define MOMENTUM 0.1f

#define NUM_MI 3
#define CANNY_THRESH 10

vector<float> mat2vec(const Mat &mat)
{
    vector<float> array;
    if (mat.isContinuous()) {
      array.assign(mat.datastart, mat.dataend);
    } else {
      for (int i = 0; i < mat.rows; ++i) {
        array.insert(array.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i)+mat.cols);
      }
    }
    
    return array;
}


void treinar(const Mat &entradas, const Mat& saidas) {

	
	// Definição das camadas.
    Mat camadas = cv::Mat(3, 1, CV_32SC1);
    camadas.row(0) = Scalar(entradas.cols);
    camadas.row(1) = Scalar(32);
    camadas.row(2) = Scalar(saidas.cols);

	// Definição dos parâmetros;
    CvANN_MLP_TrainParams params;
    CvTermCriteria criteria;
    criteria.max_iter = NUM_MAX_ITER;
    criteria.epsilon = EPSILON;
    criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
    params.train_method = CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale = LEARNING_RATE;
    params.bp_moment_scale = MOMENTUM;
    params.term_crit = criteria;
	
	
	CvANN_MLP mlp;
    mlp.create(camadas, CvANN_MLP::SIGMOID_SYM);

    // Realiza o treinamento.
    int numIteracoes = mlp.train(entradas, saidas, cv::Mat(), cv::Mat(), params);
    
    cout << "[info] Treinamento concluído após " << numIteracoes << " iterações." << endl;
    
    
    // Salva o resultado.
    
    FileStorage fs("mlp.yml", FileStorage::WRITE); // or xml
	mlp.write(*fs, "mlp"); 
    
    //mlp.predict(sample, response);

}

int main()
{
	initModule_nonfree();
    
    // Definindo conjunto treinamento.
    Mat entradas(TIPOS_MOEDAS * NUM_AMOSTRAS, 6400, CV_32FC1);
    Mat saidas = Mat::zeros(TIPOS_MOEDAS * NUM_AMOSTRAS, TIPOS_MOEDAS, CV_32FC1);
    
	vector<Mat> moedas10f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas10n(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas25f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas25n(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas50f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas50n(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas100f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas100n(NUM_AMOSTRAS, Mat());
	
	vector<vector<float> > dMoedas10f(NUM_AMOSTRAS, vector<float>(6400, 0));
	vector<vector<float> >  dMoedas10n(NUM_AMOSTRAS, vector<float>(6400, 0));
	vector<vector<float> >  dMoedas25f(NUM_AMOSTRAS, vector<float>(6400, 0));
	vector<vector<float> >  dMoedas25n(NUM_AMOSTRAS, vector<float>(6400, 0));
	vector<vector<float> >  dMoedas50f(NUM_AMOSTRAS, vector<float>(6400, 0));
	vector<vector<float> >  dMoedas50n(NUM_AMOSTRAS, vector<float>(6400, 0));
	vector<vector<float> >  dMoedas100f(NUM_AMOSTRAS, vector<float>(6400, 0));
	vector<vector<float> >  dMoedas100n(NUM_AMOSTRAS, vector<float>(6400, 0));
	
	/* Lê todas as amostras. */
	for(int i = 0; i < NUM_AMOSTRAS; i++)
	{
		moedas10f[i] = imread("10f/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		moedas10n[i] = imread("10n/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		moedas25f[i] = imread("25f/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		moedas25n[i] = imread("25n/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		moedas50f[i] = imread("50f/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		moedas50n[i] = imread("50n/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		moedas100f[i] = imread("100f/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		moedas100n[i] = imread("100n/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
        
        resize(moedas10f[i], moedas10f[i], Size(80, 80));
        resize(moedas10n[i], moedas10n[i], Size(80, 80));
        resize(moedas25f[i], moedas25f[i], Size(80, 80));
        resize(moedas25n[i], moedas25n[i], Size(80, 80));
        resize(moedas50f[i], moedas50f[i], Size(80, 80));
        resize(moedas50n[i], moedas50n[i], Size(80, 80));
        resize(moedas100f[i], moedas100f[i], Size(80, 80));
        resize(moedas100n[i], moedas100n[i], Size(80, 80));
        
        
        IplImage ipl10f = moedas10f[i];
        IplImage ipl10n = moedas10n[i];
        IplImage ipl25f = moedas25f[i];
        IplImage ipl25n = moedas25n[i];
        IplImage ipl50f = moedas50f[i];
        IplImage ipl50n = moedas50n[i];
        IplImage ipl100f = moedas100f[i];
        IplImage ipl100n = moedas100n[i];
        
        IplImage *p10f = &ipl10f;
        IplImage *p10n = &ipl10n;
        IplImage *p25f = &ipl25f;
        IplImage *p25n = &ipl25n;
        IplImage *p50f = &ipl50f;
        IplImage *p50n = &ipl50n;
        IplImage *p100f = &ipl100f;
        IplImage *p100n = &ipl100n;
        
        cvLinearPolar(p10f, p10f, CvPoint2D32f(Point(40, 40)), 40);
        cvLinearPolar(p10n, p10n, CvPoint2D32f(Point(40, 40)), 40);
        cvLinearPolar(p25f, p25f, CvPoint2D32f(Point(40, 40)), 40);
        cvLinearPolar(p25n, p25n, CvPoint2D32f(Point(40, 40)), 40);        
        cvLinearPolar(p50f, p50f, CvPoint2D32f(Point(40, 40)), 40);
        cvLinearPolar(p50n, p50n, CvPoint2D32f(Point(40, 40)), 40);
        cvLinearPolar(p100f, p100f, CvPoint2D32f(Point(40, 40)), 40);
        cvLinearPolar(p100n, p100n, CvPoint2D32f(Point(40, 40)), 40);

	}
    
    for(int i = 0; i < NUM_AMOSTRAS; i++)
    {
        dMoedas10f[i] = mat2vec(moedas10f[i]);
        dMoedas10n[i] = mat2vec(moedas10n[i]);
        dMoedas25f[i] = mat2vec(moedas25f[i]);
        dMoedas25n[i] = mat2vec(moedas25n[i]);
        dMoedas50f[i] = mat2vec(moedas50f[i]);
        dMoedas50n[i] = mat2vec(moedas50n[i]);
        dMoedas100f[i] = mat2vec(moedas100f[i]);
        dMoedas100n[i] = mat2vec(moedas100n[i]);
    }
	    
    
    for(int i = 0; i < TIPOS_MOEDAS * NUM_AMOSTRAS; i += TIPOS_MOEDAS)
    {
        for(int j = 0; j < 6400; j++)
        {
            entradas.at<float>(i,j) = dMoedas10f[i/TIPOS_MOEDAS][j];
            entradas.at<float>(i+1,j) = dMoedas10n[i/TIPOS_MOEDAS][j];
            entradas.at<float>(i+2,j) = dMoedas25f[i/TIPOS_MOEDAS][j];
            entradas.at<float>(i+3,j) = dMoedas25n[i/TIPOS_MOEDAS][j];
            entradas.at<float>(i+4,j) = dMoedas50f[i/TIPOS_MOEDAS][j];
            entradas.at<float>(i+5,j) = dMoedas50n[i/TIPOS_MOEDAS][j];
            entradas.at<float>(i+6,j) = dMoedas100f[i/TIPOS_MOEDAS][j];
            entradas.at<float>(i+7,j) = dMoedas100n[i/TIPOS_MOEDAS][j];
        }
        
        saidas.at<float>(i,0) = 1;
        saidas.at<float>(i+1,1) = 1;
        saidas.at<float>(i+2,2) = 1;
        saidas.at<float>(i+3,3) = 1;
        saidas.at<float>(i+4,4) = 1;
        saidas.at<float>(i+5,5) = 1;
        saidas.at<float>(i+6,6) = 1;
        saidas.at<float>(i+7,7) = 1;
    }
    
    
    treinar(entradas, saidas);
    
	Mat saidasMLP(TIPOS_MOEDAS * NUM_AMOSTRAS, TIPOS_MOEDAS, CV_32FC1);
	
	CvANN_MLP mlp;
	mlp.load("mlp.yml", "mlp");
	
	mlp.predict(entradas, saidasMLP);

	for(int i = 0; i < saidasMLP.rows; i++)
	{
        for(int j = 0; j < TIPOS_MOEDAS; j++)
        {
            cout << saidasMLP.at<float>(i, j) << endl;
        }
        cout <<  "----------------------------------------" << endl;
    }


		
}
