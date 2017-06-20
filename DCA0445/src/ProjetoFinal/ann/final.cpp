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

#define NUM_MAX_ITER 100000
#define EPSILON 1e-6f
#define LEARNING_RATE 0.1f
#define MOMENTUM 0.7f

#define NUM_MI 3

void treinar(const Mat &entradas, const Mat& saidas) {

	
	// Definição das camadas.
    Mat camadas = cv::Mat(3, 1, CV_32SC1);
    camadas.row(0) = Scalar(entradas.cols);
    camadas.row(1) = Scalar((entradas.cols + saidas.cols)/2);
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
    Mat entradas(TIPOS_MOEDAS * NUM_AMOSTRAS, NUM_MI, CV_32FC1);
    Mat saidas = Mat::zeros(TIPOS_MOEDAS * NUM_AMOSTRAS, TIPOS_MOEDAS, CV_32FC1);
    
	vector<Mat> moedas10f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas10n(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas25f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas25n(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas50f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas50n(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas100f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas100n(NUM_AMOSTRAS, Mat());
	
	vector<vector<double> > dMoedas10f(NUM_AMOSTRAS, vector<double>(7));
	vector<vector<double> > dMoedas10n(NUM_AMOSTRAS, vector<double>(7));
	vector<vector<double> > dMoedas25f(NUM_AMOSTRAS, vector<double>(7));
	vector<vector<double> > dMoedas25n(NUM_AMOSTRAS, vector<double>(7));
	vector<vector<double> > dMoedas50f(NUM_AMOSTRAS, vector<double>(7));
	vector<vector<double> > dMoedas50n(NUM_AMOSTRAS, vector<double>(7));
	vector<vector<double> > dMoedas100f(NUM_AMOSTRAS, vector<double>(7));
	vector<vector<double> > dMoedas100n(NUM_AMOSTRAS, vector<double>(7));
	
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
		
		equalizeHist(moedas10f[i], moedas10f[i]);
		equalizeHist(moedas10n[i], moedas10n[i]);
		equalizeHist(moedas25f[i], moedas25f[i]);
		equalizeHist(moedas25n[i], moedas25n[i]);
		equalizeHist(moedas50f[i], moedas50f[i]);
		equalizeHist(moedas50n[i], moedas50n[i]);
		equalizeHist(moedas100f[i], moedas100f[i]);
		equalizeHist(moedas100n[i], moedas100n[i]);
	}
	
	// Calcula descritores para as imagens de amostra. 
	for(int i = 0; i < NUM_AMOSTRAS; i++)
	{
        double mi10f[7];
        double mi10n[7];
        double mi25f[7];
        double mi25n[7];
        double mi50f[7];
        double mi50n[7];
        double mi100f[7];
        double mi100n[7];
        
        Moments mom10f = moments(moedas10f[i], false);
        Moments mom10n = moments(moedas10n[i], false);
        Moments mom25f = moments(moedas25f[i], false);
        Moments mom25n = moments(moedas25n[i], false);
        Moments mom50f = moments(moedas50f[i], false);
        Moments mom50n = moments(moedas50n[i], false);
        Moments mom100f = moments(moedas100f[i], false);
        Moments mom100n = moments(moedas100n[i], false);
        
        HuMoments(mom10f, mi10f);
        HuMoments(mom10n, mi10n);
        HuMoments(mom25f, mi25f);
        HuMoments(mom25n, mi25n);
        HuMoments(mom50f, mi50f);
        HuMoments(mom50n, mi50n);
        HuMoments(mom100f, mi100f);
        HuMoments(mom100n, mi100n);
        
        for(int j = 0; j < NUM_MI; j++)
        {
            dMoedas10f[i][j] = mi10f[j];
            dMoedas10n[i][j] = mi10n[j];
            dMoedas25f[i][j] = mi25f[j];
            dMoedas25n[i][j] = mi25n[j];
            dMoedas50f[i][j] = mi50f[j];
            dMoedas50n[i][j] = mi50n[j];
            dMoedas100f[i][j] = mi100f[j];
            dMoedas100n[i][j] = mi100n[j];
            
        }
        
	}
    
    
    for(int i = 0; i < TIPOS_MOEDAS * NUM_AMOSTRAS; i += TIPOS_MOEDAS)
    {
        for(int j = 0; j < NUM_MI; j++)
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
    
    /*
	Mat saidasMLP(TIPOS_MOEDAS * NUM_AMOSTRAS, TIPOS_MOEDAS, CV_32FC1);
	
	CvANN_MLP mlp;
	mlp.load("pesos.yml", "mlp");
	
	mlp.predict(entradas, saidasMLP);

	for(int i = 0; i < saidasMLP.rows; i++)
	{
        for(int j = 0; j < TIPOS_MOEDAS; j++)
        {
            cout << saidasMLP.at<float>(i, j) << endl;
        }
        cout <<  "----------------------------------------" << endl;
    }
    * */
	
	
		
}
