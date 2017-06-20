#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>          
          
using namespace cv;
using namespace std;

// Parâmetros da MLP

#define NUM_MAX_ITER 100000
#define EPSILON 1e-10f
#define LEARNING_RATE 0.1f
#define MOMENTUM 0.1f

#define TIPOS_MOEDAS 8
#define NUM_AMOSTRAS 60

// Rótulos da classificação

#define TAM_DICIONARIO 800

vector<float> mat2vec(Mat &mat)
{
	vector<float> array;
	if (mat.isContinuous()) {
	  array.assign((float*)mat.datastart, (float*)mat.dataend);
	} else {
	  for (int i = 0; i < mat.rows; ++i) {
		array.insert(array.end(), (float*)mat.ptr<uchar>(i), (float*)mat.ptr<uchar>(i)+mat.cols);
	  }
	}
	
	return array;
}

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
    
    FileStorage fs("pesos.yml", FileStorage::WRITE); // or xml
	mlp.write(*fs, "mlp"); 
    
    //mlp.predict(sample, response);

}

int main()
{
	initModule_nonfree();
	
	// Definindo conjunto treinamento.
    Mat entradas(TIPOS_MOEDAS * NUM_AMOSTRAS, TAM_DICIONARIO, CV_32FC1);
    Mat saidas = Mat::zeros(TIPOS_MOEDAS * NUM_AMOSTRAS, TIPOS_MOEDAS, CV_32FC1);
    
    vector<Mat> dMoedas10f(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas10n(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas25f(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas25n(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas50f(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas50n(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas100f(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas100n(NUM_AMOSTRAS, Mat());
    
    
    /* Lê todos os descritores as amostras. */
	for(int i = 0; i < NUM_AMOSTRAS; i++)
	{
		
		FileStorage fs10f("10f/" + to_string(i+1) + ".yml", FileStorage::READ);    
		FileStorage fs10n("10n/" + to_string(i+1) + ".yml", FileStorage::READ); 
		FileStorage fs25f("25f/" + to_string(i+1) + ".yml", FileStorage::READ);    
		FileStorage fs25n("25n/" + to_string(i+1) + ".yml", FileStorage::READ); 
		FileStorage fs50f("50f/" + to_string(i+1) + ".yml", FileStorage::READ);    
		FileStorage fs50n("50n/" + to_string(i+1) + ".yml", FileStorage::READ); 
		FileStorage fs100f("100f/" + to_string(i+1) + ".yml", FileStorage::READ);    
		FileStorage fs100n("100n/" + to_string(i+1) + ".yml", FileStorage::READ); 	
		
		fs10f["m10f"] >> dMoedas10f[i];
		fs10n["m10n"] >> dMoedas10n[i];
		fs25f["m25f"] >> dMoedas25f[i];
		fs25n["m25n"] >> dMoedas25n[i];
		fs50f["m50f"] >> dMoedas50f[i];
		fs50n["m50n"] >> dMoedas50n[i];
		fs100f["m100f"] >> dMoedas100f[i];
		fs100n["m100n"] >> dMoedas100n[i];
		
		
		fs10f.release();  
		fs10n.release();  
		fs25f.release();  
		fs25n.release();  
		fs50f.release();  
		fs50n.release();  
		fs100f.release();  
		fs100n.release(); 
	}
    
    for(int i = 0; i < TIPOS_MOEDAS * NUM_AMOSTRAS; i += TIPOS_MOEDAS)
    {
		dMoedas10f[i/TIPOS_MOEDAS].row(0).copyTo(entradas.row(i));
		dMoedas10n[i/TIPOS_MOEDAS].row(0).copyTo(entradas.row(i+1));
		dMoedas25f[i/TIPOS_MOEDAS].row(0).copyTo(entradas.row(i+2));
		dMoedas25n[i/TIPOS_MOEDAS].row(0).copyTo(entradas.row(i+3));
		dMoedas50f[i/TIPOS_MOEDAS].row(0).copyTo(entradas.row(i+4));
		dMoedas50n[i/TIPOS_MOEDAS].row(0).copyTo(entradas.row(i+5));
		dMoedas100f[i/TIPOS_MOEDAS].row(0).copyTo(entradas.row(i+6));
		dMoedas100n[i/TIPOS_MOEDAS].row(0).copyTo(entradas.row(i+7));
        
        saidas.at<float>(i,0) = 1;
        saidas.at<float>(i+1,1) = 1;
        saidas.at<float>(i+2,2) = 1;
        saidas.at<float>(i+3,3) = 1;
        saidas.at<float>(i+4,4) = 1;
        saidas.at<float>(i+5,5) = 1;
        saidas.at<float>(i+6,6) = 1;
        saidas.at<float>(i+7,7) = 1;
  
	}
    
    for(int i = 0; i < TIPOS_MOEDAS * NUM_AMOSTRAS; i++)
    {
        for(int j = 0; j < TIPOS_MOEDAS; j++)
        {
            cout << saidas.at<float>(i,j) << ", ";
        }
        
        cout << endl;
    }
    
    treinar(entradas, saidas);
    
    Mat saidasMLP(TIPOS_MOEDAS * NUM_AMOSTRAS, 8, CV_32FC1);
	
	CvANN_MLP mlp;
	mlp.load("pesos.yml", "mlp");
	
	mlp.predict(entradas, saidasMLP);


	for(int i = 0; i < saidasMLP.rows; i++)
	{
        for(int j = 0; j < TIPOS_MOEDAS; j++)
        {
            cout << saidasMLP.at<float>(i, j) << " ";
        }
        cout << endl <<  "-----------------" << endl;
	}

    
    return 0;
    

	
	
}