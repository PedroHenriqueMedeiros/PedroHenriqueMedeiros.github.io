#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Parâmetros da MLP

#define NUM_MAX_ITER 10000
#define EPSILON 0.00001f
#define PESO_GRADIENTE 0.1f
#define MOMENTUM 0.1f


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
    Mat camadas = cv::Mat(5, 1, CV_32SC1);
    camadas.row(0) = Scalar(entradas.cols);
    camadas.row(1) = Scalar(3);
    camadas.row(2) = Scalar(10);
    camadas.row(3) = Scalar(2);
    camadas.row(4) = Scalar(saidas.cols);

	// Definição dos parâmetros;
    CvANN_MLP_TrainParams params;
    CvTermCriteria criteria;
    criteria.max_iter = NUM_MAX_ITER;
    criteria.epsilon = EPSILON;
    criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
    params.train_method = CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale = PESO_GRADIENTE;
    params.bp_moment_scale = MOMENTUM;
    params.term_crit = criteria;
	
	
	CvANN_MLP mlp;
    mlp.create(camadas);

    // Realiza o treinamento.
    mlp.train(entradas, saidas, cv::Mat(), cv::Mat(), params);
    Mat response(1, 1, CV_32FC1);
    
    
    // Salva o resultado.
    
    FileStorage fs("pesos.yml", FileStorage::WRITE); // or xml
	mlp.write(*fs, "mlp"); 
    
    //mlp.predict(sample, response);

}

int main()
{
	
	// Definindo conjunto treinamento.
    Mat entradas(80, 10000, CV_32FC1);
    Mat saidas(80, 1, CV_32FC1);
    
    Mat m10f = imread("10f.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v10f = mat2vec(m10f);
    Mat m10n = imread("10n.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v10n = mat2vec(m10n);
    Mat m25f = imread("25f.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v25f = mat2vec(m25f);
    Mat m25n = imread("25n.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v25n = mat2vec(m25n);
    Mat m50f = imread("50f.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v50f = mat2vec(m50f);
    Mat m50n = imread("50n.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v50n = mat2vec(m50n);
    Mat m100f = imread("100f.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v100f = mat2vec(m100f);
    Mat m100n = imread("100n.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    vector<float> v100n = mat2vec(m100n);
    
    for(int h = 0; h < 80; h += 8)
    {
		for(int i = 0; i < 10000; i++)
		{
			entradas.at<float>(h, i) = v10f[i];
			entradas.at<float>(h+1, i) = v10n[i];
			entradas.at<float>(h+2, i) = v25f[i];
			entradas.at<float>(h+3, i) = v25n[i];
			entradas.at<float>(h+4, i) = v50f[i];
			entradas.at<float>(h+5, i) = v50n[i];
			entradas.at<float>(h+6, i) = v100f[i];
			entradas.at<float>(h+7, i) = v100n[i];
			
		}
		
		saidas.at<float>(h,0) = 1;
		saidas.at<float>(h+1,0) = 1;
		saidas.at<float>(h+2,0) = 3;
		saidas.at<float>(h+3,0) = -3;
		saidas.at<float>(h+4,0) = 5;
		saidas.at<float>(h+5,0) = 5;
		saidas.at<float>(h+6,0) = 7;
		saidas.at<float>(h+7,0) = 7; 
		
		
		
	}
    

    
    treinar(entradas, saidas);
	
	
}
