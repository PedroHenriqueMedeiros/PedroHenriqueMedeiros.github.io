#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Parâmetros da MLP

#define NUM_MAX_ITER 10000
#define EPSILON 0.00001f
#define PESO_GRADIENTE 0.05f
#define MOMENTUM 0.05f


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

int main()
{
	// Definindo conjunto treinamento.
    Mat entradas(8, 10000, CV_32FC1);
    Mat saidas(8, 1, CV_32FC1);
    
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
    
    for(int i = 0; i < 10000; i++)
    {
		entradas.at<float>(0, i) = v10f[i];
		entradas.at<float>(1, i) = v10n[i];
		entradas.at<float>(2, i) = v25f[i];
		entradas.at<float>(3, i) = v25n[i];
		entradas.at<float>(4, i) = v50f[i];
		entradas.at<float>(5, i) = v50n[i];
		entradas.at<float>(6, i) = v100f[i];
		entradas.at<float>(7, i) = v100n[i];
		
	}
    
    saidas.at<float>(0,0) = 9;
    saidas.at<float>(1,0) = 11;
    saidas.at<float>(2,0) = 99;
    saidas.at<float>(3,0) = 101;
    saidas.at<float>(4,0) = 999;
    saidas.at<float>(5,0) = 1001;
    saidas.at<float>(6,0) = 9999;
    saidas.at<float>(7,0) = 10001; 
    
	
	Mat saidasMLP(8, 1, CV_32FC1);
	
	CvANN_MLP mlp;
	mlp.load("pesos.yml", "mlp");
	
	mlp.predict(entradas, saidasMLP);
	
	for(int i = 0; i < 8; i++)
	{
		cout << saidasMLP.at<float>(i, 0) << endl;
	}
	

	
	
}
