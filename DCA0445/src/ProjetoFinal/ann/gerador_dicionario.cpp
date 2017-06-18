#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>          
    
using namespace cv;
using namespace std;

#define TIPOS_MOEDAS 8
#define NUM_AMOSTRAS 20

#define TAM_DICIONARIO 800
#define NUM_MAX_ITER 200
#define ERRO_MAX 1e-5
#define NOVAS_TENTATIVAS 1

vector<KeyPoint> detectKeyPoints(const Mat &image) 
{
    auto featureDetector = FeatureDetector::create("SIFT");
    vector<KeyPoint> keyPoints;
    featureDetector->detect(image, keyPoints);
    return keyPoints;
}

Mat computeDescriptors(const Mat &image, vector<KeyPoint> &keyPoints)
{
    auto featureExtractor = DescriptorExtractor::create("SIFT");
    Mat descriptors;
    featureExtractor->compute(image, keyPoints, descriptors);
    return descriptors;
}

int main()
{
	initModule_nonfree();
    

	vector<Mat> moedas10f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas10n(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas25f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas25n(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas50f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas50n(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas100f(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas100n(NUM_AMOSTRAS, Mat());
	
	vector<Mat> dMoedas10f(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas10n(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas25f(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas25n(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas50f(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas50n(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas100f(NUM_AMOSTRAS, Mat());
	vector<Mat> dMoedas100n(NUM_AMOSTRAS, Mat());
		
	Mat todosDescritores;
	
	/* Lê todas as amostras. */
	for(int i = 0; i < NUM_AMOSTRAS; i++)
	{
		moedas10f[i] = imread("10f/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas10n[i] = imread("10n/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas25f[i] = imread("25f/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas25n[i] = imread("25n/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas50f[i] = imread("50f/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas50n[i] = imread("50n/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas100f[i] = imread("100f/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas100n[i] = imread("100n/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
	}
	
	// Calcula descritores para as imagens de amostra. 
	for(int i = 0; i < NUM_AMOSTRAS; i++)
	{
		// Encontra os keypoints.
		vector<KeyPoint> kp10f = detectKeyPoints(moedas10f[i]);
		vector<KeyPoint> kp10n = detectKeyPoints(moedas10n[i]);
		vector<KeyPoint> kp25f = detectKeyPoints(moedas25f[i]);
		vector<KeyPoint> kp25n = detectKeyPoints(moedas25n[i]);
		vector<KeyPoint> kp50f = detectKeyPoints(moedas50f[i]);
		vector<KeyPoint> kp50n = detectKeyPoints(moedas50n[i]);
		vector<KeyPoint> kp100f = detectKeyPoints(moedas100f[i]);
		vector<KeyPoint> kp100n = detectKeyPoints(moedas100n[i]);
		
		// Calcula os descritores.
		dMoedas10f[i] = computeDescriptors(moedas10f[i], kp10f);
		dMoedas10n[i] = computeDescriptors(moedas10n[i], kp10n);
		dMoedas25f[i] = computeDescriptors(moedas25f[i], kp25f);
		dMoedas25n[i] = computeDescriptors(moedas25n[i], kp25n);
		dMoedas50f[i] = computeDescriptors(moedas50f[i], kp50f);
		dMoedas50n[i] = computeDescriptors(moedas50n[i], kp50n);
		dMoedas100f[i] = computeDescriptors(moedas100f[i], kp100f);
		dMoedas100n[i] = computeDescriptors(moedas100n[i], kp100n);
		
		todosDescritores.push_back(dMoedas10f[i]);
		todosDescritores.push_back(dMoedas10n[i]);
		todosDescritores.push_back(dMoedas25f[i]);
		todosDescritores.push_back(dMoedas25n[i]);
		todosDescritores.push_back(dMoedas50f[i]);
		todosDescritores.push_back(dMoedas50n[i]);
		todosDescritores.push_back(dMoedas100f[i]);
		todosDescritores.push_back(dMoedas100n[i]);
		
	}
	
	
	TermCriteria tc(CV_TERMCRIT_ITER, NUM_MAX_ITER, ERRO_MAX);
	
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(TAM_DICIONARIO, tc, NOVAS_TENTATIVAS, KMEANS_PP_CENTERS);
	Mat dicionario = bowTrainer.cluster(todosDescritores);  
	
	// Salva o dicionário.  
	FileStorage fs("dicionario.yml", FileStorage::WRITE);
	fs << "vocabulary" << dicionario;
	fs.release();
		
}
