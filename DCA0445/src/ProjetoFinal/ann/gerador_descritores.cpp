#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>          
    
using namespace cv;
using namespace std;

#define TIPOS_MOEDAS 8
#define NUM_AMOSTRAS 20


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
	
	vector<Mat> todosDescritores;
	
	Mat featuresUnclustered;
	
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
		
	Mat dicionario; 
    FileStorage fs("dicionario.yml", FileStorage::READ);
    fs["vocabulary"] >> dicionario;
    fs.release();    
    
    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift feature point extracter
    Ptr<FeatureDetector> detector(new SiftFeatureDetector());
    //create Sift descriptor extractor
    Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);    
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(dicionario);
 
   
    
    for(int i = 0; i < NUM_AMOSTRAS; i++)
	{
		vector<KeyPoint> kp10f;
		vector<KeyPoint> kp10n;
		vector<KeyPoint> kp25f;
		vector<KeyPoint> kp25n;
		vector<KeyPoint> kp50f;
		vector<KeyPoint> kp50n;
		vector<KeyPoint> kp100f;
		vector<KeyPoint> kp100n;
		
		Mat bd10f;
		Mat bd10n; 
		Mat bd25f; 
		Mat bd25n; 
		Mat bd50f; 
		Mat bd50n; 
		Mat bd100f; 
		Mat bd100n;  
		
		
		FileStorage fs10f("10f/" + to_string(i+1) + ".yml", FileStorage::WRITE);    
		FileStorage fs10n("10n/" + to_string(i+1) + ".yml", FileStorage::WRITE); 
		FileStorage fs25f("25f/" + to_string(i+1) + ".yml", FileStorage::WRITE);    
		FileStorage fs25n("25n/" + to_string(i+1) + ".yml", FileStorage::WRITE); 
		FileStorage fs50f("50f/" + to_string(i+1) + ".yml", FileStorage::WRITE);    
		FileStorage fs50n("50n/" + to_string(i+1) + ".yml", FileStorage::WRITE); 
		FileStorage fs100f("100f/" + to_string(i+1) + ".yml", FileStorage::WRITE);    
		FileStorage fs100n("100n/" + to_string(i+1) + ".yml", FileStorage::WRITE); 		
		   
		detector->detect(moedas10f[i], kp10f);
		detector->detect(moedas10n[i], kp10n);
		detector->detect(moedas25f[i], kp25f);
		detector->detect(moedas25n[i], kp25n);
		detector->detect(moedas50f[i], kp50f);
		detector->detect(moedas50n[i], kp50n);
		detector->detect(moedas100f[i], kp100f);
		detector->detect(moedas100n[i], kp100n);
		
		bowDE.compute(moedas10f[i], kp10f, bd10f);
		bowDE.compute(moedas10n[i], kp10n, bd10n);
		bowDE.compute(moedas25f[i], kp25f, bd25f);
		bowDE.compute(moedas25n[i], kp25n, bd25n);
		bowDE.compute(moedas50f[i], kp50f, bd50f);
		bowDE.compute(moedas50n[i], kp50n, bd50n);		
		bowDE.compute(moedas100f[i], kp100f, bd100f);
		bowDE.compute(moedas100n[i], kp100n, bd100n);		
		
		
		fs10f << "m10f" << bd10f;
		fs10n << "m10n" << bd10n;
		fs25f << "m25f" << bd25f;
		fs25n << "m25n" << bd25n;
		fs50f << "m50f" << bd50f;
		fs50n << "m50n" << bd50n;
		fs100f << "m100f" << bd100f;
		fs100n << "m100n" << bd100n;		
		
		fs10f.release();  
		fs10n.release();  
		fs25f.release();  
		fs25n.release();  
		fs50f.release();  
		fs50n.release();  
		fs100f.release();  
		fs100n.release();  
					
	}
                  		
	

     
	
}
