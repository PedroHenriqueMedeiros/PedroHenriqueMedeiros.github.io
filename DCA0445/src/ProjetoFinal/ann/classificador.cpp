#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>          
          
using namespace cv;
using namespace std;

// Parâmetros da MLP

#define TIPOS_MOEDAS 8
#define NUM_AMOSTRAS 5

#define TAM_DICIONARIO 800

Mat gerarDescritor(const Mat &imagem)
{
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
    
    vector<KeyPoint> kp;
    Mat bd;
    
    detector->detect(imagem, kp);
	bowDE.compute(imagem, kp, bd);
	
	return bd;
 
}

int main()
{
	initModule_nonfree();
	
	Mat imagem = imread("teste.jpg", CV_LOAD_IMAGE_COLOR);
	Mat descritor = gerarDescritor(imagem);
	
    Mat saidasMLP(TIPOS_MOEDAS * NUM_AMOSTRAS, 1, CV_32FC1);
	
	CvANN_MLP mlp;
	mlp.load("pesos.yml", "mlp");
	
	mlp.predict(descritor, saidasMLP);

	for(int i = 0; i < saidasMLP.rows; i++)
	{
		cout << saidasMLP.at<float>(i, 0) << endl;
	}
	

    
    return 0;
    

	
	
}
