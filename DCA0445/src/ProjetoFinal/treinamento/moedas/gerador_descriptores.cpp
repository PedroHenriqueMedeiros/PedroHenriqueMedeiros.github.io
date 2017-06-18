#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp>

#include "boost/filesystem.hpp"   
#include <iostream>          
    
namespace fs = boost::filesystem;
       
using namespace cv;
using namespace std;

// Parâmetros da MLP

#define NUM_MAX_ITER 1000
#define EPSILON 1e-10f
#define LEARNING_RATE 0.1f
#define MOMENTUM 0.1f

#define NUM_AMOSTRAS 5


vector<string> getFilesInDirectory(const string& directory)
{
  std::vector<std::string> files;
  fs::path root(directory);
  fs::directory_iterator it_end;
  for (fs::directory_iterator it(root); it != it_end; ++it)
  {
      if (fs::is_regular_file(it->path()))
      {
          files.push_back(it->path().string());
      }
  }
  return files;
}




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
    camadas.row(1) = Scalar(10);
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
    Mat entradas(NUM_AMOSTRAS, 10000, CV_32FC1);
    Mat saidas(NUM_AMOSTRAS, 1, CV_32FC1);
    
    /*
    
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
    * 
    */
    
    /*
    vector<KeyPoint> keypoints = detectKeyPoints(m50n); 
	Mat descriptors = computeDescriptors(m50n, keypoints);
	cout << "kp = " << keypoints.size() << endl;
	cout << "descr " << descriptors.rows << " cols = " << descriptors.cols << endl;
	
	Mat t1;
	drawKeypoints(m50n, keypoints, t1);
	imshow("t1", t1);
	
	
	cout << "-----" << endl;
    vector<KeyPoint> keypoints2 = detectKeyPoints(m10f); 
	Mat descriptors2 = computeDescriptors(m10f, keypoints2);
	cout << "kp = " << keypoints2.size() << endl;
	cout << "descr " << descriptors2.rows << " cols = " << descriptors2.cols << endl;  
	
		Mat t2;
	drawKeypoints(m10f, keypoints2, t2);
	imshow("t2", t2);
	
	cout << "-----" << endl;
    vector<KeyPoint> keypoints3 = detectKeyPoints(m100n); 
	Mat descriptors3 = computeDescriptors(m100n, keypoints3);
	cout << "kp = " << keypoints3.size() << endl;
	cout << "descr " << descriptors3.rows << " cols = " << descriptors3.cols << endl;  
    	
	Mat t3;
	drawKeypoints(m100n, keypoints3, t3);
	imshow("t3", t3);
	
	
	
	Mat x = imread("10f.jpg", CV_LOAD_IMAGE_COLOR);
	vector<KeyPoint> kp;
	Mat d;
	
	auto det = DescriptorExtractor::create("MSER");
	DynamicAdaptedFeatureDetector detector(det, 10, 100);
	detector.detect(x, kp);
	
	cout << kp.size() << endl;
	
	waitKey(0);
	* 
	* */
    
    /*
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
	
	saidas.at<float>(0,0) = -0.9;
	saidas.at<float>(1,0) = -0.8;
	saidas.at<float>(2,0) = -0.4;
	saidas.at<float>(3,0) = -0.3;
	saidas.at<float>(4,0) = 0.3;
	saidas.at<float>(5,0) = 0.4;
	saidas.at<float>(6,0) = 0.8;
	saidas.at<float>(7,0) = 0.9;	

    
    treinar(entradas, saidas);
    
    Mat saidasMLP(8, 1, CV_32FC1);
	
	CvANN_MLP mlp;
	mlp.load("pesos.yml", "mlp");
	
	mlp.predict(entradas, saidasMLP);
	
	
	for(int i = 0; i < saidasMLP.rows; i++)
	{
		cout << saidasMLP.at<float>(i, 0) << endl;
	}
	
    */
    

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
	
	// Calcula descritores para as imagens de amostra. 
	
	/*
	
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
		
		featuresUnclustered.push_back(dMoedas10f[i]);
		featuresUnclustered.push_back(dMoedas10n[i]);
		featuresUnclustered.push_back(dMoedas25f[i]);
		featuresUnclustered.push_back(dMoedas25n[i]);
		featuresUnclustered.push_back(dMoedas50f[i]);
		featuresUnclustered.push_back(dMoedas50n[i]);
		featuresUnclustered.push_back(dMoedas100f[i]);
		featuresUnclustered.push_back(dMoedas100n[i]);
		
	}
	
	for(int i = 0; i < NUM_AMOSTRAS; i++)
	{
		cout << dMoedas10f[i].size() << endl;
		cout << dMoedas10n[i].size() << endl;
		cout << dMoedas25f[i].size() << endl;
		cout << dMoedas25n[i].size() << endl;
		cout << dMoedas50f[i].size() << endl;
		cout << dMoedas50n[i].size() << endl;
		cout << dMoedas100f[i].size() << endl;
		cout << dMoedas100n[i].size() << endl;
		cout << "----------------" << endl;
	}
	* 
	* */
	
	/*
	
	int tamDicionario = 200; // Numero de 'bags'
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(tamDicionario, tc, retries, flags);
	Mat dictionary = bowTrainer.cluster(featuresUnclustered);    
	FileStorage fs("dicionario.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
	
	
	*/
	
	
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
                  		
	
	
	
	/*
	
	

	
	Mat labels;
    Mat vocabulary;
    int networkInputSize = 32;
    
	

	

	
	
	Mat t1 = moedas10f[3];
	blur(t1, t1, Size(3, 3));
	
	vector<KeyPoint> kp = detectKeyPoints(t1);
	Mat d = computeDescriptors(t1, kp);
	Mat t2;
	drawKeypoints(t1, kp, t2);
	cout << "num de kp = " << kp.size() << " e num de d = " << d.size() << endl;
	imshow("t2", t2);
	waitKey(0);
	
	*/
     
	
}
