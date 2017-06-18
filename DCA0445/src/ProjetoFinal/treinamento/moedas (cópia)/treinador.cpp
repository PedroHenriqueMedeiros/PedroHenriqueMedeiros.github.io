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
	}
	
	/* Calcula descritores para as imagens de amostra. */
	
	Mat t1 = moedas100f[4];
	vector<KeyPoint> kp = detectKeyPoints(t1);
	Mat d = computeDescriptors(t1, kp);
	Mat t2;
	drawKeypoints(t1, kp, t2);
	cout << "num de kp = " << kp.size() << " e num de d = " << d.size() << endl;
	imshow("t2", t2);
	waitKey(0);
	
     
	
}
