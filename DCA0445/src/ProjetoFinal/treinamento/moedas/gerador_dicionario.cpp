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

#define TIPOS_MOEDAS 8
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
	
	
	
	
     
	
}
