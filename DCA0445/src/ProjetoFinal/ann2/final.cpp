#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>          
    
using namespace cv;
using namespace std;

#define TIPOS_MOEDAS 3 // 25, 50 e 1 real.
#define NUM_AMOSTRAS 24 // 24 fotos de cada moeda.

// Parâmetros da MLP

#define MLP_MAX_ITER 10000
#define EPSILON 1e-10
#define LEARNING_RATE 0.1
#define MOMENTUM 0.1

// Parâmetros do Kmeans
#define NUM_CLUSTERS 16
#define KMEANS_MAX_ITER 1000
#define KMEANS_MAX_ERRO 0.0001
#define TENTATIVAS 5

// Parâmetros do histograma
#define NUM_NIVEIS_MATIZ 128
#define NUM_NIVEIS_SATURACAO 128
#define HIST_UNIFORME true
#define HIST_ACUMULADO false


vector<float> mat2vec(const Mat &mat)
{
    vector<float> array;
    if (mat.isContinuous()) {
      array.assign(mat.datastart, mat.dataend);
    } else {
      for (int i = 0; i < mat.rows; ++i) {
        array.insert(array.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i)+mat.cols);
      }
    }
    
    return array;
}


void treinar(const Mat &entradas, const Mat& saidas) {

	
	// Definição das camadas.
    Mat camadas = cv::Mat(3, 1, CV_32SC1);
    camadas.row(0) = Scalar(entradas.cols);
    camadas.row(1) = Scalar(32);
    camadas.row(2) = Scalar(saidas.cols);

	// Definição dos parâmetros;
    CvANN_MLP_TrainParams params;
    CvTermCriteria criteria;
    criteria.max_iter = MLP_MAX_ITER;
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

void reduzirCores(Mat &imagem)
{
	Mat amostras(imagem.rows * imagem.cols, 3, CV_32F);
	
	for(int y = 0; y < imagem.rows; y++)
	{
		for( int x = 0; x < imagem.cols; x++)
		{
			for( int z = 0; z < 3; z++)
			{
				amostras.at<float>(y + x*imagem.rows, z) = imagem.at<Vec3b>(y,x)[z];
			}
		}
	}
	
	Mat rotulos, centros;
	kmeans(amostras, NUM_CLUSTERS, rotulos, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, KMEANS_MAX_ITER, KMEANS_MAX_ERRO), TENTATIVAS, KMEANS_PP_CENTERS, centros);

	//Mat saida(imagem.size(), imagem.type());
	for( int y = 0; y < imagem.rows; y++ )
	{
		for( int x = 0; x < imagem.cols; x++ )
		{ 
			int cluster_idx = rotulos.at<int>(y + x*imagem.rows,0);
			imagem.at<Vec3b>(y,x)[0] = centros.at<float>(cluster_idx, 0);
			imagem.at<Vec3b>(y,x)[1] = centros.at<float>(cluster_idx, 1);
			imagem.at<Vec3b>(y,x)[2] = centros.at<float>(cluster_idx, 2);
		}
	}
	imshow( "clustered image", imagem);
	waitKey( 0 );

	
}

int main()
{
	initModule_nonfree();
    
    // Definindo conjunto treinamento.
    Mat entradas(TIPOS_MOEDAS * NUM_AMOSTRAS, 130, CV_32FC1);
    Mat saidas = Mat::zeros(TIPOS_MOEDAS * NUM_AMOSTRAS, TIPOS_MOEDAS, CV_32FC1);
    
	vector<Mat> moedas25(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas50(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas100(NUM_AMOSTRAS, Mat());

	vector<Mat> descMoedas25(NUM_AMOSTRAS, Mat(1, 128, CV_32FC1));
	vector<Mat> descMoedas50(NUM_AMOSTRAS, Mat(1, 128, CV_32FC1));
	vector<Mat> descMoedas100(NUM_AMOSTRAS, Mat(1, 128, CV_32FC1));
	
	/* Lê todas as amostras. */
	for(int i = 0; i < NUM_AMOSTRAS; i++)
	{
		moedas25[i] = imread("25/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas50[i] = imread("50/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas100[i] = imread("100/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);

		//reduzirCores(moedas100[i]);
		
		cvtColor(moedas25[i], moedas25[i], CV_BGR2HSV);
		cvtColor(moedas50[i], moedas50[i], CV_BGR2HSV);
		cvtColor(moedas100[i], moedas100[i], CV_BGR2HSV);
	
		/*
		// Quantize the hue to 30 levels
		// and the saturation to 32 levels
		int hbins = 30, sbins = 32;
		int histSize[] = {hbins, sbins};
		
		// hue varies from 0 to 179, see cvtColor
		float hranges[] = { 0, 180 };
		// saturation varies from 0 (black-gray-white) to
		// 255 (pure spectrum color)
		float sranges[] = { 0, 256 };
		const float* ranges[] = { hranges, sranges };
		MatND hist;
		vector<uchar> histograma;
		
		// we compute the histogram from the 0-th and 1-st channels
		int channels[] = {0, 1};

		calcHist(&moedas100[i], 1, channels, Mat(), 
		hist, 2, histSize, ranges,
		true, // the histogram is uniform
		false );
		double maxVal =0;
		minMaxLoc(hist, 0, &maxVal, 0, 0);
		* 
		*/
		
		int histSize[] = {NUM_NIVEIS_MATIZ};
		float hranges[] = {0, 180}; // hue varies from 0 to 179, see cvtColor
		const float* ranges[] = {hranges};
		int channels[] = {0};
		
		Mat hist25, hist50, hist100;
		
		calcHist(&moedas25[i], 1, channels, Mat(), hist25, 1, histSize, ranges, HIST_UNIFORME, HIST_ACUMULADO);
		calcHist(&moedas50[i], 1, channels, Mat(), hist50, 1, histSize, ranges, HIST_UNIFORME, HIST_ACUMULADO);
		calcHist(&moedas100[i], 1, channels, Mat(), hist100, 1, histSize, ranges, HIST_UNIFORME, HIST_ACUMULADO);
		
		
		for(int j = 0; j < 128; j++)
		{
			descMoedas25[i].at<float>(0,j) = hist25.at<float>(j,0);
			descMoedas50[i].at<float>(0,j) = hist50.at<float>(j,0);
			descMoedas100[i].at<float>(0,j) = hist100.at<float>(j,0);
		}
		
		descMoedas25[i].at<float>(0,128) = moedas25[i].rows;
		descMoedas25[i].at<float>(0,129) = moedas25[i].cols;
		descMoedas50[i].at<float>(0,128) = moedas50[i].rows;
		descMoedas50[i].at<float>(0,129) = moedas50[i].cols;		
		descMoedas100[i].at<float>(0,128) = moedas100[i].rows;
		descMoedas100[i].at<float>(0,129) = moedas100[i].cols;


	}	    
    
    for(int i = 0; i < TIPOS_MOEDAS * NUM_AMOSTRAS; i += TIPOS_MOEDAS)
    {
		int k = i/TIPOS_MOEDAS;
		
        for(int j = 0; j < 130; j++)
        {
            entradas.at<float>(i,j) = descMoedas25[k].at<float>(0, j);
            entradas.at<float>(i+1,j) = descMoedas50[k].at<float>(0, j);
            entradas.at<float>(i+2,j) = descMoedas100[k].at<float>(0, j);
        }
        
        saidas.at<float>(i,0) = 1;
        saidas.at<float>(i+1,1) = 1;
        saidas.at<float>(i+2,2) = 1;
    }

    
    treinar(entradas, saidas);
    
    /*
	Mat saidasMLP(TIPOS_MOEDAS * NUM_AMOSTRAS, TIPOS_MOEDAS, CV_32FC1);
	
	CvANN_MLP mlp;
	mlp.load("mlp.yml", "mlp");
	
	mlp.predict(entradas, saidasMLP);

	for(int i = 0; i < saidasMLP.rows; i++)
	{
        for(int j = 0; j < TIPOS_MOEDAS; j++)
        {
            cout << saidasMLP.at<float>(i, j) << endl;
        }
        cout <<  "----------------------------------------" << endl;
    }
	*/
 

		
}
