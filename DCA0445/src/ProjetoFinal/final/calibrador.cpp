#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>          
    
using namespace cv;
using namespace std;

#define TIPOS_MOEDAS 3 
#define NUM_AMOSTRAS 20 

// Parâmetros da MLP

#define MLP_MAX_ITER 10000
#define EPSILON 1e-8
#define LEARNING_RATE 0.1
#define MOMENTUM 0.1

// Parâmetros do histograma
#define NUM_NIVEIS_MATIZ 32
#define HIST_UNIFORME true
#define HIST_ACUMULADO false

// Momentos invariantes
#define NUM_MOMENTOS 7

/* Realiza o treinamento da rede neural artificial. */
void treinar(const Mat &entradas, const Mat& saidas) {
	// Definição das camadas.
    Mat camadas = cv::Mat(4, 1, CV_32SC1);
    camadas.row(0) = Scalar(entradas.cols);
    camadas.row(1) = Scalar(32);
    camadas.row(2) = Scalar(32);
    camadas.row(3) = Scalar(saidas.cols);

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
    
    cout << "[treinar] Treinamento concluído após " << numIteracoes << " iterações." << endl;
    if(numIteracoes == MLP_MAX_ITER)
    {
        cout << "[treinar] Porém, o erro mínimo de " << EPSILON << " não foi alcançado." << endl;
    }

    // Salva o resultado em um arquivo XML.
    FileStorage fs("mlp.yml", FileStorage::WRITE); // or xml
	mlp.write(*fs, "mlp"); 
    cout << "[treinar] Pesos obtidos após o treinamento armazenados em 'mlp.yml'." << endl;

    

}

/* Faz a equalização do histograma de uma imagem colorida. */
void equalizarHistograma(Mat& imagem)
{
    if(imagem.channels() >= 3)
    {
        Mat ycrcb;
        cvtColor(imagem, ycrcb, CV_BGR2YCrCb);
        vector<Mat> canais;
        split(ycrcb, canais);
        equalizeHist(canais[0], canais[0]);
        merge(canais, ycrcb);
        cvtColor(ycrcb, imagem, CV_YCrCb2BGR);
    }
}


int main()
{    
    // Definindo conjunto treinamento.
    Mat entradas(TIPOS_MOEDAS * NUM_AMOSTRAS, NUM_NIVEIS_MATIZ + 2 + NUM_MOMENTOS, CV_32FC1);
    Mat saidas = -1*Mat::ones(TIPOS_MOEDAS * NUM_AMOSTRAS, TIPOS_MOEDAS, CV_32FC1);
    
    // Matrizes onde serão carregadas as imagens do conjunto treinamento.
	vector<Mat> moedas25(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas50(NUM_AMOSTRAS, Mat());
	vector<Mat> moedas100(NUM_AMOSTRAS, Mat());
    
    // Matrizes com os descritores para as imagens.
    // Descritor = histograma + largura + altura + momentos invariantes
	vector<Mat> descMoedas25(NUM_AMOSTRAS, Mat());
	vector<Mat> descMoedas50(NUM_AMOSTRAS, Mat());
	vector<Mat> descMoedas100(NUM_AMOSTRAS, Mat());
    
    // Inicializa os vetores.
    for(int i = 0; i < NUM_AMOSTRAS; i++)
    {
        descMoedas25[i] = Mat(1, NUM_NIVEIS_MATIZ + 2 + NUM_MOMENTOS, CV_32FC1);
        descMoedas50[i] = Mat(1, NUM_NIVEIS_MATIZ + 2 + NUM_MOMENTOS, CV_32FC1);
        descMoedas100[i] = Mat(1, NUM_NIVEIS_MATIZ + 2 + NUM_MOMENTOS, CV_32FC1);
    }
    
	// Faz a leitura das imagens do conjunto treinamento e prepara seus descritores.
    cout << "[main] Carregando imagens do conjunto treinamento e extraindo seus descritores." << endl;
	for(int i = 0; i < NUM_AMOSTRAS; i++)
	{
        
        // Abre as imagens.
		moedas25[i] = imread("treinamento/25/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas50[i] = imread("treinamento/50/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
		moedas100[i] = imread("treinamento/100/" + to_string(i+1) + ".jpg", CV_LOAD_IMAGE_COLOR);
        
        // Equaliza o histograma das imagens.
        equalizarHistograma(moedas25[i]);
        equalizarHistograma(moedas50[i]);
        equalizarHistograma(moedas100[i]);
        
        Mat moeda25cinza;
        Mat moeda50cinza;
        Mat moeda100cinza;
        
        // Calcula os momentos invariantes das moedas.
        cvtColor(moedas25[i], moeda25cinza, CV_BGR2GRAY);
        cvtColor(moedas50[i], moeda50cinza, CV_BGR2GRAY);
        cvtColor(moedas100[i], moeda100cinza, CV_BGR2GRAY);
        
        Moments moments25 = moments(moeda25cinza, false);
        Moments moments50 = moments(moeda50cinza, false);
        Moments moments100 = moments(moeda100cinza, false);
        
        double hu25[7], hu50[7], hu100[7];
        
        HuMoments(moments25, hu25);
        HuMoments(moments50, hu50);
        HuMoments(moments100, hu100);

        // Converte as moedas para HSV, para trabalhar apenas com suas matizes.
		cvtColor(moedas25[i], moedas25[i], CV_BGR2HSV);
		cvtColor(moedas50[i], moedas50[i], CV_BGR2HSV);
		cvtColor(moedas100[i], moedas100[i], CV_BGR2HSV);
        
        // Calcula o histograma das matizes das imagens.
		int histSize[] = {NUM_NIVEIS_MATIZ};
		float hranges[] = {0, 180}; // hue varies from 0 to 179, see cvtColor
		const float* ranges[] = {hranges};
		int channels[] = {0};
		
		Mat hist25, hist50, hist100;
		
		calcHist(&moedas25[i], 1, channels, Mat(), hist25, 1, histSize, ranges, HIST_UNIFORME, HIST_ACUMULADO);
		calcHist(&moedas50[i], 1, channels, Mat(), hist50, 1, histSize, ranges, HIST_UNIFORME, HIST_ACUMULADO);
		calcHist(&moedas100[i], 1, channels, Mat(), hist100, 1, histSize, ranges, HIST_UNIFORME, HIST_ACUMULADO);
        
        // Armazena os descritores calculados acima.
		for(int j = 0; j < NUM_NIVEIS_MATIZ; j++)
		{
			descMoedas25[i].at<float>(0,j) = hist25.at<float>(j,0);
			descMoedas50[i].at<float>(0,j) = hist50.at<float>(j,0);
			descMoedas100[i].at<float>(0,j) = hist100.at<float>(j,0);
		}
		
		descMoedas25[i].at<float>(0,NUM_NIVEIS_MATIZ) = moedas25[i].rows;
		descMoedas25[i].at<float>(0,NUM_NIVEIS_MATIZ+1) = moedas25[i].cols;
		descMoedas50[i].at<float>(0,NUM_NIVEIS_MATIZ) = moedas50[i].rows;
		descMoedas50[i].at<float>(0,NUM_NIVEIS_MATIZ+1) = moedas50[i].cols;		
		descMoedas100[i].at<float>(0,NUM_NIVEIS_MATIZ) = moedas100[i].rows;
		descMoedas100[i].at<float>(0,NUM_NIVEIS_MATIZ+1) = moedas100[i].cols;
        
        
        for(int j = NUM_NIVEIS_MATIZ + 2; j < NUM_NIVEIS_MATIZ + 2 + NUM_MOMENTOS; j++)
        {
            descMoedas25[i].at<float>(0,j) = hu25[j-NUM_NIVEIS_MATIZ - 2];
			descMoedas50[i].at<float>(0,j) = hu50[j-NUM_NIVEIS_MATIZ - 2];
			descMoedas100[i].at<float>(0,j) = hu100[j-NUM_NIVEIS_MATIZ - 2];
        }
	}	 
    
    // Prepara a entrada e saída para a rede neural artificial.
    cout << "[main] Preparando entradas e saídas da RNA." << endl;
    for(int i = 0; i < TIPOS_MOEDAS * NUM_AMOSTRAS; i += TIPOS_MOEDAS)
    {
		int k = i/TIPOS_MOEDAS;
		
        for(int j = 0; j < NUM_NIVEIS_MATIZ + 2 + NUM_MOMENTOS; j++)
        {
            entradas.at<float>(i,j) = descMoedas25[k].at<float>(0, j);
            entradas.at<float>(i+1,j) = descMoedas50[k].at<float>(0, j);
            entradas.at<float>(i+2,j) = descMoedas100[k].at<float>(0, j);
        }
        
        saidas.at<float>(i,0) = 1;
        saidas.at<float>(i+1,1) = 1;
        saidas.at<float>(i+2,2) = 1;
    }

    // Realiza o treinamento da RNA (o resultado é salvo em "mlp.yml".
    cout << "[main] Iniciando treinamento da RNA..." << endl;
    treinar(entradas, saidas);
    
    /*
    // Exibe o resultado para todas as entradas.
	Mat saidasTeste(TIPOS_MOEDAS * NUM_AMOSTRAS, TIPOS_MOEDAS, CV_32FC1);
	
	CvANN_MLP mlp;
	mlp.load("mlp.yml", "mlp");
	
	mlp.predict(entradas, saidasTeste);

	for(int i = 0; i < saidasTeste.rows; i++)
	{  
        cout <<  "----------------------------------------" << endl;
        cout << "Moeda " << i+1 << ":" << endl;
        cout << "(25) = " << saidasTeste.at<float>(i, 0) << endl;
        cout << "(50) = " << saidasTeste.at<float>(i, 1) << endl;
        cout << "(100) = " << saidasTeste.at<float>(i, 2) << endl;
        cout <<  "----------------------------------------" << endl;
    }
    */
    
    return 0;
	
}
