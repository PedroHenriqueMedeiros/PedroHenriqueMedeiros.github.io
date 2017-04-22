#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#define GAMA_MAX 100
#define C_MAX 100

using namespace cv;
using namespace std;

Mat imagem, padded, filter, tmp, complexImage, complexImageTmp, imagemFinal;
vector<Mat> planos;
Mat_<float> realInput, zeros;

int sliderGamaL;
int sliderGamaH ;
int sliderD0;
int sliderC;

int dft_M;
int dft_N;
int altura;
int largura; 

/* Troca os quadrantes da imagem da DFT. */
void deslocaDFT(Mat& image)
{
  Mat tmp, A, B, C, D;

  // se a imagem tiver tamanho impar, recorta a regiao para
  // evitar cópias de tamanho desigual
  image = image(Rect(0, 0, image.cols & -2, image.rows & -2));
  int cx = image.cols/2;
  int cy = image.rows/2;

  // reorganiza os quadrantes da transformada
  // A B   ->  D C
  // C D       B A
  A = image(Rect(0, 0, cx, cy));
  B = image(Rect(cx, 0, cx, cy));
  C = image(Rect(0, cy, cx, cy));
  D = image(Rect(cx, cy, cx, cy));

  // A <-> D
  A.copyTo(tmp);  D.copyTo(A);  tmp.copyTo(D);

  // C <-> B
  C.copyTo(tmp);  B.copyTo(C);  tmp.copyTo(B);
}


void aplicarFiltroHomomorfico()
{
	cout << "Filtro executado!" << endl;
	
    // Construção do filtro com base nos parâmetros.
    Mat Du2, expoente, resultExp, Hu;
    Mat Du = Mat(padded.size(), CV_32FC1, Scalar(0));
     
    for(int i = 0; i < dft_M; i++)
    {
		for(int j = 0; j < dft_N; j++)
		{
			Du.at<float>(i,j) = sqrt(pow((i - dft_M/2.0),2) + pow((j - dft_N/2.0),2)) / sliderD0;
		}
	}
	
	multiply(Du, Du, Du2);
	expoente = -sliderC * Du2;
	exp(expoente, resultExp);
	
	Hu = (sliderGamaH - sliderGamaL) * (1 - resultExp) + sliderGamaL;
    
    Mat comps[]= {Hu, Hu};
	merge(comps, 2, filter);
    
    cout << complexImage.rows << ", " << complexImage.cols << endl;
    cout << filter.rows << ", " << filter.cols << endl;
    
    cout << complexImage.type() << endl;
    cout << filter.type() << endl;
   
    // aplica o filtro frequencial
    mulSpectrums(complexImage, filter, complexImageTmp, 0);
 
	// calcula a DFT inversa
    idft(complexImageTmp, complexImageTmp);

	// limpa o array de planos
    planos.clear();

    // separa as partes real e imaginaria da
    // imagem filtrada
    split(complexImageTmp, planos);
    
    // normaliza a parte real para exibicao
    normalize(planos[0], imagemFinal, 0, 1, CV_MINMAX);	
	
	/*
	// realiza o padding da imagem
	copyMakeBorder(imagem, padded, 0,
                 dft_M - altura, 0,
                 dft_N - largura,
                 BORDER_CONSTANT, Scalar::all(0));
                 
     /I
                 
	// parte imaginaria da matriz complexa (preenchida com zeros)
	zeros = Mat_<float>::zeros(padded.size());

	// prepara a matriz complexa para ser preenchida
	complexImage = Mat(padded.size(), CV_32FC2, Scalar(0));

	// a função de transferência (filtro frequencial) deve ter o
	// mesmo tamanho e tipo da matriz complexa
	filter = complexImage.clone();

	// cria uma matriz temporária para criar as componentes real
	// e imaginaria do filtro ideal
	tmp = Mat(dft_M, dft_N, CV_32F);
	*/
    
}



void alterarSliderGamaH(int, void*)
{
	imshow("resultado", imagemFinal);
	aplicarFiltroHomomorfico();
}

void alterarSliderGamaL(int, void*)
{
	if(sliderGamaL > sliderGamaH)
	{
		sliderGamaL = sliderGamaH;
	}
	
	aplicarFiltroHomomorfico();
	imshow("resultado", imagemFinal);
}

void alterarSliderD0(int, void*)
{
	if(sliderD0 == 0)
    {
        sliderD0 = 1;
    }
    
    aplicarFiltroHomomorfico();
	imshow("resultado", imagemFinal);
}

void alterarSliderC(int, void*)
{
	if(sliderC == 0)
    {
        sliderC = 1;
    }
    aplicarFiltroHomomorfico();
	imshow("resultado", imagemFinal);
}

int main(int argc, char* argv[]){

    /* Verifica o número de argumentos.  */
    if (argc != 2) 
    {
        cout << "A lista de argumentos deve ser: "
        "./homomorfico <imagem>" << endl;
        return -1;
    }
    
    /* Checa se a imagem pode ser aberta. */
    imagem = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    if (!imagem.data) 
    {
        cout << "A imagem não pode ser aberta." << endl;
        return -2;
    }
    
    imagemFinal = imagem.clone();
    
    largura = imagem.cols;
    altura = imagem.rows;
    
    dft_M = getOptimalDFTSize(altura);
    dft_N = getOptimalDFTSize(largura);
	
	namedWindow("resultado", 1);
	
	  // realiza o padding da imagem
	  copyMakeBorder(imagem, padded, 0,
					 dft_M - altura, 0,
					 dft_N - largura,
					 BORDER_CONSTANT, Scalar::all(0));
					 
	  // parte imaginaria da matriz complexa (preenchida com zeros)
	  zeros = Mat_<float>::zeros(padded.size());

	  // prepara a matriz complexa para ser preenchida
	  complexImage = Mat(padded.size(), CV_32FC2, Scalar(0));
	  
	// cria a compoente real
    realInput = Mat_<float>(padded);

    
    // insere as duas componentes no array de matrizes
    planos.push_back(realInput);
    planos.push_back(zeros);
    
    merge(planos, complexImage);
    
    // calcula o dft
    dft(complexImage, complexImage);

    // realiza a troca de quadrantes
    deslocaDFT(complexImage);
    
    /* Cria as barras de rolagem. */
    createTrackbar("GamaH", "resultado",
        &sliderGamaH,
        GAMA_MAX,
        alterarSliderGamaH);  
    alterarSliderGamaH(sliderGamaH, 0);
    
    createTrackbar("GamaL", "resultado",
        &sliderGamaL,
        GAMA_MAX,
        alterarSliderGamaL);  
    alterarSliderGamaH(sliderGamaL, 0);
    
    createTrackbar("D0", "resultado",
        &sliderD0,
        altura,
        alterarSliderD0);  
    alterarSliderD0(sliderD0, 0);
    
	createTrackbar("C", "resultado",
        &sliderD0,
        C_MAX,
        alterarSliderC);  
    alterarSliderC(sliderC, 0);

    /* Fecha o programa quando o usuário digita ESC. */
    while(1)
    {
      if(waitKey(30) >= 0) 
      {
        break;
      } 
    }

  return 0;
}
