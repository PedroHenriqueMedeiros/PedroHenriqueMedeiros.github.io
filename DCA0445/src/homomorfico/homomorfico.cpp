#include <iostream>
#include <opencv2/opencv.hpp>

#define GAMA_MAX 100
#define C_MAX 100

using namespace cv;
using namespace std;

Mat complexImage;

int sliderGamaL = 2;
int sliderGamaH = 1;
int sliderD0 = 1;
int sliderC = 1;

float gamaH = 1;
float gamaL = 0.5;
float D0 = 0.2;
float C = 0.2;

int dft_M;
int dft_N;

/* Exibe o espectro da uma imagem complexa. */
void exibirEspectro(Mat& complexI)
{
    Mat espectro;
    Mat planos[2];
    
    deslocaDFT(complexI);
    
    split(complexI, planos);
    
    magnitude(planos[0], planos[1], espectro);
    Mat magI = planos[0];
    
    magI += Scalar::all(1);                    
    log(magI, magI);
    
    normalize(magI, magI, 0, 1, CV_MINMAX); 
                                            

    imshow("spectrum magnitude", magI);   

}

/* Troca os quadrantes da imagem da DFT. */
void deslocaDFT(Mat& image)
{
  Mat tmp, A, B, C, D;

  /* Se a imagem tiver tamanho impar, recorta a regiao para
   * evitar cópias de tamanho desigual. */
  image = image(Rect(0, 0, image.cols & -2, image.rows & -2));
  int cx = image.cols/2;
  int cy = image.rows/2;

  /* reorganiza os quadrantes da transformada
   * A B   ->  D C
   * C D       B A */
  A = image(Rect(0, 0, cx, cy));
  B = image(Rect(cx, 0, cx, cy));
  C = image(Rect(0, cy, cx, cy));
  D = image(Rect(cx, cy, cx, cy));

  /* A <-> D */
  A.copyTo(tmp);  D.copyTo(A);  tmp.copyTo(D);

  /* C <-> B */
  C.copyTo(tmp);  B.copyTo(C);  tmp.copyTo(B);
}


void aplicarFiltroHomomorfico()
{
    Mat filter, complexImageTmp, imagemFinal;
    Mat planos[2];
	
    /* Construção do filtro com base nos parâmetros. */
    
    Mat Du = Mat(complexImage.size(), CV_32FC1, Scalar(0));
    Mat Du2, expoente, resultExp, Hu;
    
    /* Definição de Du. */
    for(int u = 0; u < dft_M; u++)
    {
		for(int v = 0; v < dft_N; v++)
		{
			Du.at<float>(u, v) = sqrt(
                (u - dft_M/2.0)*(u - dft_M/2.0) + 
                (v - dft_N/2.0)*(v - dft_N/2.0)
            );
		}
	}
    
    /* Operações matemáticas para definidor o filtro homomórfico Hu. */
    
	multiply(Du, Du, Du2);
    Du2 = Du2 / (D0 * D0);
	expoente = -1.0 * C * Du2;
	exp(expoente, resultExp);
	
	Hu = (gamaH - gamaL) * (1 - resultExp) + gamaL;
    
    Mat comps[]= {Hu, Hu};
	merge(comps, 2, filter);
   
    /* Aplica o filtro frequencial. */
    mulSpectrums(complexImage, filter, complexImageTmp, 0);
    
    deslocaDFT(filter);
    exibirEspectro(filter);
    
    deslocaDFT(complexImageTmp);
 
	/* Calcula a DFT inversa. */
    idft(complexImageTmp, complexImageTmp);
    
    /* Separa os planos real (0) e complexo (1). */
    split(complexImageTmp, planos);
    
    /* Aqui é necessário normalizar os valores do plano real, pois eles ainda 
     * serão submetidos a uma operação de exponenciação e altos valores gerariam
     * uma falha de representação. */
    normalize(planos[0], planos[0], 0, 1, CV_MINMAX);	
    
    exp(planos[0], planos[0]);
 
    /* Normaliza novamente para poder exibir o resultado final. */
    normalize(planos[0], imagemFinal, 0, 1, CV_MINMAX);	
    
    imshow("resultado", imagemFinal);
}

void alterarSliderGamaH(int, void*)
{
    if(sliderGamaH < 1)
    {
        sliderGamaH = 1;
    }
    
    if(sliderGamaH <= sliderGamaL)
    {
        sliderGamaH = sliderGamaL + 1;
    }
    gamaH = (float) sliderGamaH/10.0;
    aplicarFiltroHomomorfico();  
}

void alterarSliderGamaL(int, void*)
{
	if(sliderGamaL >= sliderGamaH)
	{
		sliderGamaL = sliderGamaH - 1;
	}
    
	gamaL = (float) sliderGamaL/10.0;
	aplicarFiltroHomomorfico();
}

void alterarSliderD0(int, void*)
{
	if(sliderD0 == 0)
    {
        sliderD0 = 1;
    }
    
    D0 = (float) sliderD0/10.0;
    aplicarFiltroHomomorfico();
}

void alterarSliderC(int, void*)
{
	if(sliderC == 0)
    {
        sliderC = 1;
    }
    
    C = (float) sliderC/1000.0;
    aplicarFiltroHomomorfico();
}

int main(int argc, char* argv[])
{

    Mat  imagem, padded;

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
       
    
    dft_M = getOptimalDFTSize(imagem.rows);
    dft_N = getOptimalDFTSize(imagem.cols);

    /* Realiza o padding da imagem. */
    copyMakeBorder(imagem, padded, 0,
                 dft_M - imagem.rows, 0,
                 dft_N - imagem.cols,
                 BORDER_CONSTANT, Scalar::all(0));
	
	namedWindow("resultado", 1);

                 
    Mat planos[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    merge(planos, 2, complexImage);         // Add to the expanded another plane with zeros
    
    log(complexImage + 1, complexImage);
    
    /* Calcula o DFT. */
    dft(complexImage, complexImage);
    
    deslocaDFT(complexImage);
    
    /* Cria as barras de rolagem com todos os parâmetros. */
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
        dft_M,
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


