#include <iostream>
#include <opencv2/opencv.hpp>
#include <cv.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  Mat image, mask;
  int width, height;
  int nobjects;
  
  CvPoint p;
  image = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  
  if(!image.data){
    cout << "imagem nao carregou corretamente\n";
    return(-1);
  }
  width=image.size().width;
  height=image.size().height;

  p.x=0;
  p.y=0;

  // Searching objects that are in the edges
  nobjects=0;
  for(int i=0; i<height; i++){
    for(int j=0; j<width; j++){
      if(image.at<uchar>(i,j) == 255){
		if(i==0 || i==height-1){
			
			p.x=j;
			p.y=i;		
			floodFill(image,p,0);
					
			

	  }
	}
  }
}
//Making bubbles in the edge disappear
  for(int i=0; i<height; i++){
    for(int j=0; j<width; j++){
      if(image.at<uchar>(i,j) == 255){
		if(j==0 || j==width-1){
			
			p.x=j;
			p.y=i;		
			floodFill(image,p,0);
					
			

	  }
	}
  }
}
//For the next trick I will change the color of the background so I can separate the objects
p.x=0;
p.y=0;
floodFill(image,p,100);
//Search for a bubble
for(int i=0; i<height; i++){
 for(int j=0; j<width; j++){
	if(image.at<uchar>(i,j) == 255){
		
			
			p.x=j;
			p.y=i;		
			floodFill(image,p,150);
					
			
		}
	  
   }
}
//Find a hole
for(int i=0; i<height; i++){
 for(int j=0; j<width; j++){
	if(image.at<uchar>(i,j) == 0){
		
			p.x=j;
			p.y=i;
			floodFill(image,p,100);
			nobjects++;
					
			
		}
	  
   }
}
  cout<<"The number of bubbles are:"<<nobjects;

  imshow("image", image);
  imwrite("labeling.png", image);

  waitKey();
  return 0;
}

