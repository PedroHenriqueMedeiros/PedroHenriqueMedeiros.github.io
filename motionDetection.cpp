#include <iostream>
#include <opencv2/opencv.hpp>
#include  <time.h>  
#include <math.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  double t;
  while(1){
   if(waitKey(30) >= 0) break;
   t = (double)getTickCount();
// do something ...
   t = ((double)getTickCount() - t)/getTickFrequency();
   cout<<t<<"\n";
   }
  return 0;
}

