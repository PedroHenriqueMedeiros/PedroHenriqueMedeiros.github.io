#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv){
  Mat image;
  image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  imshow("image", image);
  waitKey();
  return 0;
}
