#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp> 

const double THRESHOLD = 400;

using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    Mat input = cv::imread("p1.jpg", 0); //Load as grayscale
    Mat image = cv::imread("t1.jpg", 0); //Load as grayscale
    
    equalizeHist(input, input);
    resize(input, input, Size(2000, 2000), 0, 0, INTER_LINEAR);
    
    equalizeHist(image, image);
    resize(image, image, Size(2000, 2000), 0, 0, INTER_LINEAR);
    
    SiftFeatureDetector detector(20000);
    SiftDescriptorExtractor extractor;
    BFMatcher matcher(NORM_L2, false);
    
    
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    
    detector.detect(input, keypoints1);
    detector.detect(image, keypoints2);
    
    
    Mat descriptors1, descriptors2;
    
    extractor.compute(input, keypoints1, descriptors1);
    extractor.compute(input, keypoints2, descriptors2);
    
    
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);


    // Add results to image and save.
    Mat output;
    drawKeypoints(input, keypoints1, output);
    imwrite("sift_result.jpg", output);
    
     //-- Draw matches
    Mat img_matches;
    drawMatches(input, keypoints1, image, keypoints2, matches, img_matches );
    cv::imwrite("sift_matches.jpg", img_matches);
    
    cout << "Quantidade de matches = " << matches.size() << endl;
    
    float distanciaAcumulada = 0;
    for(int i = 0; i < (int) matches.size(); i++)
    {
        distanciaAcumulada += matches[i].distance;
    }
    cout << "Distância acumulada = " << distanciaAcumulada << endl;
    
    FileStorage fsWrite("1real-face.descriptors", FileStorage::WRITE);
    fsWrite << "descriptors" << descriptors1;
    fsWrite.release();
    
    // Lendo os descriptors e testando novamente.
    Mat descriptors3;
    
    FileStorage fsRead("1real-face.descriptors", FileStorage::READ);
    fsRead["descriptors"] >>  descriptors3;
    
    
    vector<DMatch> matches2;
    matcher.match(descriptors3, descriptors2, matches2);
    
    cout << "Quantidade de matches = " << matches2.size() << endl;
    
    float distanciaAcumulada2 = 0;
    for(int i = 0; i < (int) matches2.size(); i++)
    {
        distanciaAcumulada2 += matches2[i].distance;
    }
    cout << "Distância acumulada = " << distanciaAcumulada2 << endl;
    
    


    return 0;
}
