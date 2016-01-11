#ifndef _MYIMAGE_
#define _MYIMAGE_ 

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class MyImage{
	public:
		MyImage(int webCamera);
		MyImage(const string& filename);
		MyImage();
	        ~MyImage();
		Mat srcLR;
		Mat src;
		Mat bw;
		vector<Mat> bwList;
		VideoCapture cap;		
		int cameraSrc; 
	        Size roiSize;
		void initWebCamera(int i);
	        void read();
};



#endif
