#include "ImageReader.hpp"

using namespace cv;

ImageReader::ImageReader(int webCamera){
	isCamera = true;
	roiSize = Size(250, 350);
	cap = VideoCapture(webCamera);
}

ImageReader::ImageReader(const string& filename){
	isCamera = false;
	roiSize = Size(750, 900);
	if (ifstream(filename))
		cap = VideoCapture(filename);
	else
		throw invalid_argument("File " + filename + " doesn't exist");
}

void ImageReader::read(Mat& src){
	Mat tmp;
	cap >> tmp;
	if (!isCamera){
		transpose(tmp, tmp);
	}
	flip(tmp, tmp, 1);
	tmp(Rect(0, tmp.rows - roiSize.height, roiSize.width, roiSize.height)).copyTo(src);
}
