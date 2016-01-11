#include "myImage.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

using namespace cv;

MyImage::MyImage(){
}

MyImage::MyImage(int webCamera){
	roiSize = Size(250, 350);
	cameraSrc=webCamera;
	cap=VideoCapture(webCamera);
}

MyImage::MyImage(const string& filename){
	roiSize = Size(750, 900);
	cameraSrc = -1;
	cap=VideoCapture(filename);
}

MyImage::~MyImage(){
	cap.release();
}

void MyImage::read(){
	Mat tmp;
	cap >> tmp;
	if (cameraSrc == -1){
		transpose(tmp, tmp);
	}
	flip(tmp, tmp, 1);
	tmp(Rect(0, tmp.rows - roiSize.height, roiSize.width, roiSize.height)).copyTo(src);
}
