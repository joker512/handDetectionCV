#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

class ImageReader{
public:
	ImageReader(int webCamera);
	ImageReader(const string& filename);
	bool read(Mat& src);

	Size roiSize;

private:
	VideoCapture cap;		
	bool isCamera;
};
