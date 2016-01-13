#include <opencv2/opencv.hpp>
#include "ShootingChecker.hpp"
#include "ImageReader.hpp"

using namespace cv;
using namespace std;

void drawBinary(Mat& image, Mat& binaryImage){
	pyrDown(binaryImage, binaryImage);
	pyrDown(binaryImage, binaryImage);

	Rect roi(Point(3 * image.cols / 4, 0), binaryImage.size());
	vector<Mat> channels;
	Mat coloredBinary;
	for(int i = 0; i < 3; ++i) {
		channels.push_back(binaryImage);
	}
	merge(channels, coloredBinary);
	coloredBinary.copyTo(image(roi));
}

// TODO: draw contours for debugging and improvement
// void drawContours(Mat& image, Scalar v){
// 	// drawContours(image,hg->hullP,hg->cIdx,cv::Scalar(200,0,0),2, 8, vector<Vec4i>(), 0, Point());
// 	// rectangle(image,hg->bRect.tl(),hg->bRect.br(),Scalar(0,0,200));
// 	vector<Vec4i>::iterator d=hg->defects[hg->cIdx].begin();
// 	int fontFace = FONT_HERSHEY_PLAIN;

// 	vector<Mat> channels;
// 		Mat result;
// 		for(int i=0;i<3;i++)
// 			channels.push_back(binaryImage);
// 		merge(channels,result);
// 		drawContours(result,hg->hullP,hg->cIdx,cv::Scalar(0,0,250),10, 8, vector<Vec4i>(), 0, Point());

// 	while( d!=hg->defects[hg->cIdx].end() ) {
//    	    Vec4i& v=(*d);
// 	    int startidx=v[0]; Point ptStart(hg->contours[hg->cIdx][startidx] );
//    		int endidx=v[1]; Point ptEnd(hg->contours[hg->cIdx][endidx] );
//   	    int faridx=v[2]; Point ptFar(hg->contours[hg->cIdx][faridx] );
// 	    float depth = v[3] / 256;
//    		circle( result, ptFar,   9, Scalar(0,205,0), 5 );
// 		d++;
// 	}
// }

void drawArrow(Mat& img, const Scalar& direction, const Scalar& color, const float scale = 0.2) {
    Point p = Point(direction[0], direction[1]);
    Point q = Point(direction[2], direction[3]);
    double angle;
    double hypotenuse;
    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, color, 2, CV_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, color, 2, CV_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, color, 2, CV_AA);
}

int main(int argc, char** argv){
	try {
		ImageReader* reader = argc == 2 ? new ImageReader(argv[1]) : new ImageReader(0);
		ShootingChecker shootingChecker(reader->roiSize);
		namedWindow("Learn Color", CV_WINDOW_AUTOSIZE);

		Mat learnColorImage;
		while(true) {
			reader->read(learnColorImage);
			for(Rect roi : shootingChecker.getColorRoisRects()) {
				rectangle(learnColorImage, roi, Scalar(0, 255, 0), 2);
			}

			imshow("Learn Color", learnColorImage);

			if (waitKey(30) == char('q')) {
				shootingChecker.learnColor(learnColorImage);
				break;
			}
		}
		destroyWindow("Learn Color");

		namedWindow("Checking Direction");
		Mat image;
		Mat binaryImage;
		while(true){
			reader->read(image);

			Scalar shootingDirection = shootingChecker.getShootingDirection(image, binaryImage);
			if (shootingDirection != Scalar()) {
				drawArrow(image, shootingDirection, Scalar(0, 255, 0), 1);
			}

			// makeContours(m, &hg);
			drawBinary(image, binaryImage);
			imshow("Checking Direction", image);

			if(waitKey(30) == char('q')) {
				break;
			}
		}

		destroyAllWindows();
		delete reader;
	}
	catch (logic_error e) {
		cerr << "Exception: " << e.what() << endl;
	}

	return 0;
}
