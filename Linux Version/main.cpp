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
                        if (!reader->read(learnColorImage))
				break;

			for(Rect roi : shootingChecker.getColorRoisRects()) {
				rectangle(learnColorImage, roi, Scalar(0, 255, 0), 2);
			}

			imshow("Learn Color", learnColorImage);

			if (waitKey(30) > char(0)) {
				shootingChecker.learnColor(learnColorImage);
				break;
			}
		}
		destroyWindow("Learn Color");

		namedWindow("Checking Direction");
		Mat image;
		Mat binaryImage;
		while(true) {
			if (!reader->read(image))
				break;

			Scalar shootingDirection = shootingChecker.getShootingDirection(image, binaryImage);
			if (shootingDirection != Scalar()) {
				drawArrow(image, shootingDirection, Scalar(0, 255, 0), 1);
			}
			drawBinary(image, binaryImage);
			imshow("Checking Direction", image);

                        if (waitKey(30) > char(0)) {
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
