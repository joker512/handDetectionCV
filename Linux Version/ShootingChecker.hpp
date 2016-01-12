// #include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>

#define DEFAULT_ROIS_SIZE 20

using namespace cv;
using namespace std;

class ShootingChecker{
public:
	ShootingChecker(const Size& imageSize);
	ShootingChecker(const vector<Point>& colorRoisPoints, const int colorRoisSize = DEFAULT_ROIS_SIZE);
	void learnColor(Mat& image);
	Scalar getShootingDirection(const Mat& image);
	Scalar getShootingDirection(const Mat& image, Mat& binaryImage);

	vector<Point> colorRoisPoints;
	int colorRoisSize;
	bool filterMuar = false;
	
private:
	static int cLower[3];
	static int cUpper[3];
	vector<Vec3i> averageColors;

	Mat getBinary(const Mat& image);
	int getMedian(vector<int> v);
	void filterGarbage(Mat& imgBinary);
	void eleminateDefects(vector<Vec4i>& defects, const vector<Point> contour, const Rect& bRect);
	void removeRedundantEndPoints(vector<Point> contour, vector<Vec4i>& newDefects, Rect bRect);
	Point getFingerTip(vector<Point> contour, const vector<Point> hullPoints, const Rect& bRect);
	Point getDirectionPoint(const Point& fingerTip, vector<Point> contour, const Rect& bRect);
	float getAngle(const Point& s, const Point& f, const Point& e);
	float distanceP2P(const Point& a, const Point& b);
};
