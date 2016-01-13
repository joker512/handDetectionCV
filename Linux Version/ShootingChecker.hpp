#include <opencv2/opencv.hpp>

#define DEFAULT_ROIS_SIZE 20

using namespace cv;
using namespace std;

class ShootingChecker{
public:
	ShootingChecker();
	ShootingChecker(const Size& imageSize);
	ShootingChecker(const vector<Point>& colorRoisPoints, const int colorRoisSize = DEFAULT_ROIS_SIZE);
	vector<Rect> getColorRoisRects() const;
	void learnColor(const Mat& image);
	Scalar getShootingDirection(const Mat& image) const;
	Scalar getShootingDirection(const Mat& image, Mat& binaryImage) const;

	vector<Point> colorRoisPoints;
	int colorRoisSize = DEFAULT_ROIS_SIZE;
	bool filterMuar = false;

private:
	static int C_LOWER[3];
	static int C_UPPER[3];
	const int COMP_PART_OF_BRECT_SIDE = 5;
	const int FINGERS_MAX_ANGLE = 95;
	vector<Vec3i> averageColors;

	void initColorRois(const Size& imageSize);
	Mat getBinary(const Mat& image) const;
	int getMedian(vector<int> v) const;
	void filterGarbage(Mat& imgBinary) const;
	void eleminateDefects(vector<Vec4i>& defects, const vector<Point> contour, const Rect& bRect) const;
	void removeRedundantEndPoints(vector<Point> contour, const vector<Vec4i> newDefects, const Rect& bRect) const;
	Point getFingerTip(vector<Point> contour, const vector<Point> hullPoints, const Rect& bRect) const;
	Point getDirectionPoint(const Point& fingerTip, vector<Point> contour, const Rect& bRect) const;
	float getAngle(const Point& s, const Point& f, const Point& e) const;
	float distanceP2P(const Point& a, const Point& b) const;
};
