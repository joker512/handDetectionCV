#include "ShootingChecker.hpp"

using namespace cv;
using namespace std;

int ShootingChecker::C_LOWER[] = {12, 30, 80};
int ShootingChecker::C_UPPER[] = {7, 40, 80};

ShootingChecker::ShootingChecker() {
}

ShootingChecker::ShootingChecker(const Size& imageSize) {
	initColorRois(imageSize);
}

ShootingChecker::ShootingChecker(const vector<Point>& colorRoisPoints, const int colorRoisSize) {
	this->colorRoisPoints = colorRoisPoints;
	this->colorRoisSize = colorRoisSize;
}

void ShootingChecker::initColorRois(const Size& imageSize) {
  	vector<Point> rois = colorRoisPoints;
	int width = imageSize.width;
	int height = imageSize.height;

	rois.push_back(Point(width / 3, height / 1.75));
	rois.push_back(Point(width / 7, height / 1.75));
	rois.push_back(Point(width / 2.5, height / 1.35));
	rois.push_back(Point(width / 4, height / 1.35));
	rois.push_back(Point(width / 8, height / 1.35));
	rois.push_back(Point(width / 3, height / 1.1));
	rois.push_back(Point(width / 7, height / 1.1));

	colorRoisPoints.swap(rois);
}

vector<Rect> ShootingChecker::getColorRoisRects() const {
	vector<Rect> rois;
	for (auto p : colorRoisPoints) {
		rois.push_back(Rect(p, Size(colorRoisSize, colorRoisSize)));
	}
	return rois;
}

void ShootingChecker::learnColor(const Mat& image) {
	if (colorRoisPoints.empty()) {
		initColorRois(image.size());
	}
	averageColors.clear();

	Mat hlsImage;
	cvtColor(image, hlsImage, CV_BGR2HLS);
	for(auto roiRect : getColorRoisRects()) {
		Mat roi;
		hlsImage(roiRect).copyTo(roi);
		vector<int> h, s, l;

		int channels = roi.channels();
		for(int i = 2; i < roi.rows - 2; i++){
			for(int j = 2; j < roi.cols - 2; j++){
				h.push_back(roi.data[channels * (roi.cols * i + j) + 0]);
				s.push_back(roi.data[channels * (roi.cols * i + j) + 1]);
				l.push_back(roi.data[channels * (roi.cols * i + j) + 2]);
			}
		}
		averageColors.push_back(Point3i(getMedian(h), getMedian(s), getMedian(l)));
	}
}

Scalar ShootingChecker::getShootingDirection(const Mat& image) const {
	Mat binaryImage;
	return getShootingDirection(image, binaryImage);
}

Scalar ShootingChecker::getShootingDirection(const Mat& image, Mat& binaryImage) const {
	const int MIN_HAND_DISPROPORTION = 4;
	binaryImage = getBinary(image);

	Mat binaryImageContour;
	binaryImage.copyTo(binaryImageContour);
	vector<vector<Point>> contours;
	findContours(binaryImageContour, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	vector<Point> biggestContour;
	int sizeOfBiggestContour = 0;
	for (auto contour : contours) {
		if (contour.size() > sizeOfBiggestContour) {
			sizeOfBiggestContour = contour.size();
			biggestContour = contour;
		}
	}
	if(!biggestContour.empty()) {
		vector<Point> hullPoints;
		vector<int> hullIndexes;
		vector<Vec4i> defects;
		Rect bRect = boundingRect(Mat(biggestContour));
		convexHull(Mat(biggestContour), hullPoints, false, true);
		convexHull(Mat(biggestContour), hullIndexes, false, false);
		approxPolyDP(Mat(hullPoints), hullPoints, 18, true);

		if(biggestContour.size() > 3) {
			convexityDefects(biggestContour, hullIndexes, defects);
			eleminateDefects(defects, biggestContour, bRect);

			if(bRect.height / bRect.width < MIN_HAND_DISPROPORTION && 
			   bRect.width / bRect.height < MIN_HAND_DISPROPORTION &&
			   defects.empty()) {
			  Point fingerTip = getFingerTip(biggestContour, hullPoints, bRect);
				if (fingerTip != Point()) {
					Point directionPoint = getDirectionPoint(fingerTip, biggestContour, bRect);
					return Scalar(directionPoint.x, directionPoint.y, fingerTip.x, fingerTip.y);
				}
			}
		}
	}
	return Scalar();
}

Mat ShootingChecker::getBinary(const Mat& image) const {
	Mat lowResImage;
	pyrDown(image, lowResImage);
	blur(lowResImage, lowResImage, Size(3, 3));
	cvtColor(lowResImage, lowResImage, CV_BGR2HLS);

	Mat binaryImage(lowResImage.size(), CV_8U, Scalar());
	for(auto avgColor : averageColors){
		// normalize colors
		for(int i = 0; i < 3; ++i) {
			if(avgColor[i] - C_LOWER[i] < 0)
				C_LOWER[i] = avgColor[i];
			if(avgColor[i] + C_UPPER[i] > 255)
				C_UPPER[i] = 255 - avgColor[i];
		}

		Scalar lowerBound = Scalar(avgColor[0] - C_LOWER[0], avgColor[1] - C_LOWER[1], avgColor[2] - C_LOWER[2]);
		Scalar upperBound = Scalar(avgColor[0] + C_UPPER[0], avgColor[1] + C_UPPER[1], avgColor[2] + C_UPPER[2]);
		Mat foo(lowResImage.size(), CV_8U);
		inRange(lowResImage, lowerBound, upperBound, foo);
		binaryImage += foo;
	}
	medianBlur(binaryImage, binaryImage, averageColors.size());
	pyrUp(binaryImage, binaryImage);
	
	if (filterMuar) {
		filterGarbage(binaryImage);
	}

	return binaryImage;
}

void ShootingChecker::filterGarbage(Mat& imgBinary) const {
	const int IMG_BINARIES_SIZE = 1;
	static deque<Mat> imgBinaries;

	// filter little garbage from the bitmap
	erode(imgBinary, imgBinary, Mat());
	dilate(imgBinary, imgBinary, Mat());

	// bitmask element is true only if it was true during the last few iterations
	Mat imgBinarySource = imgBinary.clone();
	for(auto it = imgBinaries.begin(); it != imgBinaries.end(); ++it) {
		bitwise_and(imgBinary, *it, imgBinary);
	}
	imgBinaries.push_back(imgBinarySource);
	if (imgBinaries.size() > IMG_BINARIES_SIZE)
		imgBinaries.pop_front();
}

void ShootingChecker::eleminateDefects(vector<Vec4i>& defects, const vector<Point> contour, const Rect& bRect) const {
	int distanceTolerance = bRect.height / COMP_PART_OF_BRECT_SIDE;

	vector<Vec4i> newDefects;
	int startidx, endidx, faridx;
	for (auto defect : defects) {
		Point ptStart(contour[defect[0]]);
		Point ptEnd(contour[defect[1]]);
		Point ptFar(contour[defect[2]]);

		int yLimit = bRect.y + 3 * bRect.height / 4;
		if(distanceP2P(ptStart, ptFar) > distanceTolerance && 
		   distanceP2P(ptEnd, ptFar) > distanceTolerance && 
		   getAngle(ptStart, ptFar, ptEnd) < FINGERS_MAX_ANGLE &&
		   ptEnd.y < yLimit && ptStart.y < yLimit) {
			newDefects.push_back(defect);
		}
	}
	defects.swap(newDefects);
	removeRedundantEndPoints(contour, defects, bRect);
}

// remove endpoint of convexity defects if they are at the same fingertip
// TODO: very strange function, can't understand what it actually does
void ShootingChecker::removeRedundantEndPoints(vector<Point> contour, const vector<Vec4i> newDefects, const Rect& bRect) const {
	float tolerance = bRect.width / 6;
	for(int i = 0; i < newDefects.size(); i++) {
		Vec4i defect1 = newDefects[i];
		for(int j = i; j < newDefects.size(); j++) {
			Vec4i defect2 = newDefects[j];

			Point ptStart(contour[defect1[0]]), ptEnd(contour[defect1[1]]);
			Point ptStart2(contour[defect2[0]]), ptEnd2(contour[defect2[1]]);
			if(distanceP2P(ptStart, ptEnd2) < tolerance){
				contour[defect1[0]] = ptEnd2;
				break;
			}
			if(distanceP2P(ptEnd, ptStart2) < tolerance){
				contour[defect2[0]] = ptEnd;
			}
		}
	}
}

Point ShootingChecker::getFingerTip(vector<Point> contour, const vector<Point> hullPoints, const Rect& bRect) const {
	int yTol = bRect.height / COMP_PART_OF_BRECT_SIDE;
	int xTol = bRect.width / COMP_PART_OF_BRECT_SIDE;

	Point highestP = *min_element(contour.begin(), contour.end(), [](const Point& p1, const Point& p2) {
			return p1.y < p2.y;
		});
        bool isShootingHand = none_of(hullPoints.begin(), hullPoints.end(), [highestP, yTol, xTol](Point v) {
			bool b = v.y < highestP.y + yTol && abs(v.y - highestP.y) > yTol && abs(v.x - highestP.x) > xTol;
#if DEVEL == 1
			if (b) {
				cerr << "Bad:" << endl;
				cerr << "v = " << v << "; p = " << highestP << "; yp+yTol = " << highestP.y + yTol << endl;
			}
#endif
			return b;
		});
	if(isShootingHand)
		return highestP;
	return Point();
}

Point ShootingChecker::getDirectionPoint(const Point& fingerTip, vector<Point> contour, const Rect& bRect) const {
	int tolerance = bRect.height / COMP_PART_OF_BRECT_SIDE;

	vector<Point>::iterator it = find(contour.begin(), contour.end(), fingerTip);
	Point p1;
	vector<Point>::iterator itFwd = it;
	do {
		float distance = distanceP2P(*it, *itFwd);
		float angle = getAngle(*(itFwd - 1), *itFwd, *(itFwd + 1));
		if (distance > tolerance) {
			p1 = *itFwd;
			break;
		}
		if (itFwd == contour.end()) {
			itFwd = contour.begin();
		}
		++itFwd;
	}
	while(itFwd != it);

	Point p2;
	vector<Point>::iterator itBack = it;
	do {
		if (distanceP2P(*it, *itBack) > tolerance) {
			p2 = *itBack;
			break;
		}
		if (itBack == contour.begin()) {
			itBack = contour.end();
		}
		--itBack;
	}
	while(itBack != it);

	return (p1 + p2) * 0.5f;
}

int ShootingChecker::getMedian(vector<int> v) const {
	sort(v.begin(), v.end());
	return v.size() % 2 == 0 ? v[v.size() / 2 - 1] : v[v.size() / 2];
}

float ShootingChecker::getAngle(const Point& s, const Point& f, const Point& e) const {
	float l1 = distanceP2P(f, s);
	float l2 = distanceP2P(f, e);
	float dot = (s.x - f.x) * (e.x - f.x) + (s.y - f.y) * (e.y - f.y);
	float angle = acos(dot / (l1 * l2));
	angle = angle * 180 / M_PI;
	return angle;
}

float ShootingChecker::distanceP2P(const Point& a, const Point& b) const {
	Point c = a - b;
	return sqrt(c.x * c.x + c.y * c.y);
}
