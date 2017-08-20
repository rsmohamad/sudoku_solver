#include <iostream>
#include <cv.hpp>

using namespace cv;
using namespace std;

const int OFFSET = 10;

void sortPolygon(vector<Point2f> &p) {
  // Sort the polygon Points
  sort(p.begin(), p.end(),
       [](const Point2f &a, const Point2f &b) {
         return a.y < b.y;
       });

  for (int i = 0; i <= 2; i += 2) {
    if (p[i].x > p[i + 1].x) {
      Point2f temp = p[i];
      p[i] = p[i + 1];
      p[i + 1] = temp;
    }
  }
}


void sortTiles(vector<vector<Point2f>> &tiles){
  sort(tiles.begin(), tiles.end(), [](const vector<Point2f> &a, const vector<Point2f> &b){
    Rect a_center = boundingRect(a);
    Rect b_center = boundingRect(b);
    return a_center.y < b_center.y;
  });

  for(int i = 0; i < 9; i++){
    sort(tiles.begin() + 9 * i, tiles.begin() + ((9 * i) + 9), [](const vector<Point2f> &a, const vector<Point2f> &b){
      Rect a_center = boundingRect(a);
      Rect b_center = boundingRect(b);
      return a_center.x < b_center.x;
    });
  }

}

int main() {
  Mat sudoku_gray, sudoku = imread("../sudoku2.jpg");
  cvtColor(sudoku, sudoku_gray, CV_BGR2GRAY);

  // 1. Pre-process the image
  Mat binary, blurred;
  GaussianBlur(sudoku_gray, blurred, Size(11, 11), 0);
  adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
  bitwise_not(binary, binary);
  Mat element = getStructuringElement(MORPH_CROSS, Size(2, 2));
  erode(binary, binary, element);
  dilate(binary, binary, element);

  // 2. Detect sudoku contour
  vector<vector<Point>> contours;
  Canny(binary, binary, 1, 3, 3);
  findContours(binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point2f(0, 0));

  double maxArea = 0;
  double minArea = (sudoku_gray.cols * sudoku_gray.rows) / 4;
  vector<Point2f> sudoku_contour;

  for (vector<Point> contour : contours) {
    double area = contourArea(contour);

    // Min area constraint to ignore small contours
    if (area > minArea) {

      // Approximate a polygon from a contour
      // Epsilon is the maximum length difference between the contour and the approximated polygon
      vector<Point2f> polygon;
      double epsilon = 0.05 * arcLength(contour, true);
      approxPolyDP(contour, polygon, epsilon, true);

      if (polygon.size() == 4) {
        area = contourArea(polygon);
        if (area > maxArea) {
          maxArea = area;
          sudoku_contour = polygon;
        }
      }
    }
  }

  for (Point2f p : sudoku_contour) {
    circle(sudoku, p, 4, Scalar(255, 0, 0));
  }

  // 3. Isolate sudoku from background
  Mat sudoku_isolated(500, 500, CV_32S, Scalar(255, 255, 255));

  sortPolygon(sudoku_contour);

  vector<Point2f> pts(4);
  pts[0] = Point2f(OFFSET, OFFSET);
  pts[1] = Point2f(sudoku_isolated.cols - OFFSET, OFFSET);
  pts[2] = Point2f(OFFSET, sudoku_isolated.rows - OFFSET);
  pts[3] = Point2f(sudoku_isolated.cols - OFFSET, sudoku_isolated.rows - OFFSET);

  cout << sudoku_contour.size() << endl;
  Mat m = getPerspectiveTransform(sudoku_contour, pts);
  warpPerspective(sudoku_gray, sudoku_isolated, m, sudoku_isolated.size());

  // 4. Pre-process the isolated image
  Mat binary2, blurred2;
  GaussianBlur(sudoku_isolated, blurred2, Size(11, 11), 3);
  adaptiveThreshold(blurred2, binary2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
  bitwise_not(binary2, binary2);
  dilate(binary2, binary2, element);

  // 5. Detect sudoku tiles
  Mat cannyOutput;
  vector<vector<Point>> contours2;
  vector<vector<Point2f>> tiles;

  Canny(binary2, cannyOutput, 1, 3, 3, true);
  findContours(cannyOutput, contours2, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point2f(0, 0));

  double tileArea = ((sudoku_isolated.rows - 2 * OFFSET) * (sudoku_isolated.cols - 2 * OFFSET)) / 81;

  for (vector<Point> contour : contours2) {
    double area = contourArea(contour, true);

    if (area < tileArea * 1.5 && area > tileArea * 0.6) {
      vector<Point2f> polygon;
      double epsilon = 0.1 * arcLength(contour, true);
      approxPolyDP(contour, polygon, epsilon, true);

      if (polygon.size() == 4) {
        sortPolygon(polygon);
        tiles.push_back(polygon);
      }
    }
  }

  // Sudoku has 81 tiles
  if (tiles.size() == 81){
    sortTiles(tiles);
    for(int i = 0; i < 81; i++){
      RotatedRect rect = minAreaRect(tiles[i]);
      putText(sudoku_isolated, to_string(i), rect.center, CV_FONT_HERSHEY_PLAIN, 1, Scalar(255));
      //circle(sudoku_isolated, rect.center, 5, Scalar(255));
    }
  }

  imshow("Sudoku Isolated", sudoku_isolated);
  imshow("Sudoku", sudoku_gray);
  imshow("Sudoku Contour", binary2);
  imshow("Thersholded", cannyOutput);
  waitKey(0);
  return 0;
}