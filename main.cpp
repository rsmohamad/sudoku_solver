#include <iostream>
#include <opencv/cv.hpp>
#include <tesseract/baseapi.h>

using namespace cv;
using namespace std;

const int OFFSET = 10;

void printSudoku(const vector<int> &sudoku)
{
    cout << "\n";
    for (int i = 0; i < sudoku.size(); i++) {
        if (i % 9 == 0)
            cout << "\n";
        if (i == 27 || i == 54)
            cout << "------------\n";
        if (i % 3 == 0 || i % 6 == 0)
            cout << "|";
        if (sudoku[i] == 0) {
            cout << "x";
        }
        else {
            cout << sudoku[i];
        }

    }
    cout << endl;
}

void sortPolygon(vector<Point2f> &p)
{
    // Sort the polygon Points
    sort(p.begin(), p.end(),
         [](const Point2f &a, const Point2f &b)
         {
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

void sortTiles(vector<vector<Point2f>> &tiles)
{
    sort(tiles.begin(), tiles.end(), [](const vector<Point2f> &a, const vector<Point2f> &b)
    {
        Rect a_center = boundingRect(a);
        Rect b_center = boundingRect(b);
        return a_center.y < b_center.y;
    });

    for (int i = 0; i < 9; i++) {
        sort(tiles.begin() + 9 * i,
             tiles.begin() + ((9 * i) + 9),
             [](const vector<Point2f> &a, const vector<Point2f> &b)
             {
                 Rect a_center = boundingRect(a);
                 Rect b_center = boundingRect(b);
                 return a_center.x < b_center.x;
             });
    }

}

int main()
{
    Mat img_gray, img = imread("../sudoku.png");
    cvtColor(img, img_gray, CV_BGR2GRAY);
    vector<int> sudoku_data;

    // 1. Pre-process the image
    Mat img_binary, img_blurred;
    GaussianBlur(img_gray, img_blurred, Size(3, 3), 0);
    adaptiveThreshold(img_blurred, img_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    bitwise_not(img_binary, img_binary);
    Mat element = getStructuringElement(MORPH_CROSS, Size(2, 2));
    erode(img_binary, img_binary, element);
    dilate(img_binary, img_binary, element);

    // 2. Detect sudoku contour
    vector<vector<Point>> contours;
    Canny(img_binary, img_binary, 1, 3, 3);
    findContours(img_binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point2f(0, 0));

    double max_area = 0;
    double min_area = (img_gray.cols * img_gray.rows) / 4;
    vector<Point2f> sudoku_contour;

    for (vector<Point> contour : contours) {
        double area = contourArea(contour);

        // Min area constraint to ignore small contours
        if (area > min_area) {

            // Approximate a polygon from a contour
            // Epsilon is the maximum length difference between the contour and the approximated polygon
            vector<Point2f> polygon;
            double epsilon = 0.05 * arcLength(contour, true);
            approxPolyDP(contour, polygon, epsilon, true);

            if (polygon.size() == 4) {
                area = contourArea(polygon);
                if (area > max_area) {
                    max_area = area;
                    sudoku_contour = polygon;
                }
            }
        }
    }

    for (Point2f p : sudoku_contour) {
        circle(img, p, 4, Scalar(255, 0, 0));
    }

    // 3. Isolate sudoku from background
    Mat sudoku(500, 500, CV_32F, Scalar(255, 255, 255));

    sortPolygon(sudoku_contour);

    vector<Point2f> pts(4);
    pts[0] = Point2f(OFFSET, OFFSET);
    pts[1] = Point2f(sudoku.cols - OFFSET, OFFSET);
    pts[2] = Point2f(OFFSET, sudoku.rows - OFFSET);
    pts[3] = Point2f(sudoku.cols - OFFSET, sudoku.rows - OFFSET);

    Mat m = getPerspectiveTransform(sudoku_contour, pts);
    warpPerspective(img_gray, sudoku, m, sudoku.size());

    // 4. Pre-process the isolated image
    Mat sudoku_binary, sudoku_blurred;
    GaussianBlur(sudoku, sudoku_blurred, Size(11, 11), 3);
    adaptiveThreshold(sudoku_blurred, sudoku_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    bitwise_not(sudoku_binary, sudoku_binary);
    dilate(sudoku_binary, sudoku_binary, element);

    // 5. Detect sudoku tiles
    Mat cannyOutput;
    vector<vector<Point>> contours2;
    vector<vector<Point2f>> tiles;

    Canny(sudoku_binary, cannyOutput, 1, 3, 3, true);
    findContours(cannyOutput, contours2, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point2f(0, 0));

    double tileArea = ((sudoku.rows - 2 * OFFSET) * (sudoku.cols - 2 * OFFSET)) / 81;

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

    // 5. Text recognition
    // Do this step if sudoku is valid ~ has 81 tiles
    if (tiles.size() == 81) {
        tesseract::TessBaseAPI ocr;

        if (ocr.Init(NULL, "eng")) {
            cout << "Could not initialize tesseract." << endl;
            exit(1);
        }

        ocr.ReadConfigFile("digits");
        ocr.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);

        sortTiles(tiles);
        sudoku_data.clear();
        for (int i = 0; i < tiles.size(); i++) {
            Mat digit(50, 50, CV_32F, Scalar(255, 255, 255));
            vector<Point2f> pts(4);
            pts[0] = Point2f(0, 0);
            pts[1] = Point2f(digit.cols, 0);
            pts[2] = Point2f(0, digit.rows);
            pts[3] = Point2f(digit.cols, digit.rows);

            Mat m = getPerspectiveTransform(tiles[i], pts);
            warpPerspective(sudoku_binary, digit, m, digit.size());

            threshold(digit, digit, 200, 255, THRESH_BINARY);
            ocr.TesseractRect(digit.data, 1, digit.step1(), 5, 5, digit.cols - 10, digit.rows - 10);

            const char *text = ocr.GetUTF8Text();
            int number = atoi(text);
            sudoku_data.push_back(number);
        }
        printSudoku(sudoku_data);
    }

    imshow("Sudoku Isolated", sudoku);
    imshow("Sudoku", img_gray);
    waitKey(0);
    return 0;
}