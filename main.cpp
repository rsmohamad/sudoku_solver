#include <iostream>
#include <opencv/cv.hpp>
#include <tesseract/baseapi.h>

using namespace cv;
using namespace std;

const int OFFSET = 10;

const int DIGIT_OFFSET = 0;

bool isSudokuCorrect(const vector<int> &sudoku)
{
    vector<int> num_set(9);
    for (int i = 0; i < 9; i++) {
        // Check each row
        fill(num_set.begin(), num_set.end(), 0);
        for (int j = 0; j < 9; j++) {
            int index = i * 9 + j;
            if (sudoku[index])
                if (++num_set[sudoku[index] - 1] > 1)
                    return false;
        }

        // Check each column
        fill(num_set.begin(), num_set.end(), 0);
        for (int j = 0; j < 9; j++) {
            int index = i + j * 9;
            if (sudoku[index])
                if (++num_set[sudoku[index] - 1] > 1)
                    return false;
        }

        //Check each box
        fill(num_set.begin(), num_set.end(), 0);
        for (int j = 0; j < 9; j++) {

            int index = (i / 3) * 27;
            index += (i % 3) * 3;
            index += (j / 3) * 9;
            index += (j % 3);

            if (sudoku[index])
                if (++num_set[sudoku[index] - 1] > 1)
                    return false;
        }
    }
    return true;
}

// Backtracking method
bool solveSudoku(vector<int> &sudoku, int numStart = 0)
{
    if (find(sudoku.begin(), sudoku.end(), 0) == sudoku.end() && isSudokuCorrect(sudoku))
        return true;

    if (sudoku[numStart])
        return solveSudoku(sudoku, numStart + 1);
    else {
        for (int i = 1; i <= 9; i++) {
            sudoku[numStart] = i;
            if (!isSudokuCorrect(sudoku))
                continue;
            else if (solveSudoku(sudoku, numStart + 1))
                return true;
        }

        sudoku[numStart] = 0;
        return false;
    }
}

void printSudoku(const vector<int> &sudoku)
{
    cout << "\n";
    for (int i = 0; i < sudoku.size(); i++) {

        if (i % 9 == 0)
            cout << "\n";
        else if (i % 3 == 0 || i % 6 == 0)
            cout << "|";

        if (i == 27 || i == 54)
            cout << "-----------\n";

        if (sudoku[i] == 0)
            cout << ".";
        else
            cout << sudoku[i];

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

void sortPoints(vector<Point2f> &tiles)
{
    sort(tiles.begin(), tiles.end(), [](const Point2f &a, const Point2f &b)
    {
        return a.y < b.y;
    });

    for (int i = 0; i < 10; i++) {
        sort(tiles.begin() + 10 * i,
             tiles.begin() + ((10 * i) + 10),
             [](const Point2f &a, const Point2f &b)
             {
                 return a.x < b.x;
             });
    }

}

int main()
{
    //Mat img = imread("../sudoku.jpg");
    Mat img = imread("/home/dandi/Desktop/sudoku.png");
    Mat img_gray;
    Mat img_binary, img_blurred;
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
    cvtColor(img, img_gray, CV_BGR2GRAY);
    vector<int> sudoku_data;
    vector<Point2f> sudoku_contour;
    vector<vector<Point>> contours;
    Mat sudoku(500, 500, CV_32F, Scalar(255, 255, 255));
    Mat sudoku_binary, sudoku_blurred;

    // 1. Pre-process the image

    GaussianBlur(img_gray, img_blurred, Size(3, 3), 0);
    adaptiveThreshold(img_blurred, img_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    bitwise_not(img_binary, img_binary);

    erode(img_binary, img_binary, kernel);
    dilate(img_binary, img_binary, kernel);


    // 2. Detect sudoku contour

    Canny(img_binary, img_binary, 100, 200, 5);
    findContours(img_binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point2f(0, 0));

    //imshow("Canny", img_binary);
    waitKey(10000);

    double max_area = 0;
    double min_area = (img_gray.cols * img_gray.rows) / 4;


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

    cout << sudoku_contour.size() << endl;

    // 3. Isolate sudoku from background
    sortPolygon(sudoku_contour);
    vector<Point2f> pts(4);
    pts[0] = Point2f(OFFSET, OFFSET);
    pts[1] = Point2f(sudoku.cols - OFFSET, OFFSET);
    pts[2] = Point2f(OFFSET, sudoku.rows - OFFSET);
    pts[3] = Point2f(sudoku.cols - OFFSET, sudoku.rows - OFFSET);
    Mat m = getPerspectiveTransform(sudoku_contour, pts);
    warpPerspective(img_gray, sudoku, m, sudoku.size());

    Mat sudoku_clone = sudoku.clone();

    // 4. Pre-process the isolated image
    adaptiveThreshold(sudoku, sudoku_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    medianBlur(sudoku_binary, sudoku_binary, 3);
    bitwise_not(sudoku_binary, sudoku_binary);
    dilate(sudoku_binary.clone(), sudoku_binary, kernel);

    // 5. Detect sudoku tiles
    Mat cannyOutput;
    vector<Vec2f> lines, strong_lines, horizontal_lines, vertical_lines;
    vector<Point2f> crossing_points;
    Canny(sudoku_binary, cannyOutput, 1, 3);

    // Find gridlines
    HoughLines(cannyOutput, lines, 1, CV_PI / 2, 100, 0, 0);

    // Remove duplicate lines
    for (int i = 0; i < lines.size(); i++) {
        vector<Vec2f>::iterator it = find_if(strong_lines.begin(), strong_lines.end(), [=](const Vec2f &a) -> bool
        {
            return fabs(a[0] - lines[i][0]) < 15 && fabs(a[1] - lines[i][1]) < CV_PI / 4;
        });

        if (it == strong_lines.end())
            strong_lines.push_back(lines[i]);

    }

    // Sort between horizontal and vertical lines
    for (int i = 0; i < strong_lines.size(); i++) {
        if (fabs(strong_lines[i][1]) < CV_PI / 4)
            vertical_lines.push_back(strong_lines[i]);
        else
            horizontal_lines.push_back(strong_lines[i]);
    }

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(sudoku, pt1, pt2, Scalar(0), 1, CV_AA);
    }

    // Find line intercepts
    for (Vec2f horizontal : horizontal_lines) {
        for (Vec2f vertical : vertical_lines) {
            // This works because HoughLines has a resolution of 90 degrees
            crossing_points.push_back(Point2f(vertical[0], horizontal[0]));
        }
    }

    for (int i = 0; i < crossing_points.size(); i++) {
        circle(sudoku, crossing_points[i], 3, Scalar(64));
    }

    // 6. Text recognition
    // Do this step if sudoku is valid ~ has 81 tiles

    cout << crossing_points.size() << endl;

    if (crossing_points.size() == 100) {
        tesseract::TessBaseAPI ocr;

        if (ocr.Init(NULL, "eng")) {
            cout << "Could not initialize tesseract." << endl;
            exit(1);
        }

        ocr.ReadConfigFile("digits");
        ocr.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);

        Mat digit(200, 200, CV_32F, Scalar(255, 255, 255));
        vector<Point2f> pts(4), tiles(4);
        pts[0] = Point2f(DIGIT_OFFSET, DIGIT_OFFSET);
        pts[1] = Point2f(digit.cols - DIGIT_OFFSET, DIGIT_OFFSET);
        pts[2] = Point2f(DIGIT_OFFSET, digit.rows - DIGIT_OFFSET);
        pts[3] = Point2f(digit.cols - DIGIT_OFFSET, digit.rows - DIGIT_OFFSET);

        sortPoints(crossing_points);
        sudoku_data.clear();
        for (int i = 0; i < 81; i++) {

            int row = i / 9;
            int col = i % 9;
            tiles[0] = crossing_points[row * 10 + col];
            tiles[1] = crossing_points[row * 10 + col + 1];
            tiles[2] = crossing_points[(row + 1) * 10 + col];
            tiles[3] = crossing_points[(row + 1) * 10 + col + 1];

            Mat m = getPerspectiveTransform(tiles, pts);
            warpPerspective(sudoku_clone, digit, m, digit.size());
            medianBlur(digit, digit, 11);
            adaptiveThreshold(digit, digit, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 2);
            medianBlur(digit, digit, 11);
            bitwise_not(digit, digit);

            // Flood fill hack to remove borders
            for (int i = 1; i < 10; i++)
                for (int j = 1; j < digit.rows; j++) {
                    if (digit.data[i + j * digit.step] == 255)
                        floodFill(digit, Point2f(i, j), Scalar(0));
                    if (digit.data[(digit.cols - i) + j * digit.step] == 255)
                        floodFill(digit, Point2f(digit.cols - i, j), Scalar(0));
                }

            for (int j = 1; j < 10; j++)
                for (int i = 1; i < digit.cols; i++) {
                    if (digit.data[i + j * digit.step] == 255)
                        floodFill(digit, Point2f(i, j), Scalar(0));
                    if (digit.data[i + (digit.rows - j) * digit.step] == 255)
                        floodFill(digit, Point2f(i, digit.rows - j), Scalar(0));
                }

            ocr.TesseractRect(digit.data, 1, digit.step1(), 10, 10, digit.cols - 20, digit.rows - 20);

            const char *text = ocr.GetUTF8Text();
            int number = atoi(text);
            sudoku_data.push_back(number);

            cout << number << endl;
            imshow("Digit", digit);
            waitKey(10000);
        }

        printSudoku(sudoku_data);
        solveSudoku(sudoku_data);
        printSudoku(sudoku_data);
    }

    imshow("Sudoku Isolated", sudoku);
    // imshow("Sudoku blurred", sudoku_blurred);
    imshow("Canny", cannyOutput);
    waitKey(0);
    return 0;
}