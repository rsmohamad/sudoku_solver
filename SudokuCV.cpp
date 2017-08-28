//
// Created by dandi on 8/24/17.
//

#include "SudokuCV.h"
#include <iostream>

using namespace cv;
using std::cout;
using std::endl;
using std::to_string;

void warpPerspectiveWithOffset(const Mat &src, Mat &dst, const vector<Point2f> source_points, const int offset)
{
    if (source_points.size() != 4 || offset > dst.rows || offset > dst.cols)
        return;

    vector<Point2f> pts(4);
    pts[0] = Point2f(offset, offset);
    pts[1] = Point2f(dst.cols - offset, offset);
    pts[2] = Point2f(offset, dst.rows - offset);
    pts[3] = Point2f(dst.cols - offset, dst.rows - offset);

    Mat m = getPerspectiveTransform(source_points, pts);
    warpPerspective(src, dst, m, dst.size());
}

void showLines(const Mat &img, vector<Vec2f> lines)
{
    Mat temp = img.clone();
    cvtColor(temp, temp, CV_GRAY2BGR);
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(temp, pt1, pt2, Scalar(0, 255, 0), 2, CV_AA);
    }
    imshow("Detected Lines", temp);
}

bool SudokuCV::findSudokuEdge()
{
    Mat img_gray, img_binary, img_blurred;
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
    vector<vector<Point>> contours;

    // Blur and convert to binary
    cvtColor(img_raw, img_gray, CV_BGR2GRAY);
    GaussianBlur(img_gray, img_blurred, Size(3, 3), 0);
    adaptiveThreshold(img_blurred, img_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    bitwise_not(img_binary, img_binary);
    erode(img_binary, img_binary, kernel);
    dilate(img_binary, img_binary, kernel);

    // Detect sudoku contour
    Canny(img_binary, img_binary, 100, 200, 5);
    findContours(img_binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point2f(0, 0));
    sudoku_contour = findLargestQuadilateralApprox(contours);

    if (sudoku_contour.size() != 4)
        return false;
    sortPoints(sudoku_contour);
    img_sudoku = Mat(500, 500, CV_32F, Scalar(255, 255, 255));
    warpPerspectiveWithOffset(img_gray, img_sudoku, sudoku_contour, OFFSET);
    return true;
}

bool SudokuCV::findSudokuTilesHoughLines()
{
    Mat sudoku_binary, canny_output;
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
    vector<Vec2f> lines, strong_lines, horizontal_lines, vertical_lines;

    // Blur and convert to binary
    adaptiveThreshold(img_sudoku, sudoku_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    medianBlur(sudoku_binary, sudoku_binary, 3);
    bitwise_not(sudoku_binary, sudoku_binary);
    dilate(sudoku_binary.clone(), sudoku_binary, kernel);
    Canny(sudoku_binary, canny_output, 1, 3);

    // Find gridlines
    HoughLines(canny_output, lines, 1, CV_PI / 2, 100, 0, 0);

    // Remove duplicate lines
    for (int i = 0; i < lines.size(); i++) {
        auto it = find_if(strong_lines.begin(), strong_lines.end(), [=](const Vec2f &a) -> bool
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

    // Find line intercepts
    for (Vec2f horizontal : horizontal_lines) {
        for (Vec2f vertical : vertical_lines) {
            // This works because HoughLines resolution is set to 90 degrees
            crossing_points.push_back(Point2f(vertical[0], horizontal[0]));
        }
    }

    if (crossing_points.size() != 100) {
        showLines(img_sudoku, strong_lines);
        imshow("Canny", canny_output);
        return false;
    }
    sortPoints(crossing_points);
    return true;
}

bool SudokuCV::findSudokuTilesPrimitive()
{
    crossing_points.clear();
    int hor_step = (img_sudoku.cols - OFFSET * 2) / 9;
    int ver_step = (img_sudoku.rows - OFFSET * 2) / 9;
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            crossing_points.push_back(Point2f(OFFSET + i * hor_step, OFFSET + j * ver_step));
    sortPoints(crossing_points);
    return true;
}

bool SudokuCV::recognizeText()
{
    if (crossing_points.size() != 100)
        return false;
    tesseract::TessBaseAPI ocr;
    Mat digit(200, 200, CV_32F, Scalar(255, 255, 255));
    vector<Point2f> tiles(4);

    if (ocr.Init(NULL, "eng"))
        return false;
    ocr.ReadConfigFile("digits");
    ocr.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
    sudoku_unsolved.clear();

    for (int i = 0; i < 81; i++) {
        int row = i / 9;
        int col = i % 9;
        tiles[0] = crossing_points[row * 10 + col];
        tiles[1] = crossing_points[row * 10 + col + 1];
        tiles[2] = crossing_points[(row + 1) * 10 + col];
        tiles[3] = crossing_points[(row + 1) * 10 + col + 1];
        warpPerspectiveWithOffset(img_sudoku, digit, tiles, DIGIT_OFFSET);
        //imshow("Digit raw", digit);
        preprocessDigit(digit);
        ocr.TesseractRect(digit.data, 1, digit.step1(), 0, 0, digit.cols, digit.rows);
        const char *text = ocr.GetUTF8Text();
        int number = atoi(text);
        sudoku_unsolved.push_back(number);

//        imshow("Digit", digit);
//        setWindowTitle("Digit", to_string(number));
//        waitKey(0);
    }
    return true;
}

bool SudokuCV::isSolved(const vector<int> &sudoku) const
{
    return find(sudoku.begin(), sudoku.end(), 0) == sudoku.end() && isSudokuCorrect(sudoku);
}

bool SudokuCV::isSudokuCorrect(const vector<int> &sudoku) const
{
    if (sudoku.size() != 81)
        return false;

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

bool SudokuCV::solveSudoku(vector<int> &sudoku, int index)
{
    if (sudoku.size() != 81)
        return false;

    if (isSolved(sudoku))
        return true;

    if (sudoku[index])
        return solveSudoku(sudoku, index + 1);
    else {
        for (int i = 1; i <= 9; i++) {
            sudoku[index] = i;
            if (!isSudokuCorrect(sudoku))
                continue;
            else if (solveSudoku(sudoku, index + 1))
                return true;
        }
        sudoku[index] = 0;
        return false;
    }
}

bool SudokuCV::solveSudoku()
{
    if (sudoku_unsolved.size() != 81)
        return false;
    sudoku_solved = sudoku_unsolved;
    return solveSudoku(sudoku_solved);
}

void SudokuCV::preprocessDigit(Mat &digit) const
{
    medianBlur(digit, digit, 7);
    //Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    Mat kernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
    filter2D(digit, digit, -1, kernel);
    adaptiveThreshold(digit, digit, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 2);

    // Flood fill hack to remove borders
    for (int i = 0; i < 15; i++)
        for (int j = 0; j < digit.rows - 1; j++) {
            if (digit.data[i + j * digit.step] == 255)
                floodFill(digit, Point2f(i, j), Scalar(0));
            if (digit.data[(digit.cols - 1 - i) + j * digit.step] == 255)
                floodFill(digit, Point2f(digit.cols - 1 - i, j), Scalar(0));
        }
    for (int j = 0; j < 15; j++)
        for (int i = 0; i < digit.cols - 1; i++) {
            if (digit.data[i + j * digit.step] == 255)
                floodFill(digit, Point2f(i, j), Scalar(0));
            if (digit.data[i + (digit.rows - 1 - j) * digit.step] == 255)
                floodFill(digit, Point2f(i, digit.rows - 1 - j), Scalar(0));
        }
    medianBlur(digit, digit, 15);
}

void SudokuCV::sortPoints(vector<Point2f> &p)
{
    double size = sqrt(p.size());
    if (size != (int) size)
        return;

    sort(p.begin(), p.end(), [](const Point2f &a, const Point2f &b)
    {
        return a.y < b.y;
    });

    for (int i = 0; i < size; i++) {
        sort(p.begin() + size * i,
             p.begin() + ((size * i) + size),
             [](const Point2f &a, const Point2f &b)
             {
                 return a.x < b.x;
             });
    }
}

void SudokuCV::printSudoku(const vector<int> &sudoku) const
{
    cout << "\n";
    for (int i = 0; i < sudoku.size(); i++) {
        if (i % 9 == 0)
            cout << "\n";
        else if (i % 3 == 0)
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

vector<Point2f> SudokuCV::findLargestQuadilateralApprox(const vector<vector<Point>> &contours) const
{
    vector<Point2f> quad;
    double max_area = 0;
    double min_area = img_raw.cols * img_raw.rows / 4;
    for (vector<Point> contour : contours) {
        double area = contourArea(contour);
        // Approximate a polygon from a contour
        // Epsilon is the maximum length difference between the contour and the approximated polygon
        vector<Point2f> polygon;
        double epsilon = 0.05 * arcLength(contour, true);
        approxPolyDP(contour, polygon, epsilon, true);
        if (polygon.size() == 4 && area > min_area) {
            area = contourArea(polygon);
            if (area > max_area) {
                max_area = area;
                quad = polygon;
            }
        }
    }
    return quad;
}

bool SudokuCV::addImageAndSolve(const Mat &img)
{
    sudoku_unsolved.clear();
    sudoku_solved.clear();
    sudoku_contour.clear();
    crossing_points.clear();
    img_raw = img.clone();

    // 1. Find sudoku edge
    if (!findSudokuEdge())
        throw "No quadilateral detected";

//    // 2. Find gridlines
//    if (!findSudokuTilesHoughLines())
//        throw "Cannot find tiles";

    // 2. Find gridlines
    if (!findSudokuTilesPrimitive())
        throw "Cannot find tiles";

    // 3. Recognize text
    if (!recognizeText())
        throw "Could not initialize tesseract";

    // 4. Solve puzzle
    if (!solveSudoku()) {
        printPuzzle();
        throw "Sudoku cannot be solved";
    }

    return true;
}

void SudokuCV::printPuzzle() const
{
    printSudoku(sudoku_unsolved);
}

void SudokuCV::printSolution() const
{
    printSudoku(sudoku_solved);
}

vector<int> SudokuCV::getPuzzle() const
{
    if (sudoku_unsolved.size() == 81)
        return sudoku_unsolved;
    else
        throw "No sudoku detected";
}

vector<int> SudokuCV::getSolution() const
{
    if (isSolved(sudoku_solved))
        return sudoku_solved;
    else
        throw "No solution found";
}

Mat SudokuCV::getProjectedResult()
{
    if (sudoku_solved.size() != 81 || sudoku_unsolved.size() != 81 || crossing_points.size() != 100)
        return Mat();

    Mat retval = img_sudoku.clone();
    cvtColor(retval, retval, CV_GRAY2BGR);
    vector<Point2f> tiles(4);
    for (int i = 0; i < 81; i++) {
        if (sudoku_unsolved[i] == 0) {
            int row = i / 9;
            int col = i % 9;
            tiles[0] = crossing_points[row * 10 + col];
            tiles[1] = crossing_points[row * 10 + col + 1];
            tiles[2] = crossing_points[(row + 1) * 10 + col];
            tiles[3] = crossing_points[(row + 1) * 10 + col + 1];
            Point2f center = minAreaRect(tiles).center;
            center += Point2f(-5, 7);
            putText(retval, to_string(sudoku_solved[i]), center, CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        }
    }
    return retval.clone();
}

bool SudokuCV::getContouredImage(const Mat &src, Mat &dst)
{
    sudoku_contour.clear();
    dst = src.clone();
    img_raw = src.clone();
    if (findSudokuEdge()) {
        line(dst, sudoku_contour[0], sudoku_contour[1], Scalar(0, 0, 255), 2);
        line(dst, sudoku_contour[1], sudoku_contour[3], Scalar(0, 0, 255), 2);
        line(dst, sudoku_contour[3], sudoku_contour[2], Scalar(0, 0, 255), 2);
        line(dst, sudoku_contour[2], sudoku_contour[0], Scalar(0, 0, 255), 2);

        return true;
    }
    return false;

}
