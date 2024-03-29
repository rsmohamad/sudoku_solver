//
// Created by dandi on 8/24/17.
//

#ifndef SUDOKU_SOLVER_SUDOKUCV_H
#define SUDOKU_SOLVER_SUDOKUCV_H

#include <opencv/cv.hpp>
#include <tesseract/baseapi.h>
#include <vector>
#include <algorithm>

using cv::Mat;
using cv::Point;
using cv::Point2f;
using std::vector;

const int OFFSET = 10;

const int DIGIT_OFFSET = 0;

class SudokuCV
{
private:
    vector<int> sudoku_unsolved;
    vector<int> sudoku_solved;
    vector<Point2f> sudoku_contour;
    vector<Point2f> crossing_points;
    Mat img_sudoku;
    Mat img_raw;

    bool findSudokuEdge();
    bool findSudokuTilesHoughLines();
    bool findSudokuTilesPrimitive();
    bool recognizeText();

    bool isSolved(const vector<int> &sudoku) const;
    bool isSudokuCorrect(const vector<int> &sudoku) const;
    bool solveSudoku(vector<int> &sudoku, int index = 0);
    bool solveSudoku();

    void preprocessDigit(Mat &digit) const;
    void sortPoints(vector<Point2f> &points);
    void printSudoku(const vector<int> &sudoku) const;
    vector<Point2f> findLargestQuadilateralApprox(const vector<vector<Point>> &contours) const;
public:
    SudokuCV()
    {}
    bool addImageAndSolve(const Mat &img);
    void printPuzzle() const;
    void printSolution() const;
    vector<int> getPuzzle() const;
    vector<int> getSolution() const;
    Mat getProjectedResult();
    bool getContouredImage(const Mat &src, Mat &dst);

};


#endif //SUDOKU_SOLVER_SUDOKUCV_H
