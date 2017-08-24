#include <iostream>
#include <opencv/cv.hpp>
#include "SudokuCV.h"

int main()
{
    SudokuCV sudokuCV;
    cv::Mat img = cv::imread("../sudoku.png");
    try {
        sudokuCV.addImageAndSolve(img);
        sudokuCV.printPuzzle();
        sudokuCV.printSolution();
        cv::Mat solution_img = sudokuCV.getProjectedResult();
        cv::imshow("Solution", solution_img);
    }
    catch (char const *e) {
        std::cout << e << std::endl;
    }

    cv::imshow("Original", img);

    while (true) {
        char key = cv::waitKey(0);
        if (key == 'q')
            break;
    }

    return 0;
}