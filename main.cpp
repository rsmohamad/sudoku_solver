#include <iostream>
#include <opencv/cv.hpp>
#include "SudokuCV.h"

int main()
{
    SudokuCV sudokuCV;
    cv::Mat img, img_show;
    cv::VideoCapture cap(0);
    img = cv::imread("../sudoku.jpg");

    while (true) {
        cap >> img;
        bool contour_found = sudokuCV.getContouredImage(img, img_show);
        cv::imshow("Camera", img_show);
        char key = cv::waitKey(1);
        if(key == 'c' && contour_found)
            try {
                sudokuCV.addImageAndSolve(img);
                sudokuCV.printPuzzle();
                sudokuCV.printSolution();
                cv::Mat solution_img = sudokuCV.getProjectedResult();
                cv::imshow("Solution", solution_img);
                cv::imshow("Original", img);
            }
            catch (char const *e) {
                std::cout << e << std::endl;
            }
        else if (key == 'q')
            break;
    }

    return 0;
}