
#include <opencv2/opencv.hpp>

#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

int main(int argc, char **argv)
{
    Yolov8Custom yolo;
    // model file path
    const char *model_file = argv[1];
    // input img path
    const char *img_file = argv[2];
    // load image
    cv::Mat img = cv::imread(img_file);
    // load model file
    yolo.LoadModel(model_file);

    std::vector<Detection> objects;
    yolo.Run(img, objects);

    DrawDetections(img, objects);

    cv::imwrite("result.jpg", img);

    return 0;
}