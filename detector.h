#ifndef DETECTOR_H
#define DETECTOR_H
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
#include <Eigen/Core>
#include <memory>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};
struct ArmorObject
{
    Point2f apex[4];
    cv::Rect_<float> rect;
    int cls;
    int area;
    float prob;
    std::vector<cv::Point2f> pts;
};


class ArmorDetector
{
public:
    ArmorDetector();
    ~ArmorDetector();
    bool detect(Mat &src,vector<ArmorObject>& objects);
    bool initModel(string path,double cof_threshold,double nms_area_threshold);
    bool process_frame(Mat& inframe);
    Mat preprocess_img(Mat& inframe);
    void generateYoloxProposals(Mat& frame,const float* feat_ptr,
                                       float prob_threshold
                                       );
private:

    Core ie;
    CNNNetwork network;                // 网络
    ExecutableNetwork executable_network;       // 可执行网络
    InferRequest infer_request;      // 推理请求
    MemoryBlob::CPtr moutput;
    string _input_name;
    string _output_name;
    string _xml_path;
    float _cof_threshold;
    float _nms_area_threshold;
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;

    float rx;   // the width ratio of original image and resized image
    float ry;   // the height ratio of original image and resized image

    Eigen::Matrix<float,3,3> transfrom_matrix;
};
#endif


