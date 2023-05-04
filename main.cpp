#include "detector.h"
#include "detector.cpp"
int main(int argc, char const *argv[])
{
    ArmorDetector* detector = new ArmorDetector;

    string xml_path = "../weights/best_new.onnx";
    detector->initModel(xml_path,0.5,0.6);

//    VideoCapture capture;
//    capture.open("/home/chenzhen/Downloads/1.avi");
//    Mat src;
//    //Mat src = imread("../res/3.jpg");
//    while(1)
//    {
//        capture>>src;
//        Mat osrc = src.clone();
//        osrc = letterbox(osrc);
//
//        vector<ArmorObject> detected_objects;
//        auto start = chrono::high_resolution_clock::now();
//
//        detector->process_frame(src,detected_objects);
//
//        auto end = chrono::high_resolution_clock::now();
//        std::chrono::duration<double> diff = end - start;
//        //cout<<"use "<<diff.count()<<" s" << endl;
//        for(int i=0;i<detected_objects.size();++i){
//            Point2f location(detected_objects[i].rect.x,detected_objects[i].rect.y);
//            int class_id = detected_objects[i].cls;
//            Point2f p1 = detected_objects[i].pts[0];
//            Point2f p2 = detected_objects[i].pts[1];
//            Point2f p3 = detected_objects[i].pts[2];
//            Point2f p4 = detected_objects[i].pts[3];
//            //cv::putText(osrc, to_string(class_id),location,2,1,Scalar(0,0,255),2,8,0);
//            cv::circle(osrc,p1,2,Scalar(0, 0, 255),-1,8,0);
//            cv::circle(osrc,p2,2,Scalar(0, 0, 255),-1,8,0);
//            cv::circle(osrc,p3,2,Scalar(0, 0, 255),-1,8,0);
//            cv::circle(osrc,p4,2,Scalar(0, 0, 255),-1,8,0);
//            cv::line(osrc,p1, p3, Scalar(0, 0, 255), 2,8,0);
//            cv::line(osrc,p1, p2, Scalar(0, 0, 255), 2,8,0);
//            cv::line(osrc,p2, p4, Scalar(0, 0, 255), 2,8,0);
//            cv::line(osrc,p3, p4, Scalar(0, 0, 255), 2,8,0);
//
////            Rect rect(xmin, ymin, width, height);//左上坐标（x,y）和矩形的长(x)宽(y)
//        }
//
//        imshow("result",osrc);
//        waitKey(10);
//    }
//    Mat osrc = imread("../data/1.jpg");
//
//    auto start = chrono::high_resolution_clock::now();
//
//    detector->process_frame(osrc);
//
//    auto end = chrono::high_resolution_clock::now();
//    std::chrono::duration<double> diff = end - start;
//    //cout<<"use "<<diff.count()<<" s" << endl;
//
//    imshow("result",osrc);
//    waitKey(0);


    VideoCapture capture;
    capture.open("/home/chenzhen/Downloads/1.avi");
    Mat osrc;
    while(1){
        capture>>osrc;
        detector->process_frame(osrc);
    }

}

