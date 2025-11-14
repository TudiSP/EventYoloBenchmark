#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <opencv2/opencv.hpp>
#include "yolo_inference.h"
#include "timing_profiler.h"

class Visualizer {
public:
    Visualizer();
    
    // Draw detections on frame
    void drawDetections(cv::Mat& frame, 
                       const std::vector<YOLOInference::DetectionBox>& detections);
    
    // Draw timing info
    void drawTimingInfo(cv::Mat& frame, const std::string& timing_text);
    
    // Draw frame info (timestamp, FPS)
    void drawFrameInfo(cv::Mat& frame, int64_t timestamp, double fps);
    
    // Set class names
    void setClassNames(const std::vector<std::string>& names);
    
    // Set colors for each class
    void setClassColors(const std::vector<cv::Scalar>& colors);
    
private:
    std::vector<std::string> class_names_;
    std::vector<cv::Scalar> class_colors_;
};

#endif // VISUALIZATION_H