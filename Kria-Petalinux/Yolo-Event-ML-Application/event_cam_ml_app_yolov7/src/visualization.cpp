#include "visualization.h"

Visualizer::Visualizer() {
    // Default class names
    class_names_ = {"person", "Vehicle"};
    
    // Default colors (BGR format)
    class_colors_ = {
        cv::Scalar(255, 0, 0),    // Blue for person
        cv::Scalar(0, 0, 255)     // Red for vehicle
    };
}

void Visualizer::setClassNames(const std::vector<std::string>& names) {
    class_names_ = names;
}

void Visualizer::setClassColors(const std::vector<cv::Scalar>& colors) {
    class_colors_ = colors;
}

void Visualizer::drawDetections(cv::Mat& frame,
                                const std::vector<YOLOInference::DetectionBox>& detections) {
    TIME_SCOPE("4_visualization");
    
    float h = frame.rows;
    float w = frame.cols;
    
    for (const auto& det : detections) {
        // Calculate box coordinates
        float xmin = (det.x - det.w / 2.0f) * w + 1.0f;
        float ymin = (det.y - det.h / 2.0f) * h + 1.0f;
        float xmax = (det.x + det.w / 2.0f) * w + 1.0f;
        float ymax = (det.y + det.h / 2.0f) * h + 1.0f;
        
        // Get class info
        int class_id = det.class_id;
        std::string class_name = (class_id < class_names_.size()) 
            ? class_names_[class_id] : "Unknown";
        cv::Scalar color = (class_id < class_colors_.size())
            ? class_colors_[class_id] : cv::Scalar(255, 200, 20);
        
        // Draw bounding box
        cv::rectangle(frame, 
                     cv::Point(xmin, ymin), 
                     cv::Point(xmax, ymax),
                     color, 2, cv::LINE_AA);
        
        // Draw label with confidence
        std::string label = class_name + " " + 
                           std::to_string(int(det.confidence * 100)) + "%";
        
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                              0.5, 1, &baseline);
        
        // Draw label background
        cv::rectangle(frame,
                     cv::Point(xmin, ymin - label_size.height - 5),
                     cv::Point(xmin + label_size.width, ymin),
                     color, cv::FILLED);
        
        // Draw label text
        cv::putText(frame, label,
                   cv::Point(xmin, ymin - 5),
                   cv::FONT_HERSHEY_SIMPLEX,
                   0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

void Visualizer::drawTimingInfo(cv::Mat& frame, const std::string& timing_text) {
    cv::putText(frame, timing_text,
               cv::Point(10, frame.rows - 10),
               cv::FONT_HERSHEY_SIMPLEX,
               0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
}

void Visualizer::drawFrameInfo(cv::Mat& frame, int64_t timestamp, double fps) {
    std::stringstream ss;
    ss << "TS: " << timestamp << " us | FPS: " << std::fixed 
       << std::setprecision(1) << fps;
    
    cv::putText(frame, ss.str(),
               cv::Point(10, 20),
               cv::FONT_HERSHEY_SIMPLEX,
               0.6, cv::Scalar(108, 143, 255), 2, cv::LINE_AA);
}