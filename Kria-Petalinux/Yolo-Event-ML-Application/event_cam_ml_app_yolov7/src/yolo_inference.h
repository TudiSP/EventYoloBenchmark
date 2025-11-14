#ifndef YOLO_INFERENCE_H
#define YOLO_INFERENCE_H

#include <opencv2/opencv.hpp>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include "common.h"
#include "timing_profiler.h"

class YOLOInference {
public:
    struct DetectionBox {
        float x, y, w, h;
        int class_id;
        float confidence;
        std::vector<float> class_scores;
    };
    
    YOLOInference(const std::string& model_path);
    ~YOLOInference();
    
    // Initialize the DPU
    bool initialize();
    
    // Run inference on a frame
    bool infer(const cv::Mat& input_frame, 
               std::vector<DetectionBox>& detections,
               float conf_threshold = 0.3f);
    
    // Get input dimensions
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }
    
private:
    // Preprocessing
    void preprocessFrame(const cv::Mat& frame, int8_t* data);
    
    // Postprocessing
    void postProcess(const cv::Mat& frame,
                    std::vector<int8_t*>& results,
                    std::vector<DetectionBox>& detections,
                    float conf_threshold);
    
    std::string model_path_;
    std::unique_ptr<xir::Graph> graph_;
    std::unique_ptr<vart::Runner> runner_;
    
    GraphInfo shapes_;
    TensorShape in_shapes_[1];
    TensorShape out_shapes_[3];
    
    int input_width_;
    int input_height_;
    float input_scale_;
    std::vector<float> output_scales_;
    
    // Buffers
    std::unique_ptr<int8_t[]> input_buffer_;
    std::unique_ptr<int8_t[]> output_buffers_[3];
    
    bool initialized_;
};

#endif // YOLO_INFERENCE_H