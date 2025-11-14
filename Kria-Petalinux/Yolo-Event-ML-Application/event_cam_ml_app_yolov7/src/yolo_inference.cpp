#include "yolo_inference.h"
#include "utils.h"
#include <glog/logging.h>

#define NMS_THRESHOLD 0.1f

// Map histogram frame value to 0-1
inline float map_to_0_1_with_step_size(float original_value) {
    float value[6] = {0, 0.2, 0.4, 0.6, 0.8, 1};
    float segment_width = 0.8;
    int segment_index = int(original_value / segment_width);
    if (segment_index > 5) return 1;
    return value[segment_index];
}

YOLOInference::YOLOInference(const std::string& model_path)
    : model_path_(model_path), initialized_(false) {
    
    shapes_.inTensorList = in_shapes_;
    shapes_.outTensorList = out_shapes_;
}

YOLOInference::~YOLOInference() {
    // Cleanup handled by smart pointers
}

bool YOLOInference::initialize() {
    TIME_SCOPE("3.1_dpu_initialization");
    
    try {
        // Load model
        graph_ = xir::Graph::deserialize(model_path_);
        auto subgraph = get_dpu_subgraph(graph_.get());
        
        CHECK_EQ(subgraph.size(), 1u) 
            << "yolov7 should have one and only one dpu subgraph.";
        
        // Create runner
        runner_ = vart::Runner::create_runner(subgraph[0], "run");
        
        // Get tensor shapes
        auto inputTensors = runner_->get_input_tensors();
        auto outputTensors = runner_->get_output_tensors();
        
        int inputCnt = inputTensors.size();
        int outputCnt = outputTensors.size();
        
        getTensorShape(runner_.get(), &shapes_, inputCnt, outputCnt);
        
        input_width_ = shapes_.inTensorList[0].width;
        input_height_ = shapes_.inTensorList[0].height;
        
        // Get scales
        input_scale_ = get_input_scale(inputTensors[0]);
        
        for (int io = 0; io < 3; io++) {
            output_scales_.push_back(
                get_output_scale(outputTensors[shapes_.output_mapping[io]]));
        }
        
        // Allocate buffers
        size_t input_size = shapes_.inTensorList[0].size;
        input_buffer_.reset(new int8_t[input_size]);
        std::memset(input_buffer_.get(), 0, input_size);
        
        for (int i = 0; i < 3; i++) {
            size_t output_size = shapes_.outTensorList[i].size;
            output_buffers_[i].reset(new int8_t[output_size]);
            std::memset(output_buffers_[i].get(), 0, output_size);
        }
        
        LOG(INFO) << "Allocated input buffer: " << input_size << " bytes";
        LOG(INFO) << "Input dimensions: " << input_width_ << "x" << input_height_;
        
        initialized_ = true;
        LOG(INFO) << "YOLO DPU initialized successfully";
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to initialize YOLO: " << e.what();
        return false;
    }
}

void YOLOInference::preprocessFrame(const cv::Mat& frame, int8_t* data) {
    TIME_SCOPE("3.2_preprocessing");
    
    if (frame.empty()) {
        throw std::runtime_error("Empty frame in preprocessing");
    }
    
    // Ensure frame is in correct format (3 channel)
    cv::Mat working_frame;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, working_frame, cv::COLOR_GRAY2BGR);
    } else {
        working_frame = frame;
    }
    
    cv::Mat resized;
    cv::resize(working_frame, resized, cv::Size(input_width_, input_height_), 
               0, 0, cv::INTER_LINEAR);
    
    if (resized.empty() || resized.rows != input_height_ || resized.cols != input_width_) {
        throw std::runtime_error("Resize failed or produced invalid dimensions");
    }
    
    int l = 0;
    for (int h = 0; h < input_height_; h++) {
        for (int w = 0; w < input_width_; w++) {
            for (int c = 0; c < 2; c++) {
                float normalized = map_to_0_1_with_step_size(
                    resized.at<cv::Vec3b>(h, w)[c] / 17.0) / 255.0;
                data[l++] = (int8_t)(normalized * input_scale_);
            }
        }
    }
}

bool YOLOInference::infer(const cv::Mat& input_frame,
                          std::vector<DetectionBox>& detections,
                          float conf_threshold) {
    if (!initialized_) {
        LOG(ERROR) << "YOLO not initialized";
        return false;
    }
    
    if (input_frame.empty()) {
        LOG(ERROR) << "Empty input frame";
        return false;
    }
    
    // Clear previous detections
    detections.clear();
    
    // Preprocessing
    try {
        preprocessFrame(input_frame, input_buffer_.get());
    } catch (const std::exception& e) {
        LOG(ERROR) << "Preprocessing failed: " << e.what();
        return false;
    }
    
    // Create tensor buffers
    auto inputTensors = cloneTensorBuffer(runner_->get_input_tensors());
    auto outputTensors = cloneTensorBuffer(runner_->get_output_tensors());
    
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    
    try {
        inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            input_buffer_.get(), inputTensors[0].get()));
        
        for (int i = 0; i < 3; i++) {
            outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
                output_buffers_[i].get(), 
                outputTensors[shapes_.output_mapping[i]].get()));
        }
        
        inputsPtr.push_back(inputs[0].get());
        outputsPtr.resize(3);
        for (int i = 0; i < 3; i++) {
            outputsPtr[shapes_.output_mapping[i]] = outputs[i].get();
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Tensor buffer creation failed: " << e.what();
        return false;
    }
    
    // DPU Execution
    try {
        TIME_SCOPE("3.3_dpu_execution");
        auto job_id = runner_->execute_async(inputsPtr, outputsPtr);
        runner_->wait(job_id.first, -1);
    } catch (const std::exception& e) {
        LOG(ERROR) << "DPU execution failed: " << e.what();
        return false;
    }
    
    // Postprocessing
    std::vector<int8_t*> results;
    for (int i = 0; i < 3; i++) {
        results.push_back(output_buffers_[i].get());
    }
    
    try {
        postProcess(input_frame, results, detections, conf_threshold);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Postprocessing failed: " << e.what();
        return false;
    }
    
    return true;
}

void YOLOInference::postProcess(const cv::Mat& frame,
                                std::vector<int8_t*>& results,
                                std::vector<DetectionBox>& detections,
                                float conf_threshold) {
    TIME_SCOPE("3.4_postprocessing");
    
    std::vector<std::vector<float>> boxes;
    
    // Detect boxes from all output layers
    for (int ii = 0; ii < 3; ii++) {
        int width = shapes_.outTensorList[ii].width;
        int height = shapes_.outTensorList[ii].height;
        int channel = shapes_.outTensorList[ii].channel;
        
        detect_new(boxes, results[ii], channel, height, width, ii,
                   input_height_, input_width_, 
                   output_scales_[ii], conf_threshold);
    }
    
    // Correct box coordinates
    correct_region_boxes(boxes, boxes.size(), 320, 320,
                        frame.cols, frame.rows);
    
    // Apply NMS
    std::vector<std::vector<float>> res;
    std::vector<float> scores(boxes.size());
    
    for (int k = 0; k < classificationCnt; k++) {
        std::transform(boxes.begin(), boxes.end(), scores.begin(), 
            [k](auto& box) {
                box[4] = k;
                return box[6 + k];
            });
        
        std::vector<size_t> result_k;
        applyNMS_new(boxes, scores, NMS_THRESHOLD, conf_threshold, result_k);
        
        std::transform(result_k.begin(), result_k.end(), 
            std::back_inserter(res),
            [&boxes](auto& k) { return boxes[k]; });
    }
    
    // Convert to DetectionBox format
    float h = frame.rows;
    float w = frame.cols;
    
    for (const auto& r : res) {
        if (r[r[4] + 6] > conf_threshold) {
            DetectionBox box;
            box.x = r[0];
            box.y = r[1];
            box.w = r[2];
            box.h = r[3];
            box.class_id = static_cast<int>(r[4]);
            box.confidence = r[r[4] + 6];
            
            for (int i = 0; i < classificationCnt; i++) {
                box.class_scores.push_back(r[6 + i]);
            }
            
            detections.push_back(box);
        }
    }
}