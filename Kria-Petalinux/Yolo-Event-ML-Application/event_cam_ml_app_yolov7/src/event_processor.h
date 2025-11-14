#ifndef EVENT_PROCESSOR_H
#define EVENT_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <metavision/sdk/driver/camera.h>
#include "cd_histo_frame_generator.h"
#include "timing_profiler.h"

class EventProcessor {
public:
    EventProcessor(int width, int height, uint32_t accumulation_time_us = 50000);
    ~EventProcessor();
    
    // Initialize event camera
    bool initializeCamera(const std::string& serial = "", 
                          const std::string& event_file = "",
                          const std::string& biases_file = "");
    
    // Start processing events
    bool start(uint16_t fps = 20);
    
    // Stop processing
    void stop();
    
    // Get latest frame (non-blocking)
    bool getLatestFrame(cv::Mat& frame, Metavision::timestamp& ts);
    
    // Check if camera is running
    bool isRunning();
    
    // Get camera reference (for configuration)
    Metavision::Camera& getCamera() { return camera_; }
    
    // Get frame generator (for configuration)
    Metavision::CDHistoFrameGenerator& getFrameGenerator() { return cd_frame_generator_; }
    
private:
    int width_;
    int height_;
    uint32_t accumulation_time_us_;
    
    Metavision::Camera camera_;
    Metavision::CDHistoFrameGenerator cd_frame_generator_;
    
    // Frame synchronization
    std::mutex frame_mutex_;
    cv::Mat current_frame_;
    Metavision::timestamp current_frame_ts_;
    
    int cd_events_cb_id_;
    bool camera_opened_;
};

#endif // EVENT_PROCESSOR_H