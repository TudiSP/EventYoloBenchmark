#include "event_processor.h"
#include <metavision/sdk/base/utils/log.h>

EventProcessor::EventProcessor(int width, int height, uint32_t accumulation_time_us)
    : width_(width), height_(height), 
      accumulation_time_us_(accumulation_time_us),
      cd_frame_generator_(width, height),
      cd_events_cb_id_(-1),
      camera_opened_(false),
      current_frame_ts_(0) {
    
    cd_frame_generator_.set_display_accumulation_time_us(accumulation_time_us_);
}

EventProcessor::~EventProcessor() {
    stop();
}

bool EventProcessor::initializeCamera(const std::string& serial,
                                      const std::string& event_file,
                                      const std::string& biases_file) {
    TIME_SCOPE("1.1_camera_initialization");
    
    try {
        if (!event_file.empty()) {
            // Open from file
            Metavision::FileConfigHints hints;
            camera_ = Metavision::Camera::from_file(event_file, hints);
        } else {
            // Open live camera
            if (!serial.empty()) {
                camera_ = Metavision::Camera::from_serial(serial);
            } else {
                camera_ = Metavision::Camera::from_first_available();
            }
            
            // Apply biases if provided
            if (!biases_file.empty()) {
                camera_.biases().set_from_file(biases_file);
            }
        }
        
        camera_opened_ = true;
        MV_LOG_INFO() << "Camera initialized successfully";
        return true;
        
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << "Camera initialization failed: " << e.what();
        return false;
    }
}

bool EventProcessor::start(uint16_t fps) {
    if (!camera_opened_) {
        MV_LOG_ERROR() << "Camera not initialized";
        return false;
    }
    
    TIME_SCOPE("1.2_event_callback_setup");
    
    // Setup frame callback
    cd_frame_generator_.start(fps, 
        [this](const Metavision::timestamp &ts, const cv::Mat &frame) {
            TIME_SCOPE("2_event_frame_generation");
            std::unique_lock<std::mutex> lock(frame_mutex_);
            current_frame_ts_ = ts;
            frame.copyTo(current_frame_);
        });
    
    // Setup event callback
    cd_events_cb_id_ = camera_.cd().add_callback(
        [this](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            TIME_SCOPE("1_event_acquisition");
            cd_frame_generator_.add_events(ev_begin, ev_end);
        });
    
    // Start camera
    try {
        camera_.start();
        MV_LOG_INFO() << "Event processing started";
        return true;
    } catch (const Metavision::CameraException &e) {
        MV_LOG_ERROR() << "Failed to start camera: " << e.what();
        return false;
    }
}

void EventProcessor::stop() {
    if (cd_events_cb_id_ >= 0) {
        camera_.cd().remove_callback(cd_events_cb_id_);
        cd_events_cb_id_ = -1;
    }
    
    cd_frame_generator_.stop();
    
    if (camera_opened_) {
        try {
            camera_.stop();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << "Error stopping camera: " << e.what();
        }
    }
}

bool EventProcessor::getLatestFrame(cv::Mat& frame, Metavision::timestamp& ts) {
    std::unique_lock<std::mutex> lock(frame_mutex_);
    
    if (current_frame_.empty()) {
        return false;
    }
    
    current_frame_.copyTo(frame);
    ts = current_frame_ts_;
    return true;
}

bool EventProcessor::isRunning() {
    return camera_opened_ && camera_.is_running();
}