#include <atomic>
#include <signal.h>
#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <metavision/sdk/base/utils/log.h>

#include "timing_profiler.h"
#include "event_processor.h"
#include "yolo_inference.h"
#include "visualization.h"

namespace po = boost::program_options;

namespace {
    std::atomic<bool> signal_caught{false};
    
    void signalHandler(int s) {
        MV_LOG_TRACE() << "Interrupt signal received." << std::endl;
        signal_caught = true;
    }
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);
    
    // === Parse Command Line Arguments ===
    std::string serial;
    std::string event_file_path;
    std::string biases_file;
    std::string model_file_path;
    std::string out_file_path;
    float conf_threshold;
    int process_interval_ms;
    bool enable_timing;
    
    po::options_description options_desc("Options");
    options_desc.add_options()
        ("help,h", "Produce help message")
        ("serial,s", po::value<std::string>(&serial), 
         "Serial ID of the camera")
        ("input-event-file,i", po::value<std::string>(&event_file_path),
         "Path to input event file (RAW, DAT or HDF5)")
        ("biases,b", po::value<std::string>(&biases_file),
         "Path to a biases file")
        ("model-file,m", po::value<std::string>(&model_file_path)->required(),
         "Path to YOLO model file")
        ("output-file,o", po::value<std::string>(&out_file_path)
         ->default_value("data.raw"),
         "Path to output file for recording")
        ("conf,c", po::value<float>(&conf_threshold)->default_value(0.3f),
         "Confidence threshold for detections")
        ("interval", po::value<int>(&process_interval_ms)->default_value(100),
         "Processing interval in milliseconds")
        ("enable-timing,t", po::value<bool>(&enable_timing)->default_value(true),
         "Enable detailed timing profiler");
    
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        
        if (vm.count("help")) {
            std::cout << "Real-time Event-based Object Detection System\n";
            std::cout << options_desc << "\n";
            return 0;
        }
        
        po::notify(vm);
    } catch (po::error& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cout << options_desc << "\n";
        return 1;
    }
    
    // === Initialize Components ===
    MV_LOG_INFO() << "Initializing system...";
    
    // 1. Event Processor
    EventProcessor event_processor(320, 320, 50000);
    if (!event_processor.initializeCamera(serial, event_file_path, biases_file)) {
        MV_LOG_ERROR() << "Failed to initialize event camera";
        return -1;
    }
    
    // 2. YOLO Inference
    YOLOInference yolo(model_file_path);
    if (!yolo.initialize()) {
        MV_LOG_ERROR() << "Failed to initialize YOLO";
        return -1;
    }
    
    // 3. Visualizer
    Visualizer visualizer;
    
    // 4. Start event processing
    if (!event_processor.start(20)) {
        MV_LOG_ERROR() << "Failed to start event processing";
        return -1;
    }
    
    MV_LOG_INFO() << "System initialized successfully";
    MV_LOG_INFO() << "Press 'q' or ESC to quit";
    MV_LOG_INFO() << "Press SPACE to toggle recording";
    
    // === Main Processing Loop ===
    int frame_count = 0;
    bool recording = false;
    auto last_process_time = std::chrono::system_clock::now();
    
    cv::Mat display_frame;
    Metavision::timestamp frame_ts = 0;
    
    MV_LOG_INFO() << "Entering main processing loop...";
    
    while (!signal_caught && event_processor.isRunning()) {
        MV_LOG_INFO() << "Entered main loop...";
        // Check if it's time to process
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_process_time).count();
        MV_LOG_INFO() << "time computed";
        
        if (elapsed < process_interval_ms) {
            MV_LOG_INFO() << "1st IF in ";
            cv::waitKey(1);
            MV_LOG_INFO() << "1st IF out ";
            continue;
        }
         MV_LOG_INFO() << "a ";
        last_process_time = now;
         MV_LOG_INFO() << "a ";
        
        // Get latest event frame
        if (!event_processor.getLatestFrame(display_frame, frame_ts)) {
            MV_LOG_INFO() << "2nd IF in ";
            cv::waitKey(1);
            MV_LOG_INFO() << "2nd IF out ";
            continue;
        }
        MV_LOG_INFO() << "b ";
        
        // Validate frame
        if (display_frame.empty() || display_frame.rows == 0 || display_frame.cols == 0) {
            MV_LOG_WARNING() << "Empty frame received, skipping...";
            cv::waitKey(1);
            continue;
        }
        MV_LOG_INFO() << "c ";
        frame_count++;
        
        // Clone frame for processing (avoid any memory issues)
        cv::Mat processing_frame = display_frame.clone();
        MV_LOG_INFO() << "d ";
        
        std::vector<YOLOInference::DetectionBox> detections;
        MV_LOG_INFO() << "e ";
        // Run YOLO inference with timing
        {
            TIME_SCOPE_ENABLED("TOTAL_FRAME_LATENCY", enable_timing);
            
            if (!yolo.infer(processing_frame, detections, conf_threshold)) {
                MV_LOG_ERROR() << "Inference failed on frame " << frame_count;
                continue;
            }
        }
        
        // Visualize results
        visualizer.drawDetections(processing_frame, detections);
        
        // Calculate FPS
        static auto fps_start = std::chrono::system_clock::now();
        static int fps_frames = 0;
        static double current_fps = 0.0;
        fps_frames++;
        
        auto fps_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - fps_start).count();
        if (fps_elapsed >= 1000) {
            current_fps = fps_frames * 1000.0 / fps_elapsed;
            fps_frames = 0;
            fps_start = now;
        }
        
        visualizer.drawFrameInfo(processing_frame, frame_ts, current_fps);
        
        // Print timing for this frame
        if (enable_timing && frame_count % 10 == 0) {
            TimingProfiler::getInstance().printLastFrame(frame_count);
        }
        
        // Display
        try {
            cv::imshow("Event-based YOLO Detection", processing_frame);
        } catch (const cv::Exception& e) {
            MV_LOG_ERROR() << "Display error: " << e.what();
        }
        
        // Handle keyboard input
        int key = cv::waitKey(1);
        switch (key) {
            case 'q':
            case 27: // ESC
                signal_caught = true;
                break;
                
            case ' ': // SPACE - toggle recording
                if (!recording) {
                    MV_LOG_INFO() << "Started recording to " << out_file_path;
                    event_processor.getCamera().start_recording(out_file_path);
                } else {
                    MV_LOG_INFO() << "Stopped recording";
                    event_processor.getCamera().stop_recording(out_file_path);
                }
                recording = !recording;
                break;
                
            case 'r': // Reset timing stats
                TimingProfiler::getInstance().reset();
                MV_LOG_INFO() << "Timing statistics reset";
                break;
        }
    }
    
    // === Cleanup ===
    MV_LOG_INFO() << "Shutting down...";
    
    event_processor.stop();
    
    if (enable_timing) {
        TimingProfiler::getInstance().printSummary();
    }
    
    cv::destroyAllWindows();
    
    MV_LOG_INFO() << "Processed " << frame_count << " frames";
    MV_LOG_INFO() << "Shutdown complete";
    
    return 0;
}