project(event_yolo_refactored)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(unilog REQUIRED)
find_package(Threads REQUIRED)

include_directories(./)

# Define source files for modular components
set(TIMING_SOURCES
    timing_profiler.h
)

set(EVENT_PROCESSOR_SOURCES
    event_processor.h
    event_processor.cpp
)

set(YOLO_INFERENCE_SOURCES
    yolo_inference.h
    yolo_inference.cpp
)

set(VISUALIZATION_SOURCES
    visualization.h
    visualization.cpp
)

# Original supporting files
set(SUPPORT_SOURCES
    cd_histo_frame_generator.cpp
    cd_histo_frame_generator.h
    periodic_histo_frame_generation_algorithm.cpp
    periodic_histo_frame_generation_algorithm.h
    common.cpp
    common.h
    utils.h
)

# Build refactored executable
add_executable(event_yolo_refactored
    main_refactored.cpp
    ${TIMING_SOURCES}
    ${EVENT_PROCESSOR_SOURCES}
    ${YOLO_INFERENCE_SOURCES}
    ${VISUALIZATION_SOURCES}
    ${SUPPORT_SOURCES}
)

target_link_libraries(event_yolo_refactored
    MetavisionSDK::core
    MetavisionSDK::driver
    MetavisionSDK::ui
    ${OpenCV_LIBS}
    glog
    xir
    Threads::Threads
    vart-runner
    Boost::program_options
)

# Optional: Build original for comparison
add_executable(metavision_histo_viewer_original
    metavision_histo_viewer.cpp
    ${SUPPORT_SOURCES}
)

target_link_libraries(metavision_histo_viewer_original
    MetavisionSDK::core
    MetavisionSDK::driver
    MetavisionSDK::ui
    ${OpenCV_LIBS}
    glog
    xir
    Threads::Threads
    vart-runner
    Boost::program_options
)