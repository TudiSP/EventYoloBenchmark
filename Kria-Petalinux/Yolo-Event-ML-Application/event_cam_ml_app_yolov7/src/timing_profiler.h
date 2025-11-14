#ifndef TIMING_PROFILER_H
#define TIMING_PROFILER_H

#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <iomanip>
#include <mutex>

class TimingProfiler {
public:
    struct TimingStats {
        double total_ms = 0.0;
        double min_ms = std::numeric_limits<double>::max();
        double max_ms = 0.0;
        int count = 0;
        
        void update(double ms) {
            total_ms += ms;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
            count++;
        }
        
        double get_avg() const {
            return count > 0 ? total_ms / count : 0.0;
        }
    };
    
    static TimingProfiler& getInstance() {
        static TimingProfiler instance;
        return instance;
    }
    
    class ScopedTimer {
    public:
        ScopedTimer(const std::string& name, bool enabled = true) 
            : name_(name), enabled_(enabled) {
            if (enabled_) {
                start_ = std::chrono::high_resolution_clock::now();
            }
        }
        
        ~ScopedTimer() {
            if (enabled_) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double, std::milli>(end - start_).count();
                TimingProfiler::getInstance().recordTiming(name_, duration);
            }
        }
        
    private:
        std::string name_;
        bool enabled_;
        std::chrono::high_resolution_clock::time_point start_;
    };
    
    void recordTiming(const std::string& name, double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        timings_[name].update(ms);
        last_frame_timings_[name] = ms;
    }
    
    void printLastFrame(int frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        double total = 0.0;
        
        std::cout << "\n========== Frame " << frame_id << " Timing Breakdown ==========\n";
        std::cout << std::fixed << std::setprecision(2);
        
        for (const auto& pair : last_frame_timings_) {
            total += pair.second;
        }
        
        for (const auto& pair : last_frame_timings_) {
            double percentage = (total > 0) ? (pair.second / total * 100.0) : 0.0;
            std::cout << "  " << std::setw(35) << std::left << pair.first 
                      << ": " << std::setw(8) << std::right << pair.second 
                      << " ms (" << std::setw(5) << percentage << "%)\n";
        }
        std::cout << "  " << std::setw(35) << std::left << "TOTAL" 
                  << ": " << std::setw(8) << std::right << total << " ms\n";
        std::cout << "================================================\n";
    }
    
    void printSummary() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n================ LATENCY SUMMARY ================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(35) << std::left << "Stage"
                  << std::setw(10) << std::right << "Mean"
                  << std::setw(10) << "Min"
                  << std::setw(10) << "Max"
                  << std::setw(10) << "Count\n";
        std::cout << std::string(75, '-') << "\n";
        
        double total_avg = 0.0;
        for (const auto& pair : timings_) {
            const auto& stats = pair.second;
            std::cout << std::setw(35) << std::left << pair.first
                      << std::setw(10) << std::right << stats.get_avg()
                      << std::setw(10) << stats.min_ms
                      << std::setw(10) << stats.max_ms
                      << std::setw(10) << stats.count << "\n";
            total_avg += stats.get_avg();
        }
        
        std::cout << std::string(75, '-') << "\n";
        std::cout << std::setw(35) << std::left << "TOTAL PIPELINE"
                  << std::setw(10) << std::right << total_avg << " ms\n";
        std::cout << "=================================================\n";
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        timings_.clear();
        last_frame_timings_.clear();
    }
    
private:
    TimingProfiler() = default;
    std::map<std::string, TimingStats> timings_;
    std::map<std::string, double> last_frame_timings_;
    std::mutex mutex_;
};

// Convenience macro for timing
#define TIME_SCOPE(name) TimingProfiler::ScopedTimer timer_##__LINE__(name)
#define TIME_SCOPE_ENABLED(name, enabled) TimingProfiler::ScopedTimer timer_##__LINE__(name, enabled)

#endif // TIMING_PROFILER_H