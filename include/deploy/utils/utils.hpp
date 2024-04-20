#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "deploy/core/macro.hpp"

namespace deploy {

/**
 * @brief Load the contents of a binary file into a vector of characters.
 *
 * @param filePath The path to the file to be loaded.
 * @return std::vector<char> A vector containing the contents of the file.
 * @throw std::runtime_error If there is an error opening or reading the file.
 */
std::vector<char> LoadFile(const std::string& filePath);

/**
 * @brief Generates pairs of labels and corresponding colors.
 *
 * This function reads labels from a file and generates pairs of labels
 * along with randomly generated colors in the form of OpenCV Scalars.
 *
 * @param label_file The path to the file containing the labels.
 * @return A vector of pairs, where each pair consists of a label (string) and a color (cv::Scalar).
 */
DEPLOY_DECL std::vector<std::pair<std::string, cv::Scalar>> GenerateLabelColorParis(const std::string& label_file);

/**
 * @brief Base class for timers.
 */
class TimerBase {
public:
    TimerBase()                            = default;
    TimerBase(const TimerBase&)            = default;
    TimerBase(TimerBase&&)                 = delete;
    TimerBase& operator=(const TimerBase&) = default;
    TimerBase& operator=(TimerBase&&)      = delete;
    virtual ~TimerBase()                   = default;

    /**
     * @brief Starts the timer.
     */
    virtual void Start() {
    }

    /**
     * @brief Stops the timer.
     */
    virtual void Stop() {
    }

    /**
     * @brief Get the elapsed time in microseconds.
     *
     * @return float Elapsed time in microseconds.
     */
    [[nodiscard]] float Microseconds() const noexcept {
        return GetMilliseconds() * MICROSECONDS_PER_MILLISECOND;
    }

    /**
     * @brief Get the elapsed time in milliseconds.
     *
     * @return float Elapsed time in milliseconds.
     */
    [[nodiscard]] float Milliseconds() const noexcept {
        return GetMilliseconds();
    }

    /**
     * @brief Get the elapsed time in seconds.
     *
     * @return float Elapsed time in seconds.
     */
    [[nodiscard]] float Seconds() const noexcept {
        return GetMilliseconds() / MILLISECONDS_PER_SECOND;
    }

    /**
     * @brief Resets the timer.
     */
    void Reset() noexcept {
        SetMilliseconds(0.0F);
    }

protected:
    /**
     * @brief Get the elapsed time in milliseconds.
     *
     * @return float Elapsed time in milliseconds.
     */
    [[nodiscard]] float GetMilliseconds() const noexcept {
        return m_milliseconds;
    }

    /**
     * @brief Set the elapsed time in milliseconds.
     *
     * @param value Elapsed time in milliseconds.
     */
    void SetMilliseconds(float value) noexcept {
        m_milliseconds = value;
    }

private:
    static constexpr float MICROSECONDS_PER_MILLISECOND = 1000.0F; /**< Conversion factor for microseconds to milliseconds */
    static constexpr float MILLISECONDS_PER_SECOND      = 1000.0F; /**< Conversion factor for milliseconds to seconds */
    float                  m_milliseconds{0.0F};                   /**< Elapsed time in milliseconds */
};

/**
 * @brief Class for GPU timer.
 */
class DEPLOY_DECL GpuTimer : public TimerBase {
public:
    /**
     * @brief Constructor.
     *
     * @param stream CUDA stream to associate with the timer.
     */
    explicit GpuTimer(cudaStream_t stream = nullptr);

    GpuTimer(const GpuTimer&)            = default;
    GpuTimer(GpuTimer&&)                 = delete;
    GpuTimer& operator=(const GpuTimer&) = default;
    GpuTimer& operator=(GpuTimer&&)      = delete;

    /**
     * @brief Destructor.
     */
    ~GpuTimer() override;

    /**
     * @brief Starts the GPU timer.
     */
    void Start() override;

    /**
     * @brief Stops the GPU timer.
     */
    void Stop() override;

private:
    cudaEvent_t  m_start{}; /**< CUDA event for start */
    cudaEvent_t  m_stop{};  /**< CUDA event for stop */
    cudaStream_t m_stream;  /**< CUDA stream */
};

/**
 * @brief Class for CPU timer.
 * @tparam Clock Type of clock to use for timing (e.g., std::chrono::high_resolution_clock).
 */
template <typename Clock = std::chrono::high_resolution_clock>
class DEPLOY_DECL CpuTimer : public TimerBase {
public:
    /**
     * @brief Starts the CPU timer.
     */
    void Start() override {
        m_start = Clock::now();
    }

    /**
     * @brief Stops the CPU timer and calculates the elapsed time.
     */
    void Stop() override {
        m_stop                                            = Clock::now();
        std::chrono::duration<float, std::milli> duration = m_stop - m_start;
        SetMilliseconds(GetMilliseconds() + duration.count());
    }

private:
    typename Clock::time_point m_start; /**< Start time */
    typename Clock::time_point m_stop;  /**< Stop time */
};

}  // namespace deploy
