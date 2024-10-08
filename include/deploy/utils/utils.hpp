#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
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
std::vector<char> loadFile(const std::string& filePath);

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
    virtual void start() {
    }

    /**
     * @brief Stops the timer.
     */
    virtual void stop() {
    }

    /**
     * @brief Get the elapsed time in microseconds.
     *
     * @return float Elapsed time in microseconds.
     */
    [[nodiscard]] float microseconds() const noexcept {
        return getMilliseconds() * microsecondsPerMillisecond;
    }

    /**
     * @brief Get the elapsed time in milliseconds.
     *
     * @return float Elapsed time in milliseconds.
     */
    [[nodiscard]] float milliseconds() const noexcept {
        return getMilliseconds();
    }

    /**
     * @brief Get the elapsed time in seconds.
     *
     * @return float Elapsed time in seconds.
     */
    [[nodiscard]] float seconds() const noexcept {
        return getMilliseconds() / millisecondsPerSecond;
    }

    /**
     * @brief Resets the timer.
     */
    void reset() noexcept {
        setMilliseconds(0.0F);
    }

protected:
    /**
     * @brief Get the elapsed time in milliseconds.
     *
     * @return float Elapsed time in milliseconds.
     */
    [[nodiscard]] float getMilliseconds() const noexcept {
        return mMilliseconds;
    }

    /**
     * @brief Set the elapsed time in milliseconds.
     *
     * @param value Elapsed time in milliseconds.
     */
    void setMilliseconds(float value) noexcept {
        mMilliseconds = value;
    }

private:
    static constexpr float microsecondsPerMillisecond = 1000.0F; /**< Conversion factor for microseconds to milliseconds */
    static constexpr float millisecondsPerSecond      = 1000.0F; /**< Conversion factor for milliseconds to seconds */
    float                  mMilliseconds{0.0F};                  /**< Elapsed time in milliseconds */
};

/**
 * @brief Class for GPU timer.
 */
class DEPLOYAPI GpuTimer : public TimerBase {
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
    void start() override;

    /**
     * @brief Stops the GPU timer.
     */
    void stop() override;

private:
    cudaEvent_t  mStart{}; /**< CUDA event for start */
    cudaEvent_t  mStop{};  /**< CUDA event for stop */
    cudaStream_t mStream;  /**< CUDA stream */
};

/**
 * @brief Class for CPU timer using high resolution clock.
 */
class DEPLOYAPI CpuTimer : public TimerBase {
public:
    /**
     * @brief Starts the CPU timer.
     */
    void start() override {
        mStart = Clock::now();
    }

    /**
     * @brief Stops the CPU timer and calculates the elapsed time.
     */
    void stop() override {
        mStop                                             = Clock::now();
        std::chrono::duration<float, std::milli> duration = mStop - mStart;
        setMilliseconds(getMilliseconds() + duration.count());
    }

private:
    using Clock = std::chrono::high_resolution_clock;
    typename Clock::time_point mStart; /**< Start time */
    typename Clock::time_point mStop;  /**< Stop time */
};

}  // namespace deploy
