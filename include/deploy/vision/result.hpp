#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "deploy/core/macro.hpp"

namespace deploy {

/**
 * @brief Represents an image.
 */
struct DEPLOYAPI Image {
    void* rgbPtr = nullptr; /**< Pointer to image data (uint8, RGB format) */
    int   width  = 0;       /**< Width of the image */
    int   height = 0;       /**< Height of the image */

    // Default constructor
    // constexpr Image() : rgbPtr(nullptr), width(0), height(0) {}
    Image() = default;

    /**
     * @brief Parameterized constructor with boundary checks.
     *
     * @param rgbPtr Pointer to image data.
     * @param width Width of the image.
     * @param height Height of the image.
     * @throws std::invalid_argument If width or height is negative.
     */
    Image(void* rgbPtr, int width, int height)
        : rgbPtr(rgbPtr), width(width), height(height) {
        if (width < 0 || height < 0) {
            throw std::invalid_argument("Width and height must be non-negative");
        }
    }
};

/**
 * @brief Represents a mask image.
 */
struct DEPLOYAPI Mask {
    std::vector<uint8_t> data;       /**< Pointer to image data (uint8, grayscale format) */
    int                  width  = 0; /**< Width of the image */
    int                  height = 0; /**< Height of the image */

    // Default constructor
    Mask() = default;

    /**
     * @brief Parameterized constructor with boundary checks.
     *
     * @param width Width of the image.
     * @param height Height of the image.
     * @throws std::invalid_argument If width or height is negative.
     */
    Mask(int width, int height) : width(width), height(height) {
        if (width < 0 || height < 0) {
            throw std::invalid_argument("Width and height must be non-negative");
        }
        // Resize the data vector to hold width * height elements
        data.resize(width * height);
    }

    // Copy constructor
    Mask(const Mask& other)
        : width(other.width), height(other.height), data(other.data) {
    }

    // Copy assignment operator
    Mask& operator=(const Mask& other) {
        if (this != &other) {
            width  = other.width;
            height = other.height;
            data   = other.data;  // Copy the data vector
        }
        return *this;
    }

    // Move constructor
    Mask(Mask&& other) noexcept
        : data(std::move(other.data)), width(other.width), height(other.height) {
        other.width  = 0;
        other.height = 0;
    }

    // Move assignment operator
    Mask& operator=(Mask&& other) noexcept {
        if (this != &other) {
            data         = std::move(other.data);
            width        = other.width;
            height       = other.height;
            other.width  = 0;
            other.height = 0;
        }
        return *this;
    }
};

/**
 * @brief Represents a key point.
 */
struct KeyPoint {
    float                x;     // x-coordinate of the key point
    float                y;     // y-coordinate of the key point
    std::optional<float> conf;  // Optional confidence value of the key point

    // Default constructor
    KeyPoint() = default;

    /**
     * @brief Parameterized constructor with boundary checks.
     *
     * @param x x-coordinate of the key point.
     * @param y y-coordinate of the key point.
     * @param conf Optional confidence value of the key point (default is std::nullopt).
     */
    KeyPoint(float x, float y, std::optional<float> conf = std::nullopt)
        : x(x), y(y), conf(conf) {}

    // Copy constructor
    KeyPoint(const KeyPoint& other)
        : x(other.x), y(other.y), conf(other.conf) {}

    // Copy assignment operator
    KeyPoint& operator=(const KeyPoint& other) {
        if (this != &other) {  // Check for self-assignment
            x    = other.x;
            y    = other.y;
            conf = other.conf;
        }
        return *this;
    }

    // Move constructor
    KeyPoint(KeyPoint&& other) noexcept
        : x(std::move(other.x)), y(std::move(other.y)), conf(std::move(other.conf)) {}

    // Move assignment operator
    KeyPoint& operator=(KeyPoint&& other) noexcept {
        if (this != &other) {  // Check for self-assignment
            x    = std::move(other.x);
            y    = std::move(other.y);
            conf = std::move(other.conf);
        }
        return *this;
    }
};

/**
 * @brief Represents a bounding box.
 */
struct DEPLOYAPI Box {
    float left;   /**< Left coordinate of the bounding box */
    float top;    /**< Top coordinate of the bounding box */
    float right;  /**< Right coordinate of the bounding box */
    float bottom; /**< Bottom coordinate of the bounding box */
};

/**
 * @brief Represents an oriented bounding box (OBB) by extending Box with a rotation angle.
 */
struct DEPLOYAPI RotatedBox : public Box {
    float theta; /**< Rotation angle of the bounding box, in radians, measured clockwise from the positive x-axis */
};

/**
 * @brief Represents the result of object detection.
 */
struct DEPLOYAPI DetResult {
    int                num = 0;   /**< Number of detected objects */
    std::vector<Box>   boxes{};   /**< Detected bounding boxes */
    std::vector<int>   classes{}; /**< Detected classes */
    std::vector<float> scores{};  /**< Detection scores */

    /**
     * @brief Copy assignment operator.
     *
     * @param other The source DetResult to copy from.
     * @return DetResult& Reference to the assigned DetResult.
     */
    DetResult& operator=(const DetResult& other) {
        if (this != &other) {
            num     = other.num;
            boxes   = other.boxes;
            classes = other.classes;
            scores  = other.scores;
        }
        return *this;
    }
};

/**
 * @brief Represents the result of object detection with oriented bounding boxes.
 */
struct DEPLOYAPI OBBResult : public DetResult {
    std::vector<RotatedBox> boxes{}; /**< Detected oriented bounding boxes */

    /**
     * @brief Copy assignment operator.
     *
     * @param other The source OBBResult to copy from.
     * @return OBBResult& Reference to the assigned OBBResult.
     */
    OBBResult& operator=(const OBBResult& other) {
        if (this != &other) {
            DetResult::operator=(static_cast<const DetResult&>(other));
            boxes = other.boxes;
        }
        return *this;
    }
};

/**
 * @brief Represents the result of instance segmentation.
 */
struct DEPLOYAPI SegResult : public DetResult {
    std::vector<Mask> masks{}; /**< Detected object masks (binary, 0 for background, 1 for foreground) */

    /**
     * @brief Copy assignment operator.
     *
     * @param other The source SegResult to copy from.
     * @return SegResult& Reference to the assigned SegResult.
     */
    SegResult& operator=(const SegResult& other) {
        DetResult::operator=(static_cast<const DetResult&>(other));
        masks = other.masks;
        return *this;
    }
};

/**
 * @brief Represents the result of Pose Estimation.
 */
struct DEPLOYAPI PoseResult : public DetResult {
    std::vector<std::vector<KeyPoint>> kpts{}; /**< Container for detected key points. */

    /**
     * @brief Copy assignment operator.
     *
     * @param other The source PoseResult to copy from.
     * @return PoseResult& Reference to the assigned PoseResult.
     */
    PoseResult& operator=(const PoseResult& other) {
        DetResult::operator=(static_cast<const DetResult&>(other));
        kpts = other.kpts;
        return *this;
    }
};

}  // namespace deploy
