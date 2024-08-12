#pragma once

#include <stdexcept>
#include <vector>

#include "deploy/core/macro.hpp"

namespace deploy {

/**
 * @brief Represents a bounding box.
 */
struct DEPLOY_DECL Box {
    float left;   /**< Left coordinate of the bounding box */
    float top;    /**< Top coordinate of the bounding box */
    float right;  /**< Right coordinate of the bounding box */
    float bottom; /**< Bottom coordinate of the bounding box */
    float theta;  /**< Rotation angle of the bounding box, in radians, measured clockwise from the positive x-axis */
};

/**
 * @brief Represents the result of object detection.
 */
struct DEPLOY_DECL DetectionResult {
    int                num = 0;   /**< Number of detected objects */
    std::vector<Box>   boxes{};   /**< Detected bounding boxes */
    std::vector<int>   classes{}; /**< Detected classes */
    std::vector<float> scores{};  /**< Detection scores */

    /**
     * @brief Copy assignment operator.
     *
     * @param other The source DetectionResult to copy from.
     * @return DetectionResult& Reference to the assigned DetectionResult.
     */
    DetectionResult& operator=(const DetectionResult& other) {
        if (this != &other) {
            num     = other.num;     /**< Assign number of detected objects */
            boxes   = other.boxes;   /**< Assign detected bounding boxes */
            classes = other.classes; /**< Assign detected classes */
            scores  = other.scores;  /**< Assign detection scores */
        }
        return *this;
    }
};

/**
 * @brief Represents an image.
 */
struct DEPLOY_DECL Image {
    void* rgbPtr = nullptr; /**< Pointer to image data (BGR format) */
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

}  // namespace deploy
