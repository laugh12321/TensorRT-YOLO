#pragma once

#include <stdexcept>
#include <vector>

#include "deploy/core/macro.hpp"

namespace deploy {

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
    float theta;  /**< Rotation angle of the bounding box, in radians, measured clockwise from the positive x-axis */
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

}  // namespace deploy
