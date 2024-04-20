#pragma once

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
            num     = other.num;
            boxes   = other.boxes;
            classes = other.classes;
            scores  = other.scores;
        }
        return *this;
    }
};

}  // namespace deploy
