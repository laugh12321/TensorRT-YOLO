/**
 * @file result.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 定义数据类型
 * @date 2025-01-13
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "deploy/core/macro.hpp"

namespace deploy {

/**
 * @brief 图像结构体，用于存储图像数据及其尺寸信息
 */
struct DEPLOYAPI Image {
    void* ptr;         // < 图像数据指针
    int   width  = 0;  // < 图像宽度
    int   height = 0;  // < 图像高度

    /**
     * @brief 构造函数，初始化图像数据和尺寸
     *
     * @param data 图像数据指针
     * @param width 图像宽度
     * @param height 图像高度
     */
    Image(void* data, int width, int height) : ptr(data), width(width), height(height) {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width and height must be positive"));
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Image& img) {
        os << "Image(width=" << img.width << ", height=" << img.height << ", ptr=" << img.ptr << ")";
        return os;
    }
};

/**
 * @brief 掩码结构体，用于存储掩码数据及其尺寸信息
 */
struct DEPLOYAPI Mask {
    std::vector<uint8_t> data;        // < 掩码数据
    int                  width  = 0;  // < 掩码宽度
    int                  height = 0;  // < 掩码高度

    /**
     * @brief 默认构造函数
     *
     */
    Mask() = default;

    /**
     * @brief 构造函数，初始化掩码尺寸并分配数据空间
     *
     * @param width 掩码宽度
     * @param height 掩码高度
     */
    Mask(int width, int height) : width(width), height(height) {
        if (width < 0 || height < 0) {
            throw std::invalid_argument(MAKE_ERROR_MESSAGE("Mask: width and height must be positive"));
        }
        data.resize(width * height);
    }

    friend std::ostream& operator<<(std::ostream& os, const Mask& mask) {
        os << "Mask(width=" << mask.width << ", height=" << mask.height << ", data size=" << mask.data.size() << ")";
        return os;
    }

    Mask(const Mask& other)            = default;      // < 默认拷贝构造函数
    Mask& operator=(const Mask& other) = default;      // < 默认拷贝赋值运算符

    Mask(Mask&& other) noexcept            = default;  // < 默认移动构造函数
    Mask& operator=(Mask&& other) noexcept = default;  // < 默认移动赋值运算符
};

/**
 * @brief 关键点结构体，用于存储关键点的坐标和置信度
 */
struct DEPLOYAPI KeyPoint {
    float                x;     // < 关键点的 x 坐标
    float                y;     // < 关键点的 y 坐标
    std::optional<float> conf;  // < 关键点的置信度，可选

    /**
     * @brief 默认构造函数
     *
     */
    KeyPoint() = default;

    /**
     * @brief 构造函数，初始化关键点坐标和置信度
     *
     * @param x 关键点的 x 坐标
     * @param y 关键点的 y 坐标
     * @param conf 关键点的置信度，可选，默认为 std::nullopt
     */
    KeyPoint(float x, float y, std::optional<float> conf = std::nullopt)
        : x(x), y(y), conf(conf) {}

    friend std::ostream& operator<<(std::ostream& os, const KeyPoint& kp) {
        os << "KeyPoint(x=" << kp.x << ", y=" << kp.y;
        if (kp.conf) {
            os << ", conf=" << *kp.conf;
        }
        os << ")";
        return os;
    }

    KeyPoint(const KeyPoint& other)            = default;      // < 默认拷贝构造函数
    KeyPoint& operator=(const KeyPoint& other) = default;      // < 默认拷贝赋值运算符

    KeyPoint(KeyPoint&& other) noexcept            = default;  // < 默认移动构造函数
    KeyPoint& operator=(KeyPoint&& other) noexcept = default;  // < 默认移动赋值运算符
};

/**
 * @brief 矩形框结构体，用于存储矩形框的坐标信息
 */
struct DEPLOYAPI Box {
    float left;    // < 矩形框的左边界坐标
    float top;     // < 矩形框的上边界坐标
    float right;   // < 矩形框的右边界坐标
    float bottom;  // < 矩形框的下边界坐标

    /**
     * @brief 默认构造函数
     *
     */
    Box() = default;

    /**
     * @brief 构造函数，初始化矩形框的坐标
     *
     * @param left 矩形框的左边界坐标
     * @param top 矩形框的上边界坐标
     * @param right 矩形框的右边界坐标
     * @param bottom 矩形框的下边界坐标
     */
    Box(float left, float top, float right, float bottom)
        : left(left), top(top), right(right), bottom(bottom) {}

    friend std::ostream& operator<<(std::ostream& os, const Box& box) {
        os << "Box(left=" << box.left << ", top=" << box.top << ", right=" << box.right << ", bottom=" << box.bottom << ")";
        return os;
    }

    Box(const Box& other)            = default;      // < 默认拷贝构造函数
    Box& operator=(const Box& other) = default;      // < 默认拷贝赋值运算符

    Box(Box&& other) noexcept            = default;  // < 默认移动构造函数
    Box& operator=(Box&& other) noexcept = default;  // < 默认移动赋值运算符
};

/**
 * @brief 旋转矩形框结构体，继承自矩形框结构体，增加旋转角度信息
 */
struct DEPLOYAPI RotatedBox : public Box {
    float theta;  // < 旋转矩形框的旋转角度，以弧度为单位，顺时针旋转角度从正 x 轴开始测量

    /**
     * @brief 默认构造函数
     *
     */
    RotatedBox() = default;

    /**
     * @brief 构造函数，初始化旋转矩形框的坐标和旋转角度
     *
     * @param left 旋转矩形框的左边界坐标
     * @param top 旋转矩形框的上边界坐标
     * @param right 旋转矩形框的右边界坐标
     * @param bottom 旋转矩形框的下边界坐标
     * @param theta 旋转矩形框的旋转角度
     */
    RotatedBox(float left, float top, float right, float bottom, float theta)
        : Box(left, top, right, bottom), theta(theta) {}

    friend std::ostream& operator<<(std::ostream& os, const RotatedBox& rbox) {
        os << "RotatedBox(left=" << rbox.left << ", top=" << rbox.top << ", right=" << rbox.right << ", bottom=" << rbox.bottom << ", theta=" << rbox.theta << ")";
        return os;
    }

    RotatedBox(const RotatedBox& other)            = default;      // < 默认拷贝构造函数
    RotatedBox& operator=(const RotatedBox& other) = default;      // < 默认拷贝赋值运算符

    RotatedBox(RotatedBox&& other) noexcept            = default;  // < 默认移动构造函数
    RotatedBox& operator=(RotatedBox&& other) noexcept = default;  // < 默认移动赋值运算符
};

/**
 * @brief 基础结果结构体，用于存储检测结果的基础信息，如数量、类别和得分
 */
struct DEPLOYAPI BaseRes {
    int                num = 0;  // < 检测结果的数量
    std::vector<int>   classes;  // < 检测结果的类别
    std::vector<float> scores;   // < 检测结果的得分

    /**
     * @brief 默认构造函数
     *
     */
    BaseRes() = default;

    /**
     * @brief 构造函数，初始化检测结果的数量、类别和得分
     *
     * @param num 检测结果的数量
     * @param classes 检测结果的类别
     * @param scores 检测结果的得分
     */
    BaseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores)
        : num(num), classes(classes), scores(scores) {}

    BaseRes(const BaseRes& other)            = default;      // < 默认拷贝构造函数
    BaseRes& operator=(const BaseRes& other) = default;      // < 默认拷贝赋值运算符

    BaseRes(BaseRes&& other) noexcept            = default;  // < 默认移动构造函数
    BaseRes& operator=(BaseRes&& other) noexcept = default;  // < 默认移动赋值运算符
};

/**
 * @brief 分类结果结构体，继承自基础结果结构体，无需额外的构造函数
 */
struct DEPLOYAPI ClassifyRes : public BaseRes {
    // 无需额外的构造函数，继承自 BaseRes 的默认构造函数已经足够

    friend std::ostream& operator<<(std::ostream& os, const BaseRes& res) {
        os << "ClassifyRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "]\n)";
        return os;
    }
};

/**
 * @brief 检测结果结构体，继承自基础结果结构体，增加矩形框信息
 */
struct DEPLOYAPI DetectRes : public BaseRes {
    std::vector<Box> boxes;  // < 检测结果的矩形框

    /**
     * @brief 默认构造函数
     *
     */
    DetectRes() = default;

    /**
     * @brief 构造函数，初始化检测结果的数量、类别、得分和矩形框
     *
     * @param num 检测结果的数量
     * @param classes 检测结果的类别
     * @param scores 检测结果的得分
     * @param boxes 检测结果的矩形框
     */
    DetectRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes)
        : BaseRes(num, classes, scores), boxes(boxes) {}

    friend std::ostream& operator<<(std::ostream& os, const DetectRes& res) {
        os << "DetectRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "],\n    boxes=[\n";
        for (const auto& box : res.boxes) os << "        " << box << ",\n";
        os << "    ]\n)";
        return os;
    }

    DetectRes(const DetectRes& other)            = default;      // < 默认拷贝构造函数
    DetectRes& operator=(const DetectRes& other) = default;      // < 默认拷贝赋值运算符

    DetectRes(DetectRes&& other) noexcept            = default;  // < 默认移动构造函数
    DetectRes& operator=(DetectRes&& other) noexcept = default;  // < 默认移动赋值运算符
};

/**
 * @brief 旋转矩形框检测结果结构体，继承自基础结果结构体，增加旋转矩形框信息
 */
struct DEPLOYAPI OBBRes : public BaseRes {
    std::vector<RotatedBox> boxes;  // < 检测结果的旋转矩形框

    /**
     * @brief 默认构造函数
     *
     */
    OBBRes() = default;

    /**
     * @brief 构造函数，初始化检测结果的数量、类别、得分和旋转矩形框
     *
     * @param num 检测结果的数量
     * @param classes 检测结果的类别
     * @param scores 检测结果的得分
     * @param boxes 检测结果的旋转矩形框
     */
    OBBRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<RotatedBox>& boxes)
        : BaseRes(num, classes, scores), boxes(boxes) {}

    friend std::ostream& operator<<(std::ostream& os, const OBBRes& res) {
        os << "OBBRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "],\n    boxes=[\n";
        for (const auto& box : res.boxes) os << "        " << box << ",\n";
        os << "    ]\n)";
        return os;
    }

    OBBRes(const OBBRes& other)            = default;      // < 默认拷贝构造函数
    OBBRes& operator=(const OBBRes& other) = default;      // < 默认拷贝赋值运算符

    OBBRes(OBBRes&& other) noexcept            = default;  // < 默认移动构造函数
    OBBRes& operator=(OBBRes&& other) noexcept = default;  // < 默认移动赋值运算符
};

/**
 * @brief 分割结果结构体，继承自基础结果结构体，增加矩形框和掩码信息
 */
struct DEPLOYAPI SegmentRes : public BaseRes {
    std::vector<Box>  boxes;  // < 分割结果的矩形框
    std::vector<Mask> masks;  // < 分割结果的掩码

    /**
     * @brief 默认构造函数
     *
     */
    SegmentRes() = default;

    /**
     * @brief 构造函数，初始化分割结果的数量、类别、得分、矩形框和掩码
     *
     * @param num 分割结果的数量
     * @param classes 分割结果的类别
     * @param scores 分割结果的得分
     * @param boxes 分割结果的矩形框
     * @param masks 分割结果的掩码
     */
    SegmentRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<Mask>& masks)
        : BaseRes(num, classes, scores), boxes(boxes), masks(masks) {}

    friend std::ostream& operator<<(std::ostream& os, const SegmentRes& res) {
        os << "SegmentRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "],\n    boxes: [\n";
        for (const auto& box : res.boxes) os << "        " << box << ",\n";
        os << "],\n    masks: [\n";
        for (const auto& mask : res.masks) os << "        " << mask << "\n";
        os << "    ]\n)";
        return os;
    }

    SegmentRes(const SegmentRes& other)            = default;      // < 默认拷贝构造函数
    SegmentRes& operator=(const SegmentRes& other) = default;      // < 默认拷贝赋值运算符

    SegmentRes(SegmentRes&& other) noexcept            = default;  // < 默认移动构造函数
    SegmentRes& operator=(SegmentRes&& other) noexcept = default;  // < 默认移动赋值运算符
};

/**
 * @brief 姿态估计结果结构体，继承自基础结果结构体，增加矩形框和关键点信息
 */
struct DEPLOYAPI PoseRes : public BaseRes {
    std::vector<Box>                   boxes;  // < 姿态估计结果的矩形框
    std::vector<std::vector<KeyPoint>> kpts;   // < 姿态估计结果的关键点

    /**
     * @brief 默认构造函数
     *
     */
    PoseRes() = default;

    /**
     * @brief 构造函数，初始化姿态估计结果的数量、类别、得分、矩形框和关键点
     *
     * @param num 姿态估计结果的数量
     * @param classes 姿态估计结果的类别
     * @param scores 姿态估计结果的得分
     * @param boxes 姿态估计结果的矩形框
     * @param kpts 姿态估计结果的关键点
     */
    PoseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<std::vector<KeyPoint>>& kpts)
        : BaseRes(num, classes, scores), boxes(boxes), kpts(kpts) {}

    friend std::ostream& operator<<(std::ostream& os, const PoseRes& res) {
        os << "PoseRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "],\n    boxes=[\n";
        for (const auto& box : res.boxes) os << "        " << box << "\n";
        os << "],\n    kpts=[\n";
        for (const auto& kp_list : res.kpts) {
            os << "        [ ";
            for (const auto& kp : kp_list) os << "            " << kp << ", ";
            os << "        ],\n";
        }
        os << "    ]\n)";
        return os;
    }

    PoseRes(const PoseRes& other)            = default;      // < 默认拷贝构造函数
    PoseRes& operator=(const PoseRes& other) = default;      // < 默认拷贝赋值运算符

    PoseRes(PoseRes&& other) noexcept            = default;  // < 默认移动构造函数
    PoseRes& operator=(PoseRes&& other) noexcept = default;  // < 默认移动赋值运算符
};

}  // namespace deploy
