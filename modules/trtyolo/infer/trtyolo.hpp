/**
 * @file trtyolo.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT YOLO 模型推理相关类和结构体的定义
 * @date 2025-06-01
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#ifdef _MSC_VER
#define TRTYOLOAPI __declspec(dllexport)
#else
#define TRTYOLOAPI __attribute__((visibility("default")))
#endif

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trtyolo {

class BaseModel;  // 前向声明 BaseModel 模板类

/**
 * @brief 图像结构体，用于存储图像数据及其尺寸信息
 */
struct TRTYOLOAPI Image {
    void*  ptr;         // < 图像数据指针
    int    width  = 0;  // < 图像宽度（像素数）
    int    height = 0;  // < 图像高度（像素数）
    size_t pitch  = 0;  // < 图像数据行距（字节数，包括可能的 padding）

    /**
     * @brief 构造函数（紧密排列，无 padding）
     *
     * @param data 图像数据指针
     * @param width 图像宽度（像素数）
     * @param height 图像高度（像素数）
     */
    Image(void* data, int width, int height);

    /**
     * @brief 构造函数（带 pitch，支持 padding）
     * @param data 图像数据指针
     * @param width 图像宽度（像素数）
     * @param height 图像高度（像素数）
     * @param pitch 每行的字节数（包括可能的 padding）
     */
    Image(void* data, int width, int height, size_t pitch);

    friend std::ostream& operator<<(std::ostream& os, const Image& img) {
        os << "Image(width=" << img.width
           << ", height=" << img.height
           << ", pitch=" << img.pitch
           << ", ptr=" << img.ptr << ")";
        return os;
    }
};

/**
 * @brief 掩码结构体，用于存储掩码数据及其尺寸信息
 */
struct TRTYOLOAPI Mask {
    std::vector<uint8_t> data;        // < 掩码数据
    int                  width  = 0;  // < 掩码宽度
    int                  height = 0;  // < 掩码高度

    /**
     * @brief 构造函数，初始化掩码尺寸并分配数据空间
     *
     * @param width 掩码宽度
     * @param height 掩码高度
     */
    Mask(int width, int height);

    friend std::ostream& operator<<(std::ostream& os, const Mask& mask) {
        os << "Mask(width=" << mask.width << ", height=" << mask.height << ", data size=" << mask.data.size() << ")";
        return os;
    }
};

/**
 * @brief 关键点结构体，用于存储关键点的坐标和置信度
 */
struct TRTYOLOAPI KeyPoint {
    float                x;     // < 关键点的 x 坐标
    float                y;     // < 关键点的 y 坐标
    std::optional<float> conf;  // < 关键点的置信度，可选

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
};

/**
 * @brief 矩形框结构体，用于存储矩形框的坐标信息
 */
struct TRTYOLOAPI Box {
    float left;    // < 矩形框的左边界坐标
    float top;     // < 矩形框的上边界坐标
    float right;   // < 矩形框的右边界坐标
    float bottom;  // < 矩形框的下边界坐标

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
};

/**
 * @brief 旋转矩形框结构体，继承自矩形框结构体，增加旋转角度信息
 */
struct TRTYOLOAPI RotatedBox : public Box {
    float theta;  // < 旋转矩形框的旋转角度，以弧度为单位，顺时针旋转角度从正 x 轴开始测量

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
};

/**
 * @brief 基础结果结构体，用于存储检测结果的基础信息，如数量、类别和得分
 */
struct TRTYOLOAPI BaseRes {
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
};

/**
 * @brief 分类结果结构体，继承自基础结果结构体，无需额外的构造函数
 */
struct TRTYOLOAPI ClassifyRes : public BaseRes {
    // 无需额外的构造函数，继承自 BaseRes 的默认构造函数已经足够

    friend std::ostream& operator<<(std::ostream& os, const ClassifyRes& res) {
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
struct TRTYOLOAPI DetectRes : public BaseRes {
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
};

/**
 * @brief 旋转矩形框检测结果结构体，继承自基础结果结构体，增加旋转矩形框信息
 */
struct TRTYOLOAPI OBBRes : public BaseRes {
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
};

/**
 * @brief 分割结果结构体，继承自基础结果结构体，增加矩形框和掩码信息
 */
struct TRTYOLOAPI SegmentRes : public BaseRes {
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
};

/**
 * @brief 姿态估计结果结构体，继承自基础结果结构体，增加矩形框和关键点信息
 */
struct TRTYOLOAPI PoseRes : public BaseRes {
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
};

/**
 * @brief 推理选项配置类
 */
class TRTYOLOAPI InferOption {
public:
    InferOption();
    ~InferOption();

    /**
     * @brief 设置 GPU 设备 ID
     *
     * @param id
     */
    void setDeviceId(int id);

    /**
     * @brief 设置推理数据在 CUDA 显存中
     *
     */
    void enableCudaMem();

    /**
     * @brief 启用统一内存
     *
     */
    void enableManagedMemory();

    /**
     * @brief 启用性能报告
     *
     */
    void enablePerformanceReport();

    /**
     * @brief 设置图像通道交换
     *
     * @param swap_rb
     */
    void enableSwapRB();

    /**
     * @brief 设置边界值
     *
     * @param border_value
     */
    void setBorderValue(float border_value);

    /**
     * @brief 设置归一化参数
     *
     * @param mean
     * @param std
     */
    void setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std);

    /**
     * @brief 设置输入数据的宽高，未设置时表示宽高可变。（用于输入数据宽高确定的任务场景：监控视频分析，AI外挂等）
     *
     * @param width 宽度
     * @param height 高度
     */
    void setInputDimensions(int width, int height);

private:
    class Impl;                   // 前向声明实现类
    std::unique_ptr<Impl> impl_;  // 隐藏实现细节;
    friend class trtyolo::BaseModel;
};

/**
 * @brief 基类模板，用于定义模型的基本结构和接口。
 *
 */
class TRTYOLOAPI BaseModel {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    BaseModel();
    ~BaseModel();

    /**
     * @brief 构造一个新的 BaseModel 对象
     *
     * @param trt_engine_file TensorRT 引擎文件路径
     * @param infer_option 推理选项
     */
    explicit BaseModel(const std::string& trt_engine_file, const InferOption& infer_option);

    /**
     * @brief 获取批量大小
     *
     * @return 批量大小
     */
    int batch() const;

    /**
     * @brief 获取性能报告
     *
     * @return 包含吞吐量、CPU延迟和GPU延迟的元组
     */
    std::tuple<std::string, std::string, std::string> performanceReport();

protected:
    class Impl;                   // 前向声明实现类
    std::unique_ptr<Impl> impl_;  // 隐藏实现细节;
};

class TRTYOLOAPI ClassifyModel : public BaseModel {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    ClassifyModel();
    ~ClassifyModel();

    /**
     * @brief 构造一个新的 ClassifyModel 对象
     *
     * @param trt_engine_file TensorRT 引擎文件路径
     * @param infer_option 推理选项
     */
    explicit ClassifyModel(const std::string& trt_engine_file, const InferOption& infer_option);

    /**
     * @brief 克隆 ClassifyModel 对象
     *
     * @return 克隆后的 ClassifyModel 对象的智能指针
     */
    std::unique_ptr<ClassifyModel> clone() const;

    /**
     * @brief 对单张图像进行推理
     *
     * @param image 输入图像
     * @return 推理结果
     */
    ClassifyRes predict(const Image& image);

    /**
     * @brief 对多张图像进行推理
     *
     * @param images 输入图像向量
     * @return 推理结果向量
     */
    std::vector<ClassifyRes> predict(const std::vector<Image>& images);
};

class TRTYOLOAPI DetectModel : public BaseModel {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    DetectModel();
    ~DetectModel();

    /**
     * @brief 构造一个新的 DetectModel 对象
     *
     * @param trt_engine_file TensorRT 引擎文件路径
     * @param infer_option 推理选项
     */
    explicit DetectModel(const std::string& trt_engine_file, const InferOption& infer_option);

    /**
     * @brief 克隆 DetectModel 对象
     *
     * @return 克隆后的 DetectModel 对象的智能指针
     */
    std::unique_ptr<DetectModel> clone() const;

    /**
     * @brief 对单张图像进行推理
     *
     * @param image 输入图像
     * @return 推理结果
     */
    DetectRes predict(const Image& image);

    /**
     * @brief 对多张图像进行推理
     *
     * @param images 输入图像向量
     * @return 推理结果向量
     */
    std::vector<DetectRes> predict(const std::vector<Image>& images);
};

class TRTYOLOAPI OBBModel : public BaseModel {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    OBBModel();
    ~OBBModel();

    /**
     * @brief 构造一个新的 OBBModel 对象
     *
     * @param trt_engine_file TensorRT 引擎文件路径
     * @param infer_option 推理选项
     */
    explicit OBBModel(const std::string& trt_engine_file, const InferOption& infer_option);

    /**
     * @brief 克隆 OBBModel 对象
     *
     * @return 克隆后的 OBBModel 对象的智能指针
     */
    std::unique_ptr<OBBModel> clone() const;

    /**
     * @brief 对单张图像进行推理
     *
     * @param image 输入图像
     * @return 推理结果
     */
    OBBRes predict(const Image& image);

    /**
     * @brief 对多张图像进行推理
     *
     * @param images 输入图像向量
     * @return 推理结果向量
     */
    std::vector<OBBRes> predict(const std::vector<Image>& images);
};

class TRTYOLOAPI SegmentModel : public BaseModel {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    SegmentModel();
    ~SegmentModel();

    /**
     * @brief 构造一个新的 SegmentModel 对象
     *
     * @param trt_engine_file TensorRT 引擎文件路径
     * @param infer_option 推理选项
     */
    explicit SegmentModel(const std::string& trt_engine_file, const InferOption& infer_option);

    /**
     * @brief 克隆 SegmentModel 对象
     *
     * @return 克隆后的 SegmentModel 对象的智能指针
     */
    std::unique_ptr<SegmentModel> clone() const;

    /**
     * @brief 对单张图像进行推理
     *
     * @param image 输入图像
     * @return 推理结果
     */
    SegmentRes predict(const Image& image);

    /**
     * @brief 对多张图像进行推理
     *
     * @param images 输入图像向量
     * @return 推理结果向量
     */
    std::vector<SegmentRes> predict(const std::vector<Image>& images);
};

class TRTYOLOAPI PoseModel : public BaseModel {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    PoseModel();
    ~PoseModel();

    /**
     * @brief 构造一个新的 PoseModel 对象
     *
     * @param trt_engine_file TensorRT 引擎文件路径
     * @param infer_option 推理选项
     */
    explicit PoseModel(const std::string& trt_engine_file, const InferOption& infer_option);

    /**
     * @brief 克隆 PoseModel 对象
     *
     * @return 克隆后的 PoseModel 对象的智能指针
     */
    std::unique_ptr<PoseModel> clone() const;

    /**
     * @brief 对单张图像进行推理
     *
     * @param image 输入图像
     * @return 推理结果
     */
    PoseRes predict(const Image& image);

    /**
     * @brief 对多张图像进行推理
     *
     * @param images 输入图像向量
     * @return 推理结果向量
     */
    std::vector<PoseRes> predict(const std::vector<Image>& images);
};

}  // namespace trtyolo