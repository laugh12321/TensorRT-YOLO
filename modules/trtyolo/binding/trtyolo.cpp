/**
 * @file pybind.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief trtyolo的Python绑定，提供结果、选项和模型模块的包装
 * @date 2025-01-17
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 */

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>

#include "infer/trtyolo.hpp"

namespace py = pybind11;

/**
 * @brief 模板函数，用于将结果类绑定到Python接口，支持对单张或多张结果进行操作。
 *
 * 该模板函数会封装常见的结果类行为，将不同类型的结果类（如分类结果ClassifyRes、检测结果DetectRes等）
 * 绑定到Python模块中，方便在Python环境中使用这些结果类。同时会根据不同的结果类型，
 * 提供不同的属性访问方法，例如获取边界框坐标、分割掩码、关键点信息等。
 *
 * @tparam ResultType 结果类型模板参数，可传入具体的结果类（如ClassifyRes、DetectRes等）。
 * @param m 要绑定类的pybind11模块对象，用于创建Python类。
 * @param result_name 结果类的名称，用于在Python中标识该类。
 */
template <typename ResultType>
void bind_result(py::module& m, const std::string& result_name) {
    std::string docstring;
    if constexpr (std::is_same_v<ResultType, trtyolo::ClassifyRes>) {
        docstring = "A class representing classification results containing predicted classes and their confidence scores.";
    } else if constexpr (std::is_same_v<ResultType, trtyolo::DetectRes>) {
        docstring = "A class representing object detection results containing bounding boxes, predicted classes and their confidence scores.";
    } else if constexpr (std::is_same_v<ResultType, trtyolo::OBBRes>) {
        docstring = "A class representing oriented object detection results containing rotated bounding boxes, predicted classes and their confidence scores.";
    } else if constexpr (std::is_same_v<ResultType, trtyolo::SegmentRes>) {
        docstring = "A class representing instance segmentation results containing bounding boxes, segmentation masks, predicted classes and their confidence scores.";
    } else if constexpr (std::is_same_v<ResultType, trtyolo::PoseRes>) {
        docstring = "A class representing pose estimation results containing bounding boxes, keypoints, predicted classes and their confidence scores.";
    }

    py::class_<ResultType, trtyolo::BaseRes> cls(m, result_name.c_str(), docstring.c_str());

    if constexpr (std::is_same_v<ResultType, trtyolo::DetectRes> ||
                  std::is_same_v<ResultType, trtyolo::OBBRes> ||
                  std::is_same_v<ResultType, trtyolo::SegmentRes> ||
                  std::is_same_v<ResultType, trtyolo::PoseRes>) {
        cls.def_property_readonly("xyxy", [](const ResultType& r) -> py::array {
            auto* owner = new std::vector<int>;
            owner->reserve(r.boxes.size() * 4);
            for (const auto& b : r.boxes) {
                const auto& arr = b.xyxy();
                owner->insert(owner->end(), arr.begin(), arr.end());
            }

            return py::array(
                py::buffer_info(
                    owner->data(),
                    sizeof(int),
                    py::format_descriptor<int>::format(),
                    2,
                    std::vector<size_t>{r.boxes.size(), 4},
                    std::vector<size_t>{sizeof(int) * 4, sizeof(int)}),
                py::cast(owner)); },
                                  "An array of shape (n, 4) containing the bounding boxes "
                                  "coordinates in format [x1, y1, x2, y2].");

        if constexpr (std::is_same_v<ResultType, trtyolo::OBBRes>) {
            cls.def_property_readonly("xyxyxyxy", [](const ResultType& r) -> py::array {
                auto* owner = new std::vector<int>;
                owner->reserve(r.boxes.size() * 8);
                for (const auto& b : r.boxes) {
                    const std::array<int,8>& arr = b.xyxyxyxy();
                    owner->insert(owner->end(), arr.begin(), arr.end());
                }

                return py::array(
                    py::buffer_info(
                        owner->data(),
                        sizeof(int),
                        py::format_descriptor<int>::format(),
                        3,
                        std::vector<size_t>{r.boxes.size(), 4, 2},
                        std::vector<size_t>{sizeof(int) * 8, sizeof(int) * 2, sizeof(int)}),
                    py::cast(owner)); },
                                      "An array of shape (n, 4, 2) containing the bounding boxes "
                                      "coordinates in format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].");
        }
    }

    if constexpr (std::is_same_v<ResultType, trtyolo::SegmentRes>) {
        cls.def_property_readonly("masks", [](const ResultType& r) -> py::array {
            if (r.masks.empty()) return py::array_t<float>();
            size_t n = r.masks.size();
            size_t h = r.masks[0].height;
            size_t w = r.masks[0].width;

            auto* owner = new std::vector<float>;
            owner->reserve(n * h * w);
            for (const auto& mask : r.masks) owner->insert(owner->end(), mask.data.begin(), mask.data.end());
            return py::array(
                py::buffer_info(
                    owner->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    3,
                    {n, h, w},
                    {sizeof(float) * h * w, sizeof(float) * w, sizeof(float)}),
                py::cast(owner)); }, "An array of shape (n, H, W) containing the segmentation masks.");
    }

    if constexpr (std::is_same_v<ResultType, trtyolo::PoseRes>) {
        cls.def_property_readonly("kpts", [](const ResultType& r) -> py::array {
            size_t n = r.kpts.size();
            size_t m = r.kpts.empty() ? 0 : r.kpts[0].size();
            size_t c = r.kpts.empty() || r.kpts[0].empty() ? 0 : r.kpts[0][0].conf ? 3 : 2;

            py::array_t<float> array({n, m, c});
            auto               rptr = array.mutable_unchecked<3>();

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    rptr(i, j, 0) = r.kpts[i][j].x;
                    rptr(i, j, 1) = r.kpts[i][j].y;
                    if (c == 3) rptr(i, j, 2) = r.kpts[i][j].conf.value_or(0.0f);
                }
            }
            return array; },
                                  "An array of shape (n, m, c) containing n detected objects, "
                                  "each composed of m equally-sized sets of keypoints and confidence scores of each keypoint. "
                                  "Where each point is [x, y] if c is 2, or [x, y, conf] if c is 3.");
    }
};

/**
 * @brief 为trtyolo结果类创建Python绑定
 *
 * 该函数负责将trtyolo中的结果相关C++类绑定到Python，使其可以在Python中使用。
 * 主要功能包括：
 * 1. 绑定基础结果类BaseRes，提供通用属性和方法
 * 2. 将C++数据结构转换为NumPy数组，便于在Python中进行数据分析和可视化
 * 3. 注册多种专门的结果类型，每种类型对应不同的计算机视觉任务
 *
 * 绑定的结果类型包括：
 * - ClassifyRes: 分类结果
 * - DetectRes: 目标检测结果(矩形框)
 * - OBBRes: 旋转目标检测结果
 * - SegmentRes: 实例分割结果(带掩码)
 * - PoseRes: 人体姿态估计结果(带关键点)
 */
void binding_result_module(py::module& m) {
    m.doc() = "Result module of trtyolo, providing Python bindings for various computer vision result types including classification, object detection, rotated object detection, instance segmentation and pose estimation.";

    // 基础结果类，所有结果类型的基类，提供通用功能和属性
    py::class_<trtyolo::BaseRes>(m, "BaseRes", "A base class for all result types, providing common properties like number of results, class IDs, and confidence scores.")
        // 允许使用Python的len()函数获取结果数量
        .def("__len__", [](const trtyolo::BaseRes& r) { return r.num; }, "Return the number of detected objects via Python's len() function.")
        // 检测对象的类别ID数组
        .def_property_readonly("class_id", [](const trtyolo::BaseRes& r) -> py::array { return py::array(
                                                                                            py::buffer_info(
                                                                                                const_cast<int*>(r.classes.data()),
                                                                                                sizeof(int),
                                                                                                py::format_descriptor<int>::format(),
                                                                                                1,
                                                                                                {r.classes.size()},
                                                                                                {sizeof(int)})); }, "An array of shape (n,) containing the class IDs of the detections.")
        // 检测对象的置信度分数数组
        .def_property_readonly("confidence", [](const trtyolo::BaseRes& r) -> py::array { return py::array(
                                                                                              py::buffer_info(
                                                                                                  const_cast<float*>(r.scores.data()),
                                                                                                  sizeof(float),
                                                                                                  py::format_descriptor<float>::format(),
                                                                                                  1,
                                                                                                  {r.scores.size()},
                                                                                                  {sizeof(float)})); }, "An array of shape (n,) containing the confidence scores of the detections.");

    // 绑定各种专门的结果类型
    bind_result<trtyolo::ClassifyRes>(m, "ClassifyRes");  // 分类结果
    bind_result<trtyolo::DetectRes>(m, "DetectRes");      // 目标检测结果(矩形框)
    bind_result<trtyolo::OBBRes>(m, "OBBRes");            // 旋转目标检测结果
    bind_result<trtyolo::SegmentRes>(m, "SegmentRes");    // 实例分割结果(带掩码)
    bind_result<trtyolo::PoseRes>(m, "PoseRes");          // 人体姿态估计结果(带关键点)
}

/**
 * @brief 为trtyolo推理选项创建Python绑定
 *
 * 该函数负责将trtyolo中的InferOption类绑定到Python，提供灵活的推理配置选项。
 * 通过这个模块，用户可以定制推理过程中的各项参数，包括但不限于：
 * 1. 设备设置：选择GPU设备
 * 2. 性能监控：启用性能分析
 * 3. 图像预处理：配置色彩转换、边界值、归一化参数等
 * 4. 模型输入尺寸：设置输入图像的高度和宽度
 *
 * 所有配置选项将直接影响推理性能和结果质量，用户可以根据具体需求进行调整。
 */
void binding_option_module(py::module& m) {
    m.doc() = "Option module of trtyolo, providing comprehensive configuration options for inference including device selection, performance monitoring, and image preprocessing.";

    py::class_<trtyolo::InferOption>(m, "InferOption", "A class to configure advanced inference options for trtyolo models, controlling device selection, performance monitoring, and image preprocessing.")
        .def(py::init<>())
        .def("set_device_id", &trtyolo::InferOption::setDeviceId, "Set the device ID (GPU) for inference.")
        // .def("enable_cuda_memory", &trtyolo::InferOption::enableCudaMem, "Inference data already in CUDA memory.")
        // .def("enable_managed_memory", &trtyolo::InferOption::enableManagedMemory, "Enable managed memory for inference.")
        .def("enable_profile", &trtyolo::InferOption::enablePerformanceReport, "Enable performance profile for inference.")
        .def("enable_swap_rb", &trtyolo::InferOption::enableSwapRB, "Enable RGB-to-BGR swap for image input.")
        .def("set_border_value", &trtyolo::InferOption::setBorderValue, "Set border value for image resizing (used for padding).")
        .def("set_normalize_params", &trtyolo::InferOption::setNormalizeParams, "Set normalization parameters for image preprocessing.")
        .def("set_input_dimensions", &trtyolo::InferOption::setInputDimensions, "Set the input dimensions (height, width) for the model.");
}

/**
 * @brief 将NumPy数组转换为trtyolo::Image对象。
 *
 * 此函数会把一个形状为(height, width, channels)、数据类型为uint8的3D NumPy数组，
 * 转换为trtyolo::Image对象，该对象可用于后续的推理任务。
 * 函数会检查输入数组的维度和通道数，若输入数组不是3维，或者第三维通道数不为3，
 * 则抛出 std::invalid_argument 异常。
 *
 * @param src 输入的NumPy数组，要求形状为(height, width, channels)，数据类型为uint8。
 * @return trtyolo::Image 转换后的可用于推理的图像对象。
 * @throws std::invalid_argument 当输入数组不是3维，或者第三维通道数不为3时抛出此异常。
 */
trtyolo::Image PyArray2Image(const py::array_t<uint8_t>& src) {
    py::buffer_info buf = src.request();
    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::invalid_argument("Require rank of array to be 3 with HWC format while converting it to trtyolo::Image.");
    }

    return trtyolo::Image(buf.ptr, buf.shape[1], buf.shape[0], buf.shape[2], buf.strides[0]);
}

/**
 * @brief 为trtyolo模型类创建Python绑定的模板函数
 *
 * 该模板函数提供了一个统一的方式来将各种trtyolo模型类绑定到Python接口，
 * 封装了模型的通用功能，如初始化、预测、克隆和性能分析等。通过这个模板，
 * 可以轻松地为不同类型的模型（如分类、检测、分割等）创建一致的Python API。
 *
 * @tparam ModelType 模型类型模板参数，可以是ClassifyModel、DetectModel、OBBModel等
 * @param m 目标pybind11模块，用于注册Python类
 * @param model_name Python中可见的模型类名称
 */
template <typename ModelType>
void bind_model(py::module& m, const std::string& model_name) {
    py::class_<ModelType, std::unique_ptr<ModelType>>(m, model_name.c_str(),
                                                      "A trtyolo model class for performing computer vision inference tasks with TensorRT acceleration.")
        .def(py::init<const std::string&, const trtyolo::InferOption&>(),
             "Initialize the model with a TensorRT engine file path and inference configuration options.")
        .def("clone", &ModelType::clone,
             "Create a clone of the current model instance with the same configuration (shared engine, create a new context).")
        .def(
            "predict", [](ModelType& self, py::array& input) {
                return self.predict(PyArray2Image(input));
            },
            "Run inference on a single image represented as a HWC-format numpy array (height, width, channels).")
        .def("predict", [](ModelType& self, std::vector<py::array>& inputs) {
                std::vector<trtyolo::Image> images;
                images.reserve(inputs.size());
                for (auto& arr : inputs) images.push_back(PyArray2Image(arr));
                return self.predict(images); }, "Run batch inference on a list of images, each represented as a HWC-format numpy array.")
        .def("profile", [](ModelType& self) {
            auto report = self.performanceReport();
            return std::make_tuple(py::str(std::get<0>(report)), py::str(std::get<1>(report)), py::str(std::get<2>(report))); }, "Get the performance profile of the model as a tuple of (preprocess_time, inference_time, postprocess_time).")
        .def_property_readonly("batch", &ModelType::batch, "Get the maximum batch size supported by the model for batch inference.");
}

/**
 * @brief 绑定模型模块，实现所有模型类到Python的映射。
 *
 * 该函数负责将trtyolo命名空间下的各类模型（分类、检测、旋转框检测、实例分割、姿态估计）
 * 绑定到Python模块中，使这些模型可以在Python环境中被实例化和调用，
 * 执行相应的计算机视觉推理任务。每个模型类都通过bind_model模板函数进行统一绑定，
 * 确保接口一致性和功能完整性。
 *
 * @param m Python模块对象，用于注册模型类
 */
void binding_model_module(py::module& m) {
    m.doc() =
        "Model module of trtyolo, providing TensorRT-accelerated inference for computer vision tasks, "
        "including ClassifyModel, DetectModel, OBBModel, SegmentModel, and PoseModel.";

    bind_model<trtyolo::ClassifyModel>(m, "ClassifyModel");
    bind_model<trtyolo::DetectModel>(m, "DetectModel");
    bind_model<trtyolo::OBBModel>(m, "OBBModel");
    bind_model<trtyolo::SegmentModel>(m, "SegmentModel");
    bind_model<trtyolo::PoseModel>(m, "PoseModel");
}

/**
 * @brief trtyolo Python绑定主模块。
 *
 * 此模块借助Pybind11创建TensorRT-YOLO的Python绑定接口，
 * 包含三个子模块：
 * - result：封装推理结果类，可将结果转换为NumPy数组以便在Python中使用；
 * - option：提供推理选项配置类，可设置设备、内存、图像预处理等参数；
 * - model：绑定各类模型，支持单张或批量图像的推理任务。
 * 用户可通过该模块在Python环境中轻松处理推理结果、配置推理选项，
 * 并执行分类、检测、目标旋转框检测、实例分割和关键点检测等任务。
 */
PYBIND11_MODULE(py_trtyolo, m) {
    m.doc() = "Python bindings for trtyolo, providing TensorRT-accelerated inference for computer vision tasks.";

    py::module result_module = m.def_submodule("result");
    binding_result_module(result_module);

    py::module option_module = m.def_submodule("option");
    binding_option_module(option_module);

    py::module model_module = m.def_submodule("model");
    binding_model_module(model_module);
}
