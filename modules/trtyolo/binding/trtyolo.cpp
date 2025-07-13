/**
 * @file pybind.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT-YOLO的Python绑定，提供结果、选项和模型模块的包装
 * @date 2025-01-17
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 */

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "infer/trtyolo.hpp"

namespace py = pybind11;

/**
 * @brief 绑定result.hpp文件，包括Mask、KeyPoint、Box等类。
 *
 * 该模块包装了多个与结果相关的类，如Mask、KeyPoint、Box等，并提供与数据交互的方法，
 * 以及将它们转换为NumPy数组以便在Python中更容易操作的功能。
 */
void binding_result_module(py::module& m) {
    m.doc() = "Result module of TensorRT-YOLO, including Mask, KeyPoint, Box, and other result-related classes.";

    py::class_<trtyolo::Mask>(m, "Mask", "A class representing a mask with height, width, and associated data.")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def_readwrite("data", &trtyolo::Mask::data, "The mask data as a list of uint8 values.")
        .def_readwrite("width", &trtyolo::Mask::width, "The width of the mask.")
        .def_readwrite("height", &trtyolo::Mask::height, "The height of the mask.")
        .def("__str__", [](const trtyolo::Mask& mask) {
            std::ostringstream oss;
            oss << mask;
            return oss.str();
        })
        .def("to_numpy", [](const trtyolo::Mask& mask) {
            // 将Mask数据转换为形状为(height, width)的NumPy数组
            py::array_t<uint8_t> np_array({mask.height, mask.width});
            auto                 ptr = np_array.mutable_data();
            std::copy(mask.data.begin(), mask.data.end(), ptr);
            return np_array;
        });

    py::class_<trtyolo::KeyPoint>(m, "KeyPoint", "A class representing a keypoint with (x, y) coordinates and confidence score.")
        .def_readwrite("x", &trtyolo::KeyPoint::x, "The x-coordinate of the keypoint.")
        .def_readwrite("y", &trtyolo::KeyPoint::y, "The y-coordinate of the keypoint.")
        .def_readwrite("conf", &trtyolo::KeyPoint::conf, "The confidence score for the keypoint.")
        .def("__str__", [](const trtyolo::KeyPoint& kpt) {
            std::ostringstream oss;
            oss << kpt;
            return oss.str();
        });

    py::class_<trtyolo::Box>(m, "Box", "A class representing a bounding box with (left, top, right, bottom) coordinates.")
        .def_readwrite("left", &trtyolo::Box::left, "The left coordinate of the bounding box.")
        .def_readwrite("top", &trtyolo::Box::top, "The top coordinate of the bounding box.")
        .def_readwrite("right", &trtyolo::Box::right, "The right coordinate of the bounding box.")
        .def_readwrite("bottom", &trtyolo::Box::bottom, "The bottom coordinate of the bounding box.")
        .def("__str__", [](const trtyolo::Box& box) {
            std::ostringstream oss;
            oss << box;
            return oss.str();
        });

    py::class_<trtyolo::RotatedBox, trtyolo::Box>(m, "RotatedBox", "A class representing a rotated bounding box with an additional rotation angle (theta).")
        .def_readwrite("theta", &trtyolo::RotatedBox::theta, "The rotation angle of the bounding box.")
        .def("__str__", [](const trtyolo::RotatedBox& rbox) {
            std::ostringstream oss;
            oss << rbox;
            return oss.str();
        });

    py::class_<trtyolo::ClassifyRes>(m, "ClassifyRes", "A class representing classification results, including number of results, classes, and their confidence scores.")
        .def_readwrite("num", &trtyolo::ClassifyRes::num, "The number of classification results.")
        .def_readwrite("classes", &trtyolo::ClassifyRes::classes, "List of class IDs for classification results.")
        .def_readwrite("scores", &trtyolo::ClassifyRes::scores, "List of confidence scores for each class.")
        .def("__str__", [](const trtyolo::ClassifyRes& res) {
            std::ostringstream oss;
            oss << res;
            return oss.str();
        });

    py::class_<trtyolo::DetectRes>(m, "DetectRes", "A class representing detection results, including the number of detections, classes, scores, and bounding boxes.")
        .def_readwrite("num", &trtyolo::DetectRes::num, "The number of detection results.")
        .def_readwrite("classes", &trtyolo::DetectRes::classes, "List of class IDs for detected objects.")
        .def_readwrite("scores", &trtyolo::DetectRes::scores, "List of confidence scores for each detection.")
        .def_readwrite("boxes", &trtyolo::DetectRes::boxes, "List of bounding boxes for detected objects.")
        .def("__str__", [](const trtyolo::DetectRes& res) {
            std::ostringstream oss;
            oss << res;
            return oss.str();
        });

    py::class_<trtyolo::OBBRes>(m, "OBBRes", "A class representing the results of rotated bounding box detection.")
        .def_readwrite("num", &trtyolo::OBBRes::num, "The number of OBB detection results.")
        .def_readwrite("classes", &trtyolo::OBBRes::classes, "List of class IDs for rotated bounding boxes.")
        .def_readwrite("scores", &trtyolo::OBBRes::scores, "List of confidence scores for rotated bounding boxes.")
        .def_readwrite("boxes", &trtyolo::OBBRes::boxes, "List of rotated bounding boxes for detected objects.")
        .def("__str__", [](const trtyolo::OBBRes& res) {
            std::ostringstream oss;
            oss << res;
            return oss.str();
        });

    py::class_<trtyolo::SegmentRes>(m, "SegmentRes", "A class representing segmentation results, including boxes and masks for detected objects.")
        .def_readwrite("num", &trtyolo::SegmentRes::num, "The number of segmentation results.")
        .def_readwrite("classes", &trtyolo::SegmentRes::classes, "List of class IDs for segmented objects.")
        .def_readwrite("scores", &trtyolo::SegmentRes::scores, "List of confidence scores for each segmentation.")
        .def_readwrite("boxes", &trtyolo::SegmentRes::boxes, "List of bounding boxes for segmented objects.")
        .def_readwrite("masks", &trtyolo::SegmentRes::masks, "List of masks for segmented objects.")
        .def("__str__", [](const trtyolo::SegmentRes& res) {
            std::ostringstream oss;
            oss << res;
            return oss.str();
        });

    py::class_<trtyolo::PoseRes>(m, "PoseRes", "A class representing pose detection results, including keypoints, scores, and bounding boxes.")
        .def_readwrite("num", &trtyolo::PoseRes::num, "The number of pose detection results.")
        .def_readwrite("classes", &trtyolo::PoseRes::classes, "List of class IDs for pose detection.")
        .def_readwrite("scores", &trtyolo::PoseRes::scores, "List of confidence scores for each pose detection.")
        .def_readwrite("boxes", &trtyolo::PoseRes::boxes, "List of bounding boxes for pose detections.")
        .def_readwrite("kpts", &trtyolo::PoseRes::kpts, "List of keypoints for pose detections.")
        .def("__str__", [](const trtyolo::PoseRes& res) {
            std::ostringstream oss;
            oss << res;
            return oss.str();
        });
}

/**
 * @brief 绑定option.hpp文件，允许用户设置各种推理选项。
 *
 * 该模块提供了InferOption类的绑定，可以配置各种推理参数，如设备设置、内存选项和图像预处理等。
 */
void binding_option_module(py::module& m) {
    m.doc() = "Option module of TensorRT-YOLO, providing options to configure inference settings.";

    py::class_<trtyolo::InferOption>(m, "InferOption", "A class to configure inference options, including device settings, memory options, and image preprocessing.")
        .def(py::init<>())
        .def("set_device_id", &trtyolo::InferOption::setDeviceId, "Set the device ID (GPU) for inference.")
        .def("enable_cuda_memory", &trtyolo::InferOption::enableCudaMem, "Inference data already in CUDA memory.")
        .def("enable_managed_memory", &trtyolo::InferOption::enableManagedMemory, "Enable managed memory for inference.")
        .def("enable_performance_report", &trtyolo::InferOption::enablePerformanceReport, "Enable performance report for inference.")
        .def("enable_swap_rb", &trtyolo::InferOption::enableSwapRB, "Enable RGB-to-BGR swap for image input.")
        .def("set_border_value", &trtyolo::InferOption::setBorderValue, "Set border value for image resizing (used for padding).")
        .def("set_normalize_params", &trtyolo::InferOption::setNormalizeParams, "Set normalization parameters for image preprocessing.")
        .def("set_input_dimensions", &trtyolo::InferOption::setInputDimensions, "Set the input dimensions (height, width) for the model.");
}

/**
 * @brief 将NumPy数组转换为trtyolo::Image对象。
 *
 * 此函数用于将一个3D的NumPy数组（高、宽、通道）转换为trtyolo::Image对象，用于推理任务。
 *
 * @param pyarray 一个形状为(height, width, channels)且数据类型为uint8的NumPy数组。
 * @return trtyolo::Image 一个可用于推理的图像对象。
 * @throws std::invalid_argument 如果输入数组不是3维，或者数据类型不是uint8，则抛出异常。
 */
trtyolo::Image PyArray2Image(py::array& pyarray) {
    if (pyarray.ndim() != 3) {
        throw std::invalid_argument("Require rank of array to be 3 with HWC format while converting it to trtyolo::Image.");
    }
    py::buffer_info buf_info = pyarray.request();
    int             height   = buf_info.shape[0];
    int             width    = buf_info.shape[1];
    void*           data     = buf_info.ptr;
    if (buf_info.format != py::format_descriptor<uint8_t>::format()) {
        throw std::invalid_argument("Expected array data type is uint8.");
    }
    return trtyolo::Image(data, width, height);
}

/**
 * @brief 模板函数，绑定一个模型类，允许对单张或多张图像进行预测。
 *
 * 该模板函数包装了常见的模型行为，便于将各种模型（如ClassifyModel、DetectModel）添加到Python接口。
 *
 * @tparam ModelType 模型类型（如ClassifyModel、DetectModel等）。
 * @param m 要绑定类的pybind11模块。
 * @param model_name 模型类的名称。
 */
template <typename ModelType>
void bind_model(py::module& m, const std::string& model_name) {
    py::class_<ModelType, std::unique_ptr<ModelType>>(m, model_name.c_str(), "A model class for performing inference tasks.")
        .def(py::init<const std::string&, const trtyolo::InferOption&>(), "Initialize the model with a model file and inference options.")
        .def("clone", &ModelType::clone, "Clone the model instance.")
        .def(
            "predict", [](ModelType& self, py::array& input) {
                return self.predict(PyArray2Image(input));
            },
            "Predict the result from a single image (HWC format numpy array).")
        .def("predict", [](ModelType& self, std::vector<py::array>& inputs) {
                std::vector<trtyolo::Image> images;
                std::transform(inputs.begin(), inputs.end(), std::back_inserter(images), [](py::array& input) {
                    return PyArray2Image(input);
                });
                return self.predict(images); }, "Predict the results from a list of images (list of HWC format numpy arrays).")
        .def("performance_report", [](ModelType& self) {
            auto report = self.performanceReport();
            return std::make_tuple(py::str(std::get<0>(report)), py::str(std::get<1>(report)), py::str(std::get<2>(report))); }, "Get the performance report of the model.")
        .def("batch", &ModelType::batch, "Get the batch size of the model.");
}

/**
 * @brief 绑定模型模块，包括分类、检测等各种模型类。
 *
 * 该模块绑定了所有模型类，使它们可以在Python中进行推理任务。
 */
void binding_model_module(py::module& m) {
    m.doc() = "Python bindings for model.hpp, including classes like ClassifyModel, DetectModel, OBBModel, etc.";

    bind_model<trtyolo::ClassifyModel>(m, "ClassifyModel");
    bind_model<trtyolo::DetectModel>(m, "DetectModel");
    bind_model<trtyolo::OBBModel>(m, "OBBModel");
    bind_model<trtyolo::SegmentModel>(m, "SegmentModel");
    bind_model<trtyolo::PoseModel>(m, "PoseModel");
}

/**
 * @brief TensorRT-YOLO的Python绑定主模块。
 *
 * 该模块包括结果、选项和模型子模块的绑定，允许用户轻松处理推理结果、配置选项，并执行分类、检测等任务。
 */
PYBIND11_MODULE(py_trtyolo, m) {
    m.doc() = "TensorRT-YOLO Python bindings using Pybind11";

    py::module result_module = m.def_submodule("result", "Result module of TensorRT-YOLO.");
    binding_result_module(result_module);

    py::module option_module = m.def_submodule("option", "Option module of TensorRT-YOLO.");
    binding_option_module(option_module);

    py::module model_module = m.def_submodule("model", "Model module of TensorRT-YOLO.");
    binding_model_module(model_module);
}
