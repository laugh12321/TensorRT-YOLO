#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>
#include <vector>

#include "deploy/utils/utils.hpp"
#include "deploy/vision/detection.hpp"
#include "deploy/vision/result.hpp"

namespace deploy {

// Convert NumPy array to Image object
Image PyArray2Image(pybind11::array &pyarray) {
    if (pyarray.ndim() != 3) {
        throw std::invalid_argument("Require rank of array to be 3 with HWC format while converting it to deploy::Image.");
    }

    int   height = pyarray.shape(0);
    int   width  = pyarray.shape(1);
    void *data   = pyarray.mutable_data();

    return Image(data, width, height);
}

// Bind utility classes
void BindUtils(pybind11::module &m) {
    m.doc() = "Python bindings for CpuTimer and GpuTimer using Pybind11";

    pybind11::class_<TimerBase>(m, "TimerBase", "Base class for timers.")
        .def(pybind11::init<>())
        .def("start", &TimerBase::start, "Starts the timer.")
        .def("stop", &TimerBase::stop, "Stops the timer.")
        .def("microseconds", &TimerBase::microseconds, "Get the elapsed time in microseconds.")
        .def("milliseconds", &TimerBase::milliseconds, "Get the elapsed time in milliseconds.")
        .def("seconds", &TimerBase::seconds, "Get the elapsed time in seconds.")
        .def("reset", &TimerBase::reset, "Resets the timer.");

    pybind11::class_<GpuTimer, TimerBase>(m, "GpuTimer", "Class for GPU timer.")
        .def(pybind11::init<>())
        .def("start", &GpuTimer::start, "Starts the GPU timer.")
        .def("stop", &GpuTimer::stop, "Stops the GPU timer.");

    pybind11::class_<CpuTimer, TimerBase>(m, "CpuTimer", "Class for CPU timer using high resolution clock.")
        .def(pybind11::init<>())
        .def("start", &CpuTimer::start, "Starts the CPU timer.")
        .def("stop", &CpuTimer::stop, "Stops the CPU timer and calculates the elapsed time.");
}

// Bind result classes
void BindResult(pybind11::module &m) {
    m.doc() = "Bindings for result structures like Box and DetectionResult";

    // Bind Box structure
    pybind11::class_<Box>(m, "Box")
        .def(pybind11::init<>())
        .def_readwrite("left", &Box::left)
        .def_readwrite("top", &Box::top)
        .def_readwrite("right", &Box::right)
        .def_readwrite("bottom", &Box::bottom)
        .def("__str__", [](const Box &box) {
            std::ostringstream oss;
            oss << "Box(left=" << box.left << ", top=" << box.top
                << ", right=" << box.right << ", bottom=" << box.bottom << ")";
            return oss.str();
        });

    // Bind DetectionResult structure
    pybind11::class_<DetectionResult>(m, "DetectionResult")
        .def(pybind11::init<>())
        .def_readwrite("num", &DetectionResult::num)
        .def_readwrite("boxes", &DetectionResult::boxes)
        .def_readwrite("classes", &DetectionResult::classes)
        .def_readwrite("scores", &DetectionResult::scores)
        .def("__copy__", [](const DetectionResult &self) {
            return DetectionResult(self);
        })
        .def("__deepcopy__", [](const DetectionResult &self, pybind11::dict) {
            return DetectionResult(self);
        })
        .def("__str__", [](const DetectionResult &dr) {
            std::ostringstream oss;
            oss << "DetectionResult(\n"
                << "  num=" << dr.num << ",\n"
                << "  classes=[";
            for (size_t i = 0; i < dr.classes.size(); ++i) {
                oss << dr.classes[i];
                if (i != dr.classes.size() - 1) oss << ", ";
            }
            oss << "],\n"
                << "  scores=[";
            for (size_t i = 0; i < dr.scores.size(); ++i) {
                oss << dr.scores[i];
                if (i != dr.scores.size() - 1) oss << ", ";
            }
            oss << "]\n"
                << "  boxes=[\n";
            for (const auto &box : dr.boxes) {
                oss << "    Box(left=" << box.left << ", top=" << box.top
                    << ", right=" << box.right << ", bottom=" << box.bottom << "),\n";
            }
            oss << "  ]\n)";
            return oss.str();
        })
        .def(pybind11::pickle([](const DetectionResult &dr) {  // __getstate__ equivalent
            return pybind11::make_tuple(dr.num, dr.boxes, dr.classes, dr.scores);
        },
                              [](pybind11::tuple t) {          // __setstate__ equivalent
                                  if (t.size() != 4)
                                      throw std::runtime_error("Invalid state!");

                                  DetectionResult dr;
                                  dr.num     = t[0].cast<int>();
                                  dr.boxes   = t[1].cast<std::vector<Box>>();
                                  dr.classes = t[2].cast<std::vector<int>>();
                                  dr.scores  = t[3].cast<std::vector<float>>();
                                  return dr;
                              }));
}

// Bind detection classes
void BindDetection(pybind11::module &m) {
    m.doc() = "Bindings for detection classes like DeployDet and DeployCGDet";

    // Bind DeployDet class
    pybind11::class_<DeployDet>(m, "DeployDet")
        .def(pybind11::init<const std::string &, bool, int>(),
             pybind11::arg("file"), pybind11::arg("cudaMem") = false, pybind11::arg("device") = 0)
        .def(
            "predict", [](DeployDet &self, pybind11::array &pyimg) {
                Image image = PyArray2Image(pyimg);
                return self.predict(image);
            },
            "Predict the result from a single image")
        .def(
            "predict", [](DeployDet &self, std::vector<pybind11::array> &pyimgs) {
                std::vector<Image> images;
                for (auto &pyimg : pyimgs) {
                    images.push_back(PyArray2Image(pyimg));
                }
                return self.predict(images);
            },
            "Predict the results from a list of images")
        .def_readwrite("batch", &DeployDet::batch, "Batch size for prediction");

    // Bind DeployCGDet class
    pybind11::class_<DeployCGDet>(m, "DeployCGDet")
        .def(pybind11::init<const std::string &, bool, int>(),
             pybind11::arg("file"), pybind11::arg("cudaMem") = false, pybind11::arg("device") = 0)
        .def(
            "predict", [](DeployCGDet &self, pybind11::array &pyimg) {
                Image image = PyArray2Image(pyimg);
                return self.predict(image);
            },
            "Predict the result from a single image")
        .def(
            "predict", [](DeployCGDet &self, std::vector<pybind11::array> &pyimgs) {
                std::vector<Image> images;
                for (auto &pyimg : pyimgs) {
                    images.push_back(PyArray2Image(pyimg));
                }
                return self.predict(images);
            },
            "Predict the results from a list of images")
        .def_readwrite("batch", &DeployCGDet::batch, "Batch size for prediction");
}

// Define the pydeploy module
PYBIND11_MODULE(pydeploy, m) {
    m.doc() = "TensorRT-YOLO Python bindings using Pybind11";

    pybind11::module timer_module = m.def_submodule("timer", "Timer module of TensorRT-YOLO.");
    BindUtils(timer_module);

    pybind11::module result_module = m.def_submodule("result", "Result module of TensorRT-YOLO.");
    BindResult(result_module);

    pybind11::module detection_module = m.def_submodule("detection", "Detection module of TensorRT-YOLO.");
    BindDetection(detection_module);
}

}  // namespace deploy
