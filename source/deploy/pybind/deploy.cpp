#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>
#include <vector>

#include "deploy/utils/utils.hpp"
#include "deploy/vision/inference.hpp"
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
    m.doc() = "Bindings for result structures like Box, RotatedBox, DetResult, and OBBResult";

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

    // Bind RotatedBox structure
    pybind11::class_<RotatedBox, Box>(m, "RotatedBox")
        .def(pybind11::init<>())
        .def_readwrite("theta", &RotatedBox::theta)
        .def("__str__", [](const RotatedBox &rbox) {
            std::ostringstream oss;
            oss << "RotatedBox(left=" << rbox.left << ", top=" << rbox.top
                << ", right=" << rbox.right << ", bottom=" << rbox.bottom
                << ", theta=" << rbox.theta << ")";
            return oss.str();
        });

    // Bind DetResult structure
    pybind11::class_<DetResult>(m, "DetResult")
        .def(pybind11::init<>())
        .def_readwrite("num", &DetResult::num)
        .def_readwrite("boxes", &DetResult::boxes)
        .def_readwrite("classes", &DetResult::classes)
        .def_readwrite("scores", &DetResult::scores)
        .def("__copy__", [](const DetResult &self) {
            return DetResult(self);
        })
        .def("__deepcopy__", [](const DetResult &self, pybind11::dict) {
            return DetResult(self);
        })
        .def("__str__", [](const DetResult &dr) {
            std::ostringstream oss;
            oss << "DetResult(num=" << dr.num << ", classes=[";
            for (size_t i = 0; i < dr.classes.size(); ++i) {
                oss << dr.classes[i];
                if (i != dr.classes.size() - 1) oss << ", ";
            }
            oss << "], scores=[";
            for (size_t i = 0; i < dr.scores.size(); ++i) {
                oss << dr.scores[i];
                if (i != dr.scores.size() - 1) oss << ", ";
            }
            oss << "], boxes=[\n";
            for (const auto &box : dr.boxes) {
                oss << "    Box(left=" << box.left << ", top=" << box.top
                    << ", right=" << box.right << ", bottom=" << box.bottom << "),\n";
            }
            oss << "  ])";
            return oss.str();
        })
        .def(pybind11::pickle([](const DetResult &dr) {
            return pybind11::make_tuple(dr.num, dr.boxes, dr.classes, dr.scores);
        },
        [](pybind11::tuple t) {
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            DetResult dr;
            dr.num = t[0].cast<int>();
            dr.boxes = t[1].cast<std::vector<Box>>();
            dr.classes = t[2].cast<std::vector<int>>();
            dr.scores = t[3].cast<std::vector<float>>();
            return dr;
        }));

    // Bind OBBResult structure
    pybind11::class_<OBBResult, DetResult>(m, "OBBResult")
        .def(pybind11::init<>())
        .def_readwrite("boxes", &OBBResult::boxes)
        .def("__copy__", [](const OBBResult &self) {
            return OBBResult(self);
        })
        .def("__deepcopy__", [](const OBBResult &self, pybind11::dict) {
            return OBBResult(self);
        })
        .def("__str__", [](const OBBResult &obr) {
            std::ostringstream oss;
            oss << "OBBResult(num=" << obr.num << ", classes=[";
            for (size_t i = 0; i < obr.classes.size(); ++i) {
                oss << obr.classes[i];
                if (i != obr.classes.size() - 1) oss << ", ";
            }
            oss << "], scores=[";
            for (size_t i = 0; i < obr.scores.size(); ++i) {
                oss << obr.scores[i];
                if (i != obr.scores.size() - 1) oss << ", ";
            }
            oss << "], boxes=[\n";
            for (const auto &rbox : obr.boxes) {
                oss << "    RotatedBox(left=" << rbox.left << ", top=" << rbox.top
                    << ", right=" << rbox.right << ", bottom=" << rbox.bottom
                    << ", theta=" << rbox.theta << "),\n";
            }
            oss << "  ])";
            return oss.str();
        })
        .def(pybind11::pickle([](const OBBResult &obr) {
            return pybind11::make_tuple(obr.num, obr.boxes, obr.classes, obr.scores);
        },
        [](pybind11::tuple t) {
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            OBBResult obr;
            obr.num = t[0].cast<int>();
            obr.boxes = t[1].cast<std::vector<RotatedBox>>();
            obr.classes = t[2].cast<std::vector<int>>();
            obr.scores = t[3].cast<std::vector<float>>();
            return obr;
        }));
}

// Bind inference class template
template <typename ClassType>
void BindClsTemplate(pybind11::module& m, const std::string& className) {
    pybind11::class_<ClassType>(m, className.c_str())
        .def(pybind11::init<const std::string &, bool, int>(),
             pybind11::arg("file"), pybind11::arg("cudaMem") = false, pybind11::arg("device") = 0)
        .def(
            "predict", [](ClassType &self, pybind11::array &pyimg) {
                Image image = PyArray2Image(pyimg);
                return self.predict(image);
            },
            "Predict the result from a single image")
        .def(
            "predict", [](ClassType &self, std::vector<pybind11::array> &pyimgs) {
                std::vector<Image> images;
                for (auto &pyimg : pyimgs) {
                    images.push_back(PyArray2Image(pyimg));
                }
                return self.predict(images);
            },
            "Predict the results from a list of images")
        .def_readwrite("batch", &ClassType::batch, "Batch size for prediction");
}

// Bind inference classes
void BindInference(pybind11::module &m) {
    m.doc() = "Bindings for inference classes like DeployDet, DeployCGDet, DeployOBB and DeployCGOBB";

    // 绑定 DeployDet 类
    BindClsTemplate<DeployDet>(m, "DeployDet");

    // 绑定 DeployCGDet 类
    BindClsTemplate<DeployCGDet>(m, "DeployCGDet");

    // 绑定 DeployOBB 类
    BindClsTemplate<DeployOBB>(m, "DeployOBB");

    // 绑定 DeployCGOBB 类
    BindClsTemplate<DeployCGOBB>(m, "DeployCGOBB");
}

// Define the pydeploy module
PYBIND11_MODULE(pydeploy, m) {
    m.doc() = "TensorRT-YOLO Python bindings using Pybind11";

    pybind11::module timer_module = m.def_submodule("timer", "Timer module of TensorRT-YOLO.");
    BindUtils(timer_module);

    pybind11::module result_module = m.def_submodule("result", "Result module of TensorRT-YOLO.");
    BindResult(result_module);

    pybind11::module inference_module = m.def_submodule("inference", "Inference module of TensorRT-YOLO.");
    BindInference(inference_module);
}

}  // namespace deploy
