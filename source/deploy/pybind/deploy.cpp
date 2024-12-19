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

// Convert Mask to NumPy array
pybind11::array_t<uint8_t> Mask2PyArray(const deploy::Mask &mask) {
    if (mask.width <= 0 || mask.height <= 0) {
        throw std::invalid_argument("Mask dimensions must be positive.");
    }
    if (static_cast<size_t>(mask.width * mask.height) != mask.data.size()) {
        throw std::invalid_argument("Data size does not match the specified width and height.");
    }

    return pybind11::array_t<uint8_t>(pybind11::buffer_info(
        const_cast<uint8_t *>(mask.data.data()),
        sizeof(uint8_t),
        pybind11::format_descriptor<uint8_t>::format(),
        2,
        {mask.height, mask.width},
        {sizeof(uint8_t) * mask.width, sizeof(uint8_t)}));
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

    // Bind KeyPoint structure
    pybind11::class_<KeyPoint>(m, "KeyPoint")
        .def(pybind11::init<>())
        .def(pybind11::init<float, float, std::optional<float>>(),
             pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("conf") = std::nullopt)
        .def_readwrite("x", &KeyPoint::x)
        .def_readwrite("y", &KeyPoint::y)
        .def_property("conf", [](const KeyPoint &kp) { return kp.conf; }, [](KeyPoint &kp, const std::optional<float> &value) { kp.conf = value; })
        .def("__str__", [](const KeyPoint &kp) {
        std::ostringstream oss;
        oss << "KeyPoint(x=" << kp.x << ", y=" << kp.y;
        if (kp.conf) {
            oss << ", conf=" << *kp.conf;
        } else {
            oss << ", conf=None";
        }
        oss << ")";
        return oss.str(); });

    // Bind ClsResult structure
    pybind11::class_<ClsResult>(m, "ClsResult")
        .def(pybind11::init<>())
        .def_readonly("num", &ClsResult::num)
        .def_readwrite("classes", &ClsResult::classes)
        .def_readwrite("scores", &ClsResult::scores)
        .def("__copy__", [](const ClsResult &self) {
            return ClsResult(self);
        })
        .def("__deepcopy__", [](const ClsResult &self, pybind11::dict) {
            return ClsResult(self);
        })
        .def("__str__", [](const ClsResult &dr) {
            std::ostringstream oss;
            oss << "ClsResult(\n    classes=[";
            for (size_t i = 0; i < dr.classes.size(); ++i) {
                oss << dr.classes[i];
                if (i != dr.classes.size() - 1) oss << ", ";
            }
            oss << "],\n    scores=[";
            for (size_t i = 0; i < dr.scores.size(); ++i) {
                oss << dr.scores[i];
                if (i != dr.scores.size() - 1) oss << ", ";
            }
            oss << "]\n)";
            return oss.str();
        })
        .def(pybind11::pickle([](const ClsResult &dr) { return pybind11::make_tuple(dr.classes, dr.scores); }, [](pybind11::tuple t) {
            if (t.size() != 2)
                throw std::runtime_error("Invalid state!");

            ClsResult dr;
            dr.classes = t[0].cast<std::vector<int>>();
            dr.scores = t[1].cast<std::vector<float>>();
            return dr; }));

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
        .def(pybind11::pickle([](const DetResult &dr) { return pybind11::make_tuple(dr.num, dr.boxes, dr.classes, dr.scores); }, [](pybind11::tuple t) {
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            DetResult dr;
            dr.num = t[0].cast<int>();
            dr.boxes = t[1].cast<std::vector<Box>>();
            dr.classes = t[2].cast<std::vector<int>>();
            dr.scores = t[3].cast<std::vector<float>>();
            return dr; }));

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
        .def(pybind11::pickle([](const OBBResult &obr) { return pybind11::make_tuple(obr.num, obr.boxes, obr.classes, obr.scores); }, [](pybind11::tuple t) {
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            OBBResult obr;
            obr.num = t[0].cast<int>();
            obr.boxes = t[1].cast<std::vector<RotatedBox>>();
            obr.classes = t[2].cast<std::vector<int>>();
            obr.scores = t[3].cast<std::vector<float>>();
            return obr; }));

    // Bind SegResult structure
    pybind11::class_<SegResult, DetResult>(m, "SegResult")
        .def(pybind11::init<>())
        .def_property(
            "masks",
            // Getter: Convert masks to list of numpy arrays
            [](const SegResult &sgr) {
                pybind11::list masks_list;
                for (const auto &mask : sgr.masks) {
                    masks_list.append(Mask2PyArray(mask));
                }
                return masks_list;
            },
            // Setter: Convert list of numpy arrays back to Mask
            [](SegResult &sgr, pybind11::list masks_list) {
                std::vector<Mask> masks_vec;
                for (auto item : masks_list) {
                    auto arr = item.cast<pybind11::array_t<uint8_t>>();
                    auto buf = arr.request();
                    if (buf.ndim != 2) {
                        throw std::invalid_argument("Each mask must be a 2D numpy array.");
                    }
                    deploy::Mask mask;
                    mask.width  = buf.shape[1];
                    mask.height = buf.shape[0];
                    mask.data.assign(static_cast<uint8_t *>(buf.ptr), static_cast<uint8_t *>(buf.ptr) + buf.size);
                    masks_vec.push_back(std::move(mask));
                }
                sgr.masks = std::move(masks_vec);
            })
        .def("__copy__", [](const SegResult &self) {
            return SegResult(self);
        })
        .def("__deepcopy__", [](const SegResult &self, pybind11::dict) {
            return SegResult(self);
        })
        .def("__str__", [](const SegResult &sgr) {
            std::ostringstream oss;
            oss << "SegResult(num=" << sgr.num << ", classes=[";
            for (size_t i = 0; i < sgr.classes.size(); ++i) {
                oss << sgr.classes[i];
                if (i != sgr.classes.size() - 1) oss << ", ";
            }
            oss << "], scores=[";
            for (size_t i = 0; i < sgr.scores.size(); ++i) {
                oss << sgr.scores[i];
                if (i != sgr.scores.size() - 1) oss << ", ";
            }
            oss << "], boxes=[\n";
            for (const auto &box : sgr.boxes) {
                oss << "    Box(left=" << box.left << ", top=" << box.top
                    << ", right=" << box.right << ", bottom=" << box.bottom << "),\n";
            }
            oss << "  ], masks=[";
            for (size_t i = 0; i < sgr.masks.size(); ++i) {
                oss << "    Mask(width=" << sgr.masks[i].width << ", height=" << sgr.masks[i].height << "),\n";
            }
            oss << "  ])";
            return oss.str();
        })
        .def(pybind11::pickle([](const SegResult &sgr) {
            pybind11::list masks_list;
            for (const auto &mask : sgr.masks) {
                masks_list.append(Mask2PyArray(mask));
            }
            return pybind11::make_tuple(
                sgr.num,
                sgr.boxes,
                sgr.classes,
                sgr.scores,
                masks_list); }, [](pybind11::tuple t) {
            if (t.size() != 5)
                throw std::runtime_error("Invalid state!");

            SegResult sgr;
            sgr.num = t[0].cast<int>();
            sgr.boxes = t[1].cast<std::vector<Box>>();
            sgr.classes = t[2].cast<std::vector<int>>();
            sgr.scores = t[3].cast<std::vector<float>>();

            pybind11::list masks_list = t[4].cast<pybind11::list>();
            for (auto item : masks_list) {
                auto arr = item.cast<pybind11::array_t<uint8_t>>();
                auto buf = arr.request();
                deploy::Mask mask;
                mask.width = buf.shape[1];
                mask.height = buf.shape[0];
                mask.data.assign(static_cast<uint8_t *>(buf.ptr), static_cast<uint8_t *>(buf.ptr) + buf.size);
                sgr.masks.push_back(std::move(mask));
            }
            return sgr; }));

    // Bind PoseResult structure
    pybind11::class_<PoseResult, DetResult>(m, "PoseResult")
        .def(pybind11::init<>())
        .def_readwrite("kpts", &PoseResult::kpts)
        .def("__copy__", [](const PoseResult &self) {
            return PoseResult(self);
        })
        .def("__deepcopy__", [](const PoseResult &self, pybind11::dict) {
            return PoseResult(self);
        })
        .def("__str__", [](const PoseResult &pr) {
            std::ostringstream oss;
            oss << "PoseResult(num=" << pr.num << ", classes=[";
            for (size_t i = 0; i < pr.classes.size(); ++i) {
                oss << pr.classes[i];
                if (i != pr.classes.size() - 1) oss << ", ";
            }
            oss << "], scores=[";
            for (size_t i = 0; i < pr.scores.size(); ++i) {
                oss << pr.scores[i];
                if (i != pr.scores.size() - 1) oss << ", ";
            }
            oss << "], boxes=[\n";
            for (size_t i = 0; i < pr.boxes.size(); ++i) {
                oss << "    Box(left=" << pr.boxes[i].left << ", top=" << pr.boxes[i].top
                    << ", right=" << pr.boxes[i].right << ", bottom=" << pr.boxes[i].bottom << "),\n";
            }
            oss << "], kpts=[\n";
            for (size_t i = 0; i < pr.kpts.size(); ++i) {
                oss << "    [";
                for (size_t j = 0; j < pr.kpts[i].size(); ++j) {
                    oss << "KeyPoint(x=" << pr.kpts[i][j].x << ", y=" << pr.kpts[i][j].y;
                    if (pr.kpts[i][j].conf) {
                        oss << ", conf=" << *pr.kpts[i][j].conf;
                    } else {
                        oss << ", conf=None";
                    }
                    oss << ")";
                    if (j != pr.kpts[i].size() - 1) oss << ", ";
                }
                oss << "]";
                if (i != pr.kpts.size() - 1) oss << ",\n";
            }
            oss << "])";
            return oss.str();
        })
        .def(pybind11::pickle([](const PoseResult &pr) { return pybind11::make_tuple(pr.num, pr.boxes, pr.classes, pr.scores, pr.kpts); }, [](pybind11::tuple t) {
            if (t.size() != 5)
                throw std::runtime_error("Invalid state!");

            PoseResult pr;
            pr.num = t[0].cast<int>();
            pr.boxes = t[1].cast<std::vector<Box>>();
            pr.classes = t[2].cast<std::vector<int>>();
            pr.scores = t[3].cast<std::vector<float>>();
            pr.kpts = t[4].cast<std::vector<std::vector<KeyPoint>>>();
            return pr; }));
}

// Bind inference class template
template <typename ClassType>
void BindClsTemplate(pybind11::module &m, const std::string &className) {
    pybind11::class_<ClassType, std::unique_ptr<ClassType>>(m, className.c_str())
        .def(pybind11::init<const std::string &, bool, int>(),
             pybind11::arg("file"), pybind11::arg("cudaMem") = false, pybind11::arg("device") = 0)
        .def(
            "predict", [](ClassType &self, pybind11::array &pyimg) {
                Image image = PyArray2Image(pyimg);
                return self.predict(image);
            },
            "Predict the result from a single image")
        .def("predict", [](ClassType &self, std::vector<pybind11::array> &pyimgs) {
                std::vector<Image> images;
                for (auto &pyimg : pyimgs) {
                    images.push_back(PyArray2Image(pyimg));
                }
                return self.predict(images); }, "Predict the results from a list of images")
        .def_readwrite("batch", &ClassType::batch, "Batch size for prediction");
}

// Bind inference classes
void BindInference(pybind11::module &m) {
    m.doc() = "Bindings for inference classes, including DeployDet, DeployCGDet, DeployOBB, DeployCGOBB, DeploySeg, DeployCGSeg, DeployPose, DeployCGPose, DeployCls, and DeployCGCls.";

    // bind DeployDet
    BindClsTemplate<DeployDet>(m, "DeployDet");

    // bind DeployCGDet
    BindClsTemplate<DeployCGDet>(m, "DeployCGDet");

    // bind DeployOBB
    BindClsTemplate<DeployOBB>(m, "DeployOBB");

    // bind DeployCGOBB
    BindClsTemplate<DeployCGOBB>(m, "DeployCGOBB");

    // bind DeploySeg
    BindClsTemplate<DeploySeg>(m, "DeploySeg");

    // bind DeployCGSeg
    BindClsTemplate<DeployCGSeg>(m, "DeployCGSeg");

    // bind DeploySeg
    BindClsTemplate<DeployPose>(m, "DeployPose");

    // bind DeployCGSeg
    BindClsTemplate<DeployCGPose>(m, "DeployCGPose");

    // bind DeployCls
    BindClsTemplate<DeployCls>(m, "DeployCls");

    // bind DeployCGCls
    BindClsTemplate<DeployCGCls>(m, "DeployCGCls");
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
