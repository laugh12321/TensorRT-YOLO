add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")
add_rules("plugin.vsxmake.autoupdate")
add_requires("opencv", "cli11")
set_languages("cxx17")

-- CUDA
add_includedirs("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include")
add_linkdirs("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/lib/x64")
add_links("cuda", "cudart")

-- cuDNN
add_includedirs("C:/Program Files/NVIDIA GPU Computing Toolkit/cuDNN/v8.4.1.50/include")
add_linkdirs("C:/Program Files/NVIDIA GPU Computing Toolkit/cuDNN/v8.4.1.50/lib")
add_links("cudnn")

-- TensorRT
add_includedirs("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6/include")
add_linkdirs("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6/lib")
add_links("nvinfer", "nvinfer_plugin", "nvparsers", "nvonnxparser")

target("detect")
    set_kind("binary")
    add_packages("opencv", "cli11")
    add_headerfiles("include/*.hpp")
    add_files("detect.cpp", "include/*.cu")