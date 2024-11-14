# 设置项目
cmake_minimum_required(VERSION 3.15.0)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0146 OLD)
project(detect LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加依赖项
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

find_package(OpenCV REQUIRED)

include(FindPkgConfig)
pkg_check_modules(GST   REQUIRED gstreamer-1.0)

# 添加编译选项
option(TENSORRT_PATH "TensorRT Path. Example: /usr/local/tensorrt" "")
if(NOT DEFINED TENSORRT_PATH)
    message(FATAL_ERROR "TensorRT path is not set. Please specify the TensorRT path.")
endif()

option(DEPLOY_PATH "TensorRT-YOLO Project Path." "")
if(NOT DEFINED DEPLOY_PATH)
    message(FATAL_ERROR "TensorRT-YOLO project path is not set. Please specify the TensorRT-YOLO Project path.")
endif()

option(VIDEOPIPE_PATH "VideoPipe Project Path." "")
if(NOT DEFINED VIDEOPIPE_PATH)
    message(FATAL_ERROR "VideoPipe project path is not set. Please specify the VideoPipe Project path.")
endif()

# 添加可执行文件
add_executable(PipeDemo "")

# 包含头文件目录
target_include_directories(PipeDemo PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_PATH}/include
    ${OpenCV_INCLUDE_DIRS}
    ${DEPLOY_PATH}/include
    ${VIDEOPIPE_PATH}
    ${GST_INCLUDE_DIRS}
)

# 链接库目录
target_link_directories(PipeDemo PRIVATE
    ${TENSORRT_PATH}/lib
    ${DEPLOY_PATH}/lib
    ${VIDEOPIPE_PATH}/build/libs
)

# 私有源文件
target_sources(PipeDemo PRIVATE
    src/main.cpp
    src/vp_trtyolo_detector.cpp
)

# 私有编译定义
target_compile_definitions(PipeDemo PRIVATE ${CUDA_DEFINITIONS})

# 私有链接库
target_link_libraries(PipeDemo PRIVATE
    ${CUDA_cudart_LIBRARY}
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    deploy
    video_pipe
    tinyexpr
)

# 根据平台选择不同的 TensorRT 库
if(MSVC AND EXISTS ${TENSORRT_PATH}/lib/nvinfer_10.dll)
    target_link_libraries(PipeDemo PRIVATE nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
else()
    target_link_libraries(PipeDemo PRIVATE nvinfer nvinfer_plugin nvonnxparser)
endif()

# 设置输出目录
set_target_properties(PipeDemo PROPERTIES
    OUTPUT_NAME "PipeDemo"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/workspace"
)

# 编译选项
if(MSVC)
    target_compile_options(PipeDemo PRIVATE
        $<$<CONFIG:Release>:-O2>
    )
    set_property(TARGET PipeDemo PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_compile_options(PipeDemo PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>
    )
    target_link_options(PipeDemo PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>
    )
endif()