# 设置项目
cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0091 NEW)
project(TensorRT-YOLO VERSION 6.0.0 LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成编译数据库，便于代码分析工具使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 添加依赖项
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    set(CUDA_LIBRARIES ${CUDAToolkit_LIBRARIES})
else()
    find_package(CUDA REQUIRED)
endif()
set(CMAKE_CUDA_ARCHITECTURES 89;86;75;70;61)  # CUDA 架构版本 8.9, 8.6, 7.5, 7.0, 6.1

# 添加编译选项
option(TENSORRT_PATH "TensorRT Path. Example: /usr/local/tensorrt" "")
if(NOT TENSORRT_PATH)
    message(FATAL_ERROR "TensorRT path is not set. Please specify the TensorRT path.")
endif()

# 添加选项：是否编译Python版本
option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
    message(STATUS "Python bindings will be built.")

    # Pybind11 依赖
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
else()
    message(STATUS "Python bindings will NOT be built.")
endif()

# 配置CUDA和TensorRT的函数
function(configure_cuda_trt target)
    target_compile_definitions(${target} PRIVATE ${CUDA_DEFINITIONS})
    target_include_directories(${target} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${CUDA_cudart_LIBRARY})

    target_include_directories(${target} PRIVATE ${TENSORRT_PATH}/include)
    target_link_directories(${target} PRIVATE ${TENSORRT_PATH}/lib)
    if(MSVC AND EXISTS ${TENSORRT_PATH}/lib/nvinfer_10.dll)
        target_link_libraries(${target} PRIVATE nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
    else()
        target_link_libraries(${target} PRIVATE nvinfer nvinfer_plugin nvonnxparser)
    endif()
endfunction()

# 添加源文件的函数
function(add_compile_files target)
    include_directories(${PROJECT_SOURCE_DIR}) 
    file(GLOB_RECURSE SOURCES
        ${PROJECT_SOURCE_DIR}/deploy/core/*.cpp
        ${PROJECT_SOURCE_DIR}/deploy/utils/*.cpp
        ${PROJECT_SOURCE_DIR}/deploy/infer/*.cpp
        ${PROJECT_SOURCE_DIR}/deploy/infer/*.cu
        ${PROJECT_SOURCE_DIR}/deploy/model.cpp
    )
    target_sources(${target} PRIVATE ${SOURCES})
endfunction()

# 设置编译选项的函数
function(set_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O2>)
        set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)
        target_link_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)
    endif()
endfunction()

# 添加子目录
add_subdirectory(${PROJECT_SOURCE_DIR}/plugin)

# 定义目标 deploy
add_library(deploy SHARED)
add_compile_files(deploy)
configure_cuda_trt(deploy)
set_compile_options(deploy)
set_target_properties(deploy PROPERTIES OUTPUT_NAME deploy)

if(MSVC)
    set_target_properties(deploy PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib
    )
else()
    set_target_properties(deploy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
endif()

# 定义目标 pydeploy（仅在BUILD_PYTHON为ON时编译）
if(BUILD_PYTHON)
    pybind11_add_module(pydeploy ${PROJECT_SOURCE_DIR}/deploy/pybind.cpp)
    add_compile_files(pydeploy)
    configure_cuda_trt(pydeploy)
    set_compile_options(pydeploy)
    set_target_properties(pydeploy PROPERTIES OUTPUT_NAME pydeploy)

    if(MSVC)
        set_target_properties(pydeploy PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/libs
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/libs
        )
    else()
        set_target_properties(pydeploy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/libs)
    endif()

    set_target_properties(pydeploy PROPERTIES
        COMPILE_DEFINITIONS "CUDA_PATH=\"${CUDA_TOOLKIT_ROOT_DIR}\""
        COMPILE_DEFINITIONS "TENSORRT_PATH=\"${TENSORRT_PATH}\""
    )

    # 配置文件
    configure_file(
        ${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/c_lib_wrap.py.in
        ${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/c_lib_wrap.py
    )
endif()