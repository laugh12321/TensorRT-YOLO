cmake_minimum_required(VERSION 3.17)
cmake_policy(SET CMP0091 NEW)  # 确保 MSVC 运行时库策略正确

#-------------------------------------------------------------------------------
# 项目基础配置
#-------------------------------------------------------------------------------
project(TensorRT-YOLO
    VERSION 6.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "🚀 Easier & Faster YOLO Deployment Toolkit for NVIDIA 🛠️"
)

# 生成编译数据库（供clangd等工具使用）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#-------------------------------------------------------------------------------
# 依赖配置
#-------------------------------------------------------------------------------
# CUDA 配置
find_package(CUDAToolkit REQUIRED)

# 允许用户覆盖默认的 CUDA 架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89;86;75;70;61")  # 默认支持架构
endif()

# TensorRT 路径配置
set(TRT_PATH "" CACHE PATH "Path to TensorRT installation (e.g., /usr/local/tensorrt)")
if(NOT TRT_PATH)
    message(FATAL_ERROR "TensorRT path must be specified with -DTRT_PATH=/path/to/tensorrt")
endif()

# 定义 TensorRT 库的版本选择
set(TRT_LIB_DIR "${TRT_PATH}/lib")
if(MSVC AND EXISTS "${TRT_LIB_DIR}/nvinfer_10.dll")
    set(TRT_LIBS nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
else()
    set(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
endif()

# Python 绑定选项
option(BUILD_PYTHON "Build Python bindings with pybind11" OFF)
if(BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

#-------------------------------------------------------------------------------
# 编译工具链配置
#-------------------------------------------------------------------------------
function(set_target_compile_options target)
    # MSVC 配置
    if(MSVC)
        # C++编译选项
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /Oi /fp:fast>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od /Ob0 /Zi /RTC1>
        )

        # CUDA编译选项
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/O2 -O3>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/Od -g -G>
        )

        set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY
            "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>"
        )

    # GCC/Clang 配置
    else()
        # C++编译选项
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -march=native -flto=auto -DNDEBUG>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g3 -fno-omit-frame-pointer -fno-inline>
        )

        # CUDA编译选项
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
        )

        target_link_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -flto=auto>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g3>
        )
    endif()

    # 跨平台宏定义
    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    )
endfunction()

#-------------------------------------------------------------------------------
# TensorRT/CUDA 目标配置
#-------------------------------------------------------------------------------
function(configure_cuda_trt_target target)
    # 包含目录
    target_include_directories(${target} PRIVATE
        ${TRT_PATH}/include
    )

    # 链接目录
    target_link_directories(${target} PRIVATE
        ${TRT_LIB_DIR}
    )

    # 添加链接库
    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        ${TRT_LIBS}
    )

    # CUDA 特性
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

#-------------------------------------------------------------------------------
# 收集源文件（使用 CONFIGURE_DEPENDS 自动更新）
#-------------------------------------------------------------------------------
function(add_target_compile_files target)
    # 添加项目源码目录到目标的私有包含目录
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR})

    # 收集源文件
    file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/deploy/core/*.cpp"
        "${PROJECT_SOURCE_DIR}/deploy/utils/*.cpp"
        "${PROJECT_SOURCE_DIR}/deploy/infer/*.cpp"
        "${PROJECT_SOURCE_DIR}/deploy/infer/*.cu"
        "${PROJECT_SOURCE_DIR}/deploy/model.cpp"
    )

    # 将收集到的源文件添加到目标
    target_sources(${target} PRIVATE ${SOURCES})
endfunction()

#-------------------------------------------------------------------------------
# 主库目标
#-------------------------------------------------------------------------------
# 创建主库
add_library(deploy SHARED)
add_target_compile_files(deploy)
configure_cuda_trt_target(deploy)
set_target_compile_options(deploy)

# 模块配置
set_target_properties(deploy PROPERTIES
    OUTPUT_NAME "deploy"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/lib"
    CXX_VISIBILITY_PRESET "hidden"
)

#-------------------------------------------------------------------------------
# Python 绑定（可选）
#-------------------------------------------------------------------------------
if(BUILD_PYTHON)
    pybind11_add_module(pydeploy ${PROJECT_SOURCE_DIR}/deploy/pybind.cpp)
    add_target_compile_files(pydeploy)
    configure_cuda_trt_target(pydeploy)
    set_target_compile_options(pydeploy)

    # Python 模块配置
    set_target_properties(pydeploy PROPERTIES
        OUTPUT_NAME "pydeploy"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/libs"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/libs"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/libs"
        CXX_VISIBILITY_PRESET "hidden"
    )

    # Python 配置文件
    configure_file(
        ${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/c_lib_wrap.py.in
        ${PROJECT_SOURCE_DIR}/python/tensorrt_yolo/c_lib_wrap.py
    )
endif()

#-------------------------------------------------------------------------------
# 子项目
#-------------------------------------------------------------------------------
add_subdirectory(plugin)