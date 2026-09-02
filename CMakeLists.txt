cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0091 NEW)  # 确保 MSVC 运行时库策略正确

#-------------------------------------------------------------------------------
# 项目基础配置
#-------------------------------------------------------------------------------
project(TensorRT-YOLO
    VERSION 6.4.0
    LANGUAGES CXX CUDA
    DESCRIPTION "🚀 Easier & Faster YOLO Deployment Toolkit for NVIDIA 🛠️"
)

# 生成编译数据库（供clangd等工具使用）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 使用 CMake 提供的默认值
set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include")

# 修复 undefined symbol: fatbinData
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

#-------------------------------------------------------------------------------
# 依赖配置
#-------------------------------------------------------------------------------
# CUDA 配置
find_package(CUDAToolkit REQUIRED)

# 默认的 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 72 75 80 86 87 89 90 CACHE STRING "CUDA architectures" FORCE)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100 120)
endif()
message(STATUS "CUDA code for default SMs: ${CMAKE_CUDA_ARCHITECTURES}")

# TensorRT 路径配置
set(TRT_PATH "" CACHE PATH "Path to TensorRT installation (e.g., /usr/local/tensorrt)")
if(NOT TRT_PATH)
    message(FATAL_ERROR "TensorRT path must be specified with -DTRT_PATH=/path/to/tensorrt")
endif()

# Linux：.so 始终在 lib/。
# Windows：导入库 .lib 在 lib/（或 lib/x64）；运行时 DLL 在
#   - 高版本 zip（TRT 10.10+）：bin/
#   - 旧 zip：与 .lib 同在 lib/
set(TRT_IMPLIB_DIRS "${TRT_PATH}/lib")
set(TRT_LIB_DIR "${TRT_PATH}/lib")
if(WIN32)
    if(EXISTS "${TRT_PATH}/lib/x64")
        list(APPEND TRT_IMPLIB_DIRS "${TRT_PATH}/lib/x64")
        set(TRT_LIB_DIR "${TRT_PATH}/lib/x64")
    endif()
    if(EXISTS "${TRT_PATH}/bin")
        list(APPEND TRT_IMPLIB_DIRS "${TRT_PATH}/bin")
    endif()
endif()

set(TRT_BIN_DIR "")
if(WIN32)
    foreach(_trt_dll nvinfer_11.dll nvinfer_10.dll nvinfer.dll)
        if(NOT TRT_BIN_DIR AND EXISTS "${TRT_PATH}/bin/${_trt_dll}")
            set(TRT_BIN_DIR "${TRT_PATH}/bin")
        endif()
        if(NOT TRT_BIN_DIR AND EXISTS "${TRT_PATH}/lib/${_trt_dll}")
            set(TRT_BIN_DIR "${TRT_PATH}/lib")
        endif()
        if(NOT TRT_BIN_DIR AND EXISTS "${TRT_PATH}/lib/x64/${_trt_dll}")
            set(TRT_BIN_DIR "${TRT_PATH}/lib/x64")
        endif()
    endforeach()
    if(NOT TRT_BIN_DIR)
        set(TRT_BIN_DIR "${TRT_LIB_DIR}")
    endif()
else()
    set(TRT_BIN_DIR "${TRT_PATH}/lib")
endif()
# c_lib_wrap.py.in 用 TRT_LIB_DIR 做 Windows DLL 搜索路径，必须指向运行时目录
set(TRT_LIB_DIR "${TRT_BIN_DIR}")

set(_trt_use_suffix "")
foreach(_d ${TRT_IMPLIB_DIRS} ${TRT_BIN_DIR})
    if(NOT _d)
        continue()
    endif()
    if(NOT _trt_use_suffix AND (EXISTS "${_d}/nvinfer_11.lib" OR EXISTS "${_d}/nvinfer_11.dll"))
        set(_trt_use_suffix 11)
    endif()
    if(NOT _trt_use_suffix AND (EXISTS "${_d}/nvinfer_10.lib" OR EXISTS "${_d}/nvinfer_10.dll"))
        set(_trt_use_suffix 10)
    endif()
endforeach()
if(WIN32 AND _trt_use_suffix)
    set(TRT_LIBS nvinfer_${_trt_use_suffix} nvinfer_plugin_${_trt_use_suffix} nvonnxparser_${_trt_use_suffix})
else()
    set(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
endif()
unset(_trt_use_suffix)

set(TRT_LIB_FILES)
foreach(_trt_lib ${TRT_LIBS})
    unset(_TRT_LIB_PATH CACHE)
    find_library(_TRT_LIB_PATH
        NAMES ${_trt_lib}
        HINTS ${TRT_IMPLIB_DIRS}
        NO_DEFAULT_PATH)
    if(NOT _TRT_LIB_PATH)
        find_library(_TRT_LIB_PATH NAMES ${_trt_lib} HINTS ${TRT_IMPLIB_DIRS} REQUIRED)
    endif()
    list(APPEND TRT_LIB_FILES "${_TRT_LIB_PATH}")
endforeach()
unset(_TRT_LIB_PATH CACHE)
message(STATUS "TensorRT implib dirs: ${TRT_IMPLIB_DIRS}")
message(STATUS "TensorRT runtime dir: ${TRT_BIN_DIR}")
message(STATUS "TensorRT libs: ${TRT_LIB_FILES}")

# Python 绑定选项
option(BUILD_PYTHON "Build Python bindings with pybind11" OFF)
if(BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED QUIET)
    if(NOT pybind11_FOUND)
        message(WARNING "pybind11 not found. Python bindings will not be built.")
        set(BUILD_PYTHON OFF)
    endif()
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

    # 添加链接库
    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        ${TRT_LIB_FILES}
    )

    # CUDA 特性
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # 确保目标使用正确的 C++ 标准
    target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

#-------------------------------------------------------------------------------
# 主库目标
#-------------------------------------------------------------------------------
add_subdirectory(modules/trtyolo)

#-------------------------------------------------------------------------------
# 插件
#-------------------------------------------------------------------------------
add_subdirectory(modules/plugin)