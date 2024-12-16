# 设置项目
cmake_minimum_required(VERSION 3.15.0)                  # 设置CMake的最低版本要求
cmake_policy(SET CMP0091 NEW)                           # 允许在CMake 3.10+中自动设置项目名作为二进制目录名
cmake_policy(SET CMP0146 OLD)                           # 忽略对find_package的过时警告
project(TensorRT-YOLO VERSION 5.0.2 LANGUAGES CXX CUDA) # 定义项目名称、版本和使用的编程语言（C++和CUDA）

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)                              # 设置C++标准为17
set(CMAKE_CXX_STANDARD_REQUIRED ON)                     # 要求C++标准必须满足17

# 添加编译规则
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)                   # 生成编译数据库，便于代码分析工具使用

# 添加依赖项
## CUDA
find_package(CUDA REQUIRED)                             # 查找CUDA包
set(CMAKE_CUDA_ARCHITECTURES native)                    # 自动检测最佳CUDA架构
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})                 # 缓存CUDA路径

## Pybind11
set(PYBIND11_FINDPYTHON ON)                                                            # 开启寻找Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)                       # 查找Python解释器和开发组件
find_package(pybind11 CONFIG REQUIRED)                                                 # 查找pybind11包

# 添加编译选项
option(TENSORRT_PATH "TensorRT Path. Example: /usr/local/tensorrt" "")                 # 添加编译选项，用于指定TensorRT路径
# 检查 TensorRT 路径是否已设置
if(NOT TENSORRT_PATH)
    message(FATAL_ERROR "TensorRT path is not set. Please specify the TensorRT path.") # 如果未设置，则报错
endif()

# 添加公共配置
function(configure_cuda_trt target)
    # 添加CUDA定义、包含目录、链接库
    target_compile_definitions(${target} PRIVATE ${CUDA_DEFINITIONS})
    target_include_directories(${target} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${CUDA_cudart_LIBRARY})

    # 添加TensorRT的包含目录、库目录、链接库
    target_include_directories(${target} PRIVATE ${TENSORRT_PATH}/include)
    target_link_directories(${target} PRIVATE ${TENSORRT_PATH}/lib)
    if(MSVC AND EXISTS ${TENSORRT_PATH}/lib/nvinfer_10.dll)
        target_link_libraries(${target} PRIVATE nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
    else()
        target_link_libraries(${target} PRIVATE nvinfer nvinfer_plugin nvonnxparser)
    endif()
endfunction()

function(add_compile_files target)
    # 添加头文件搜索路径
    include_directories(${PROJECT_SOURCE_DIR}/include) 
    file(GLOB_RECURSE SOURCES
        ${PROJECT_SOURCE_DIR}/source/deploy/core/*.cpp
        ${PROJECT_SOURCE_DIR}/source/deploy/utils/*.cpp
        ${PROJECT_SOURCE_DIR}/source/deploy/vision/*.cpp
        ${PROJECT_SOURCE_DIR}/source/deploy/vision/*.cu
    )
    # 添加源文件
    target_sources(${target} PRIVATE ${SOURCES}) 
endfunction()

function(set_compile_options target)

    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O2>)                                   # 对于MSVC，在Release模式下使用-O2优化
        set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>") # 设置MSVC运行时库
    else()
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)                  # 对于非MSVC，使用-O3优化和自动LTO
        target_link_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)                     # 链接时同样使用-O3优化和自动LTO
    endif()

endfunction()

# 添加子目录
add_subdirectory(${PROJECT_SOURCE_DIR}/plugin)                                                        # 添加插件目录

# 定义目标 deploy
add_library(deploy SHARED)                                                                            # 创建共享库
add_compile_files(deploy)                                                                             # 添加源文件
configure_cuda_trt(deploy)                                                                            # 配置CUDA和TensorRT
set_compile_options(deploy)                                                                           # 设置编译选项
set_target_properties(deploy PROPERTIES OUTPUT_NAME deploy)                                           # 设置输出名称
if(MSVC)
    set_target_properties(deploy PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/lib) # 设置Release模式下输出目录
    set_target_properties(deploy PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/lib)
    set_target_properties(deploy PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/lib)   # 设置Debug模式下输出目录
    set_target_properties(deploy PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/lib)
else()
    set_target_properties(deploy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)         # 设置库文件输出目录
endif()

# 定义目标 pydeploy
pybind11_add_module(pydeploy ${PROJECT_SOURCE_DIR}/source/deploy/pybind/deploy.cpp)                   # 创建pybind11模块
add_compile_files(pydeploy)                                                                           # 添加源文件
configure_cuda_trt(pydeploy)                                                                          # 配置CUDA和TensorRT
set_compile_options(pydeploy)                                                                         # 设置编译选项
set_target_properties(pydeploy PROPERTIES OUTPUT_NAME pydeploy)                                       # 设置输出名称
if(MSVC)
    set_target_properties(pydeploy PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/tensorrt_yolo/libs) # 设置Release模式下输出目录
    set_target_properties(pydeploy PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/tensorrt_yolo/libs)   # 设置Debug模式下输出目录
else()
    set_target_properties(pydeploy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorrt_yolo/libs)         # 设置库文件输出目录
endif()
set_target_properties(pydeploy PROPERTIES
    COMPILE_DEFINITIONS "CUDA_PATH=\"${CUDA_TOOLKIT_ROOT_DIR}\""
    COMPILE_DEFINITIONS "TENSORRT_PATH=\"${TENSORRT_PATH}\""
) # 设置编译定义
configure_file(
    ${CMAKE_SOURCE_DIR}/tensorrt_yolo/c_lib_wrap.py.in
    ${CMAKE_SOURCE_DIR}/tensorrt_yolo/c_lib_wrap.py
) # 配置文件