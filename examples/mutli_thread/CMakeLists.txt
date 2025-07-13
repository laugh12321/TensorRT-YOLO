cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0091 NEW)  # 确保 MSVC 运行时库策略正确

#-------------------------------------------------------------------------------
# 项目基础配置
#-------------------------------------------------------------------------------
project(mutli_thread LANGUAGES CXX)

# 生成编译数据库（供clangd等工具使用）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#-------------------------------------------------------------------------------
# 依赖配置
#-------------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(TensorRT-YOLO REQUIRED)

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
# 可执行文件
#-------------------------------------------------------------------------------
add_executable(mutli_thread "mutli_thread.cpp")

# 包含头文件目录
target_include_directories(mutli_thread PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT-YOLO_INCLUDE_DIRS}
)

# 私有链接库
target_link_libraries(mutli_thread PRIVATE
    ${OpenCV_LIBS}
    ${TensorRT-YOLO_LIBs}
)

set_target_compile_options(mutli_thread)

# 模块配置
set_target_properties(mutli_thread PROPERTIES
    OUTPUT_NAME "mutli_thread"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin"
)