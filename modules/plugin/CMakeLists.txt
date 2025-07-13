#-------------------------------------------------------------------------------
# 自定义插件 custom_plugins
#-------------------------------------------------------------------------------
add_library(custom_plugins SHARED)

# 添加头文件搜索路径
target_include_directories(custom_plugins PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 定义插件源文件目录
set(PLUGIN_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/efficientRotatedNMSPlugin
    ${CMAKE_CURRENT_SOURCE_DIR}/efficientIdxNMSPlugin
)

# 使用更可控的方式收集源文件（替代 GLOB_RECURSE）
set(SOURCES)
foreach(PLUGIN_DIR ${PLUGIN_DIRS})
    # 显式列出源文件比 GLOB 更可靠（实际项目中推荐）
    file(GLOB CPP_FILES "${PLUGIN_DIR}/*.cpp")
    file(GLOB CU_FILES "${PLUGIN_DIR}/*.cu")
    list(APPEND SOURCES ${CPP_FILES} ${CU_FILES})
endforeach()

# 将源文件添加到目标中
target_sources(custom_plugins PRIVATE ${SOURCES})

# 配置 TensorRT 和 CUDA 目标
configure_cuda_trt_target(custom_plugins)
set_target_compile_options(custom_plugins)

# 设置目标属性
set_target_properties(custom_plugins PROPERTIES
    OUTPUT_NAME "custom_plugins"  # 设置输出库的名称
    CXX_VISIBILITY_PRESET "hidden"  # 默认隐藏符号，减少符号冲突
)

# 安装 custom_plugins 库
install(TARGETS custom_plugins
    EXPORT custom_pluginsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  # 安装到 lib 目录
    # ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}  # 安装到 lib 目录
    # RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}  # 安装到 lib 目录
)
