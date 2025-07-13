#-------------------------------------------------------------------------------
# 自定义插件 custom_plugins
#-------------------------------------------------------------------------------
add_library(custom_plugins SHARED)

# 添加头文件搜索路径
target_include_directories(custom_plugins PRIVATE
    ${PROJECT_SOURCE_DIR}/plugin
)

# 定义插件源文件目录
set(PLUGIN_DIRS
    ${PROJECT_SOURCE_DIR}/plugin/common
    ${PROJECT_SOURCE_DIR}/plugin/efficientRotatedNMSPlugin
    ${PROJECT_SOURCE_DIR}/plugin/efficientIdxNMSPlugin
)

# 使用更可控的方式收集源文件（替代 GLOB_RECURSE）
set(SOURCES)
foreach(PLUGIN_DIR ${PLUGIN_DIRS})
    # 显式列出源文件比 GLOB 更可靠（实际项目中推荐）
    file(GLOB CPP_FILES "${PLUGIN_DIR}/*.cpp")
    file(GLOB CU_FILES "${PLUGIN_DIR}/*.cu")
    list(APPEND SOURCES ${CPP_FILES} ${CU_FILES})
endforeach()

# 将搜索到的文件添加到目标中
target_sources(custom_plugins PRIVATE ${SOURCES})

# 配置设置
configure_cuda_trt_target(custom_plugins)
set_target_compile_options(custom_plugins)
set_target_properties(custom_plugins PROPERTIES
    OUTPUT_NAME "custom_plugins"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/plugin"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/plugin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/lib/plugin"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/lib/plugin"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/lib/plugin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/lib/plugin"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/lib/plugin"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/lib/plugin"
    CXX_VISIBILITY_PRESET "hidden"
)