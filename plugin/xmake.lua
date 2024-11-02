target("custom_plugins")
    -- 设置目标类型
    set_kind("shared")

    -- 设置编译路径
    set_targetdir("$(projectdir)/lib/plugin")

    -- 添加库目录
    add_includedirs("$(projectdir)/plugin")

    -- 添加文件
    add_files(
        "$(projectdir)/plugin/common/*.cpp",
        "$(projectdir)/plugin/common/*.cu",
        "$(projectdir)/plugin/efficientRotatedNMSPlugin/*.cpp",
        "$(projectdir)/plugin/efficientRotatedNMSPlugin/*.cu",
        "$(projectdir)/plugin/efficientIdxNMSPlugin/*.cpp",
        "$(projectdir)/plugin/efficientIdxNMSPlugin/*.cu"
    )

    -- 添加 cuda
    configure_cuda("custom_plugins")

    -- 添加TensorRT链接目录和链接库
    configure_tensorrt("custom_plugins")