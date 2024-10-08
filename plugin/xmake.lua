target("custom_plugins")
    set_languages("cxx17")

    -- 设置编译路径
    set_targetdir("$(projectdir)/lib")

    -- 添加库目录
    add_includedirs("$(projectdir)/plugin")

    -- 添加文件
    add_files(
        "$(projectdir)/plugin/common/*.cpp",
        "$(projectdir)/plugin/common/*.cu",
        "$(projectdir)/plugin/efficientRotatedNMSPlugin/*.cpp",
        "$(projectdir)/plugin/efficientRotatedNMSPlugin/*.cu"
    )

    -- 设置目标类型
    set_kind("shared")

    -- 添加 cuda
    add_rules("cuda")
    add_cugencodes("native")
    add_cuflags("-allow-unsupported-compiler")

    -- 添加TensorRT链接目录和链接库
    if has_config("tensorrt") then
        add_includedirs(path.join("$(tensorrt)", "include"))
        add_linkdirs(path.join("$(tensorrt)", "lib"))
        if is_host("windows") and os.exists(path.join(get_config("tensorrt"), "lib", "nvinfer_10.dll")) then
            add_links("nvinfer_10", "nvinfer_plugin_10", "nvonnxparser_10")
        else
            add_links("nvinfer", "nvinfer_plugin", "nvonnxparser")
        end
    end