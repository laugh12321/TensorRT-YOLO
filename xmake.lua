-- 设置项目信息
set_project("TensorRT-YOLO")
set_version("4.1.0")
set_languages("cxx17")
set_allowedplats("windows", "linux")

add_requires("pybind11")
add_requireconfs("pybind11.python", {version = "3.10", override = true})

-- 添加编译规则
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")

-- 定义选项
option("tensorrt")
    set_showmenu(true)
    set_description("TensorRT Path. Example: /usr/local/tensorrt")
    on_check(function (option)
        if not option:enabled() then 
            raise("TensorRT path is not set. Please specify the TensorRT path.")
        end 
    end)

-- 定义目标
target("deploy")
    -- 设置目标类型
    set_kind("$(kind)")

    -- 设置编译路径
    set_targetdir("$(projectdir)/lib")

    -- 添加库目录
    add_includedirs("$(projectdir)/include", {public = true})

    -- 添加文件
    add_files(
        "$(projectdir)/source/deploy/core/*.cpp",
        "$(projectdir)/source/deploy/utils/*.cpp",
        "$(projectdir)/source/deploy/vision/*.cpp",
        "$(projectdir)/source/deploy/vision/*.cu"
    )

    -- 添加 cuda
    add_rules("cuda")
    add_cugencodes("native")
    add_cuflags("-allow-unsupported-compiler")

    -- 如果目标类型是静态库
    if is_kind("static") then 
        -- 设置 CUDA 开发链接为 true
        set_policy("build.cuda.devlink", true)
    else
        add_defines("ENABLE_DEPLOY_BUILDING_DLL")
    end

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

-- 定义目标
target("pydeploy")
    -- 定义规则
    add_rules("python.library")

    -- 设置编译路径
    set_targetdir("$(projectdir)/tensorrt_yolo/libs")

    -- 添加依赖
    add_deps("deploy")
    add_packages("pybind11")

    -- 添加文件
    add_files("$(projectdir)/source/deploy/pybind/deploy.cpp")

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

    -- 在配置阶段查找 CUDA SDK
    local cuda
    on_load(function (target)
        import("detect.sdks.find_cuda")
        cuda = assert(find_cuda(nil, {verbose = true}), "Cuda SDK not found!")

        -- 设置配置变量
        target:set("configvar", "CUDA", cuda.bindir)
        target:set("configvar", "TensorRT", "$(tensorrt)")

        -- 设置配置目录和配置文件
        target:set("configdir", "$(projectdir)/tensorrt_yolo")
        target:add("configfiles", "$(projectdir)/tensorrt_yolo/c_lib_wrap.py.in")
    end)

    after_build(function (target)
        -- 将 $(projectdir)/lib 文件夹中的所有文件复制到 $(projectdir)/tensorrt_yolo/libs
        os.cp("$(projectdir)/lib/*", "$(projectdir)/tensorrt_yolo/libs/")
    end)
