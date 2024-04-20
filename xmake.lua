-- 设置项目信息
set_project("TensorRT-YOLO")
set_version("3.0")
set_xmakever("2.8.0")
set_allowedplats("windows", "linux")

-- 添加编译规则
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")
add_requires("opencv")

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
    set_languages("cxx17")
    add_packages("opencv")
    set_targetdir("$(projectdir)/lib")

    -- 添加库目录
    add_includedirs("$(projectdir)/include", {public = true})

    -- 添加文件
    add_files("$(projectdir)/source/**.cpp", "$(projectdir)/source/**.cu")

    -- 设置目标类型
    set_kind("$(kind)")

    -- 如果目标类型是静态库
    if is_kind("static") then 
        -- 设置 CUDA 开发链接为 true
        set_policy("build.cuda.devlink", true)
    else
        add_defines("ENABLE_DEPLOY_BUILDING_DLL")
    end

    -- 添加 cuda
    add_rules("cuda")
    add_cugencodes("native")

    -- 添加TensorRT链接目录和链接库
    if has_config("tensorrt") then
        add_includedirs(path.join("$(tensorrt)", "include"))
        add_linkdirs(path.join("$(tensorrt)", "lib"))
        add_links("nvinfer", "nvinfer_plugin", "nvparsers", "nvonnxparser")
    end
