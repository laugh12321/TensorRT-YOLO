-- 设置项目信息
set_languages("cxx17")
set_allowedplats("linux")

add_requires("opencv", "gstreamer-1.0", {system = true})

-- 添加编译规则
add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})
add_rules("mode.release")

-- 定义选项
option("tensorrt")
    set_showmenu(true)
    set_description("TensorRT Path. Example: /usr/local/tensorrt")
    on_check(function (option)
        if not option:enabled() then 
            raise("TensorRT path is not set. Please specify the TensorRT path.")
        end 
    end)

option("deploy")
    set_showmenu(true)
    set_description("TensorRT-YOLO Project Path.")
    on_check(function (option)
        if not option:enabled() then 
            raise("TensorRT-YOLO project path is not set. Please specify the TensorRT-YOLO Project path.")
        end 
    end)

option("videopipe")
set_showmenu(true)
set_description("VideoPipe Project Path.")
on_check(function (option)
    if not option:enabled() then 
        raise("VideoPipe project path is not set. Please specify the VideoPipe Project path.")
    end 
end)

-- 定义目标
target("PipeDemo")
    set_languages("cxx17")

    -- 设置编译路径
    set_targetdir("$(projectdir)/workspace")

    -- 设置运行路径
    set_rundir("$(projectdir)/workspace")

    -- 添加依赖
    add_packages("opencv", "gstreamer-1.0")

    -- 添加文件
    add_files("src/*.cpp")

    -- 设置目标类型
    set_kind("binary")

    -- 添加 cuda
    add_rules("cuda")
    add_cugencodes("native")
    add_cuflags("-allow-unsupported-compiler")

    -- 添加TensorRT链接目录和链接库
    if has_config("tensorrt") then
        add_includedirs(path.join("$(tensorrt)", "include"))
        add_linkdirs(path.join("$(tensorrt)", "lib"))
        add_links("nvinfer", "nvinfer_plugin", "nvonnxparser")
    end

    -- 添加deploy链接目录和链接库
    if has_config("deploy") then
        add_includedirs(path.join("$(deploy)", "include"))
        add_linkdirs(path.join("$(deploy)", "lib"))
        add_links("deploy")
    end

    -- 添加VideoPipe链接目录和链接库
    if has_config("videopipe") then
        add_includedirs(path.join("$(videopipe)"))
        add_linkdirs(path.join("$(videopipe)", "build/libs"))
        add_links("video_pipe", "tinyexpr")
    end