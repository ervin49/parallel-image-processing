
cmake_minimum_required(VERSION 3.20)
project(parallel_image_processing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
)
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED IMPORTED_TARGET glfw3)

set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/external/imgui")

add_executable(parallel_image_processing
        main.cpp

        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp

        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_compile_definitions(parallel_image_processing PRIVATE
    ASSETS_DIR="${CMAKE_SOURCE_DIR}/external/assets"
)

target_include_directories(parallel_image_processing PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
)

add_subdirectory(external/nfd)

target_link_libraries(parallel_image_processing PRIVATE
        ${OpenCV_LIBS}
        PkgConfig::GLFW
        OpenGL::GL
        nfd
        stdc++
)
