#include <GLFW/glfw3.h>
#include <opencv2/core/ocl.hpp>
#include <clocale>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "external/nfd/src/include/nfd.h"
#include "imgui.h"
#include "src/Benchmark.h"
#include "src/EditableImage.h"
#include "src/EditorUiState.h"
#include "src/ImageProcessor.h"
#include "src/ImageEditorUI.h"

int main()
{
    NFD_Init();
    // Native dialog initialization may adopt a locale that formats OpenCL
    // kernel constants with commas, which makes generated kernels invalid.
    std::setlocale(LC_NUMERIC, "C");

    if (!cv::ocl::haveOpenCL() || !cv::ocl::useOpenCL())
    {
        std::cerr << "OpenCL not available";
    }

    if (!glfwInit())
        return 1;

    glfwWindowHintString(GLFW_WAYLAND_APP_ID, "parallel-image-processing");
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Parallel Image Processing", nullptr, nullptr);

    if (!window)
    {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    const ImGuiIO& io = ImGui::GetIO();
    ImFont* font = io.Fonts->AddFontFromFileTTF(ASSETS_DIR "/fonts/ARIAL.TTF");

    GLuint texture = 0;

    EditableImage editableImage;
    EditorUiState uiState;
    ImageProcessor imageProcessor;
    Benchmark benchmark;
    {
        ImageEditorUI editorUi(editableImage, uiState, imageProcessor, benchmark, texture);
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.11f, 0.11f, 0.11f, 1.0f));
            ImGui::NewFrame();
            editorUi.draw(io.DisplaySize, font, window);
            ImGui::Render();
            ImGui::PopStyleColor(1);

            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            glViewport(0, 0, width, height);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }
    }

    // clean up
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (texture != 0)
    {
        glDeleteTextures(1, &texture);
    }
    glfwDestroyWindow(window);
    glfwTerminate();

    NFD_Quit();
    return 0;
}
