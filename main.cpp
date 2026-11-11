#include <GLFW/glfw3.h>
#include <opencv2/core/ocl.hpp>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "external/nfd/src/include/nfd.h"
#include "imgui.h"
#include "imgui_internal.h"

cv::Mat cpuOriginalImage;
double originalCpuTime, originalGpuTime, cpuTime, gpuTime;

void uploadTexture(const cv::Mat& m, GLuint& texture)
{
    const cv::Mat src = m.isContinuous() ? m : m.clone();

    if (texture != 0)
        glDeleteTextures(1, &texture);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (src.channels() == 1)
    {
        const GLint swizzle[4] = {GL_RED, GL_RED, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, src.cols, src.rows, 0, GL_RED, GL_UNSIGNED_BYTE, src.data);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, src.cols, src.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, src.data);
    }
}
void displayImage(cv::Mat& img, GLuint& texture)
{
    nfdu8char_t* outPath = nullptr;
    constexpr nfdu8filteritem_t filters[1] = {{.name = "Images", .spec = "png,jpg,jpeg,bmp,tiff"}};
    nfdopendialogu8args_t args = {.filterList = nullptr};
    args.filterList = filters;
    args.filterCount = 1;

    const nfdresult_t result = NFD_OpenDialogU8_With(&outPath, &args);
    if (result == NFD_OKAY)
    {
        cv::Mat cpuLoaded = cv::imread(outPath);
        cv::UMat gpuLoaded;
        cpuLoaded.copyTo(gpuLoaded);
        NFD_FreePathU8(outPath);

        if (cpuLoaded.empty())
        {
            std::cerr << "Failed to load image\n";
            return;
        }

        auto start = std::chrono::steady_clock::now();
        cv::cvtColor(cpuLoaded, cpuLoaded, cv::COLOR_BGR2RGB);
        auto end = std::chrono::steady_clock::now();
        originalCpuTime = std::chrono::duration<double, std::milli>(end - start).count();
        start = std::chrono::steady_clock::now();
        cv::cvtColor(gpuLoaded, gpuLoaded, cv::COLOR_BGR2RGB);
        end = std::chrono::steady_clock::now();
        originalGpuTime = std::chrono::duration<double, std::milli>(end - start).count();
        cpuTime = originalCpuTime;
        gpuTime = originalGpuTime;
        cpuOriginalImage = cpuLoaded;
        img = cpuOriginalImage.clone();
        uploadTexture(img, texture);
    }
    else if (result == NFD_ERROR)
    {
        std::cerr << "Error: " << NFD_GetError() << '\n';
    }
}

int main()
{
    NFD_Init();
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);

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

    bool benchmarkMode = true, cpuCheck = false, gpuCheck = false;
    const ImGuiIO& io = ImGui::GetIO();
    ImFont* font = io.Fonts->AddFontFromFileTTF(ASSETS_DIR "/fonts/ARIAL.TTF");
    cv::Mat img;
    GLuint texture = 0;
    GLuint sidebarTexture = 0;
    cv::Mat sidebarIcon = cv::imread(ASSETS_DIR "/hide-sidebar-icon.png", cv::IMREAD_UNCHANGED);
    if (!sidebarIcon.empty())
    {
        if (sidebarIcon.channels() == 4)      cv::cvtColor(sidebarIcon, sidebarIcon, cv::COLOR_BGRA2RGBA);
        else if (sidebarIcon.channels() == 3) cv::cvtColor(sidebarIcon, sidebarIcon, cv::COLOR_BGR2RGBA);
        else if (sidebarIcon.channels() == 1) cv::cvtColor(sidebarIcon, sidebarIcon, cv::COLOR_GRAY2RGBA);

        glGenTextures(1, &sidebarTexture);
        glBindTexture(GL_TEXTURE_2D, sidebarTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, sidebarIcon.cols, sidebarIcon.rows, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, sidebarIcon.data);
    }

    bool isSidebarOpen = true;
    float mainMenuBarHeight = 0;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.11f, 0.11f, 0.11f, 1.0f));
        ImGui::NewFrame();
        ImGui::PushFont(font);

        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y));
        bool open = true;
        if (ImGui::BeginMainMenuBar())
        {
            mainMenuBarHeight = ImGui::GetWindowSize().y;
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.35f, 0.35f, 0.35f, 1));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.45f, 0.45f, 0.45f, 1));
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Open", "Ctrl+O"))
                {
                    displayImage(img, texture);
                }
                ImGui::BeginDisabled(img.empty());
                if (ImGui::MenuItem("Save", "Ctrl+S"))
                {
                }
                ImGui::EndDisabled();
                ImGui::BeginDisabled(img.empty());
                if (ImGui::MenuItem("Export"))
                {
                }
                ImGui::EndDisabled();
                if (ImGui::MenuItem("Exit"))
                {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Edit"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Image"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Processing"))
            {
                if (ImGui::MenuItem("Benchmark Mode", nullptr, &benchmarkMode))
                {
                    benchmarkMode = true;
                    cpuCheck = false;
                    gpuCheck = false;
                }
                if (ImGui::MenuItem("CPU Mode", nullptr, &cpuCheck))
                {
                    benchmarkMode = false;
                    cpuCheck = true;
                    gpuCheck = false;
                }
                if (ImGui::MenuItem("GPU Mode", nullptr, &gpuCheck))
                {
                    benchmarkMode = false;
                    cpuCheck = false;
                    gpuCheck = true;
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help"))
            {
                if (ImGui::MenuItem("About"))
                {
                }
                ImGui::EndMenu();
            }
            ImGui::PopStyleColor(2);

            if (!img.empty())
            {
                ImGui::SetCursorPosX(ImGui::GetCurrentWindow()->Size.x - 30);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.35f, 0.35f, 1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.45f, 0.45f, 1));
                if (ImGui::ImageButton("sidebar_toggle",
                                       static_cast<ImTextureID>(static_cast<intptr_t>(sidebarTexture)), ImVec2(20, 20)))
                {
                    isSidebarOpen = !isSidebarOpen;
                }
                ImGui::PopStyleColor(3);
            }
            ImGui::EndMainMenuBar();
        }
        ImGui::SetNextWindowPos(ImVec2(0.0f, mainMenuBarHeight));
        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y - mainMenuBarHeight));
        ImGui::Begin("Parallel Image Processing", &open,
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar |
                         ImGuiWindowFlags_NoBringToFrontOnFocus);

        if (img.empty())
        {
            float fontHeight = ImGui::GetTextLineHeight();
            float fontSize = ImGui::CalcTextSize("No image loaded").x;
            ImGui::SetCursorPos(
                ImVec2(ImGui::GetWindowSize().x / 2 - fontSize / 2, ImGui::GetWindowSize().y / 2 - fontHeight * 1.5f));
            ImGui::Text("No image loaded");
            fontSize = ImGui::CalcTextSize("Open an image to start processing").x;
            ImGui::SetCursorPos(ImVec2(ImGui::GetWindowSize().x / 2 - fontSize / 2, ImGui::GetWindowSize().y / 2));
            ImGui::Text("Open an image to start processing.");
            fontSize = ImGui::CalcTextSize("Open Image").x;
            ImGui::SetCursorPos(ImVec2(ImGui::GetWindowSize().x / 2 - fontSize / 2 - 8.0f,
                                       ImGui::GetWindowSize().y / 2 + fontHeight * 1.5f));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 14.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 6.0f));
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.35f, 0.35f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.45f, 0.45f, 1.0f));
            if (ImGui::Button("Open Image"))
            {
                displayImage(img, texture);
            }
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(3);
        }

        if (!img.empty())
        {
            // sidebar
            if (isSidebarOpen)
            {
                float sidebarWidth = ImGui::GetContentRegionAvail().x * 0.2f;
                sidebarWidth = std::clamp(sidebarWidth, 200.0f, 250.0f);
                const ImVec2 mainWindowPos = ImGui::GetWindowPos();
                const ImVec2 mainWindowSize = ImGui::GetWindowSize();
                ImGui::SetNextWindowSize(ImVec2(sidebarWidth, mainWindowSize.y));
                ImGui::SetNextWindowPos(ImVec2(mainWindowPos.x + mainWindowSize.x - sidebarWidth,
                                               mainWindowPos.y));
                ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.145f, 0.145f, 0.149f, 1.0f));
                ImGui::Begin("Sidebar", &isSidebarOpen, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
                ImGui::Text("Adjustments Filters");
                if (ImGui::MenuItem("Original"))
                {
                    img = cpuOriginalImage.clone();
                    cpuTime = originalCpuTime;
                    uploadTexture(img, texture);
                }

                if (ImGui::MenuItem("Grayscale") && img.channels() >= 3)
                {
                    const auto start = std::chrono::steady_clock::now();
                    cv::cvtColor(img, img, img.channels() == 3 ? cv::COLOR_RGB2GRAY
                        : cv::COLOR_RGBA2GRAY);
                    const auto end = std::chrono::steady_clock::now();
                    cpuTime = std::chrono::duration<double, std::milli>(end - start).count();
                    uploadTexture(img, texture);
                }
                ImGui::End();
                ImGui::PopStyleColor(1);
            }

            // image view
            float sidebarWidth = 0.0f;

            if (isSidebarOpen)
            {
                sidebarWidth = std::clamp(
                    ImGui::GetWindowSize().x * 0.2f,
                    200.0f,
                    250.0f
                );
            }

            const float padding = 20.0f;
            const ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
            const ImVec2 contentMax = ImGui::GetWindowContentRegionMax();
            const float statsHeight = ImGui::GetTextLineHeightWithSpacing() * 2.0f;
            const ImVec2 available(contentMax.x - contentMin.x - sidebarWidth - padding,
                                   contentMax.y - contentMin.y - statsHeight - padding);

            const float aspectRatio = static_cast<float>(img.cols) / static_cast<float>(img.rows);

            float width = available.x;
            float height = width / aspectRatio;

            if (height > available.y)
            {
                height = available.y;
                width = height * aspectRatio;
            }

            // Center image in the area above the metrics.
            const float offsetX = (available.x - width) * 0.5f;
            const float offsetY = (available.y - height) * 0.5f;

            const float contentLeft = contentMin.x;
            const float contentTop = contentMin.y;
            ImGui::SetCursorPos(ImVec2(contentLeft + padding * 0.5f + offsetX,
                                       contentTop + padding * 0.5f + offsetY));

            ImGui::Image(
                static_cast<ImTextureID>(static_cast<intptr_t>(texture)),
                ImVec2(width, height)
            );

            char cpuTimeText[32];
            snprintf(cpuTimeText, sizeof(cpuTimeText), "%.2f ms", cpuTime);

            char gpuTimeText[32];
            snprintf(gpuTimeText, sizeof(gpuTimeText), "%.2f ms", gpuTime);

            const float columnWidth = std::max(
                ImGui::CalcTextSize("CPU Time").x,
                ImGui::CalcTextSize(cpuTimeText).x);
            const float metricsY = contentTop + contentMax.y - contentMin.y - statsHeight;
            const float metricsWidth = available.x;
            const float groupWidth = columnWidth * 2.0f + 200.0f;
            const float firstColumnX = contentLeft + (metricsWidth - groupWidth) * 0.5f;
            const float secondColumnX = firstColumnX + columnWidth + 100.0f;
            const float thirdColumnX = secondColumnX + columnWidth + 100.0f;

            ImGui::SetCursorPos(ImVec2(firstColumnX, metricsY));
            ImGui::Text("CPU Time");

            ImGui::SameLine();
            ImGui::SetCursorPosX(secondColumnX);
            ImGui::Text("GPU Time");

            ImGui::SameLine();
            ImGui::SetCursorPosX(thirdColumnX);
            ImGui::Text("Speedup");

            ImGui::SetCursorPos(ImVec2(firstColumnX, metricsY + ImGui::GetTextLineHeightWithSpacing()));
            ImGui::TextUnformatted(cpuTimeText);

            ImGui::SameLine();
            ImGui::SetCursorPosX(secondColumnX);
            ImGui::TextUnformatted(gpuTimeText);

            ImGui::SameLine();
            ImGui::SetCursorPosX(thirdColumnX);
            ImGui::Text("%.2fx", cpuTime / gpuTime);
        }

        ImGui::PopFont();
        ImGui::End();
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
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (texture != 0)
    {
        glDeleteTextures(1, &texture);
    }
    if (sidebarTexture)
    {
        glDeleteTextures(1, &sidebarTexture);
    }
    glfwDestroyWindow(window);
    glfwTerminate();

    NFD_Quit();
    return 0;
}
