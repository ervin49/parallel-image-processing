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

cv::Mat multiThreadedCpuOriginalTime;
double originalSingleThreadedCpuTime, originalMultiThreadedCpuTime, originalGpuTime, singleThreadedCpuTime, multiThreadedCpuTime, gpuTime;
const int TOTAL_THREADS = cv::getNumThreads();

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
        cv::Mat multiCpuLoaded = cv::imread(outPath), singleCpuLoaded;
        cv::UMat gpuLoaded;
        multiCpuLoaded.copyTo(singleCpuLoaded);
        multiCpuLoaded.copyTo(gpuLoaded);
        NFD_FreePathU8(outPath);

        if (multiCpuLoaded.empty())
        {
            std::cerr << "Failed to load image\n";
            return;
        }

        auto start = std::chrono::steady_clock::now();
        cv::cvtColor(multiCpuLoaded, multiCpuLoaded, cv::COLOR_BGR2RGB);
        auto end = std::chrono::steady_clock::now();
        originalMultiThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();

        cv::setNumThreads(1);
        start = std::chrono::steady_clock::now();
        cv::cvtColor(singleCpuLoaded, singleCpuLoaded, cv::COLOR_BGR2RGB);
        end = std::chrono::steady_clock::now();
        originalSingleThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();
        cv::setNumThreads(TOTAL_THREADS);

        start = std::chrono::steady_clock::now();
        cv::cvtColor(gpuLoaded, gpuLoaded, cv::COLOR_BGR2RGB);
        end = std::chrono::steady_clock::now();
        originalGpuTime = std::chrono::duration<double, std::milli>(end - start).count();

        multiThreadedCpuTime = originalMultiThreadedCpuTime;
        singleThreadedCpuTime = originalSingleThreadedCpuTime;
        gpuTime = originalGpuTime;

        multiThreadedCpuOriginalTime = multiCpuLoaded;
        img = multiThreadedCpuOriginalTime.clone();

        uploadTexture(img, texture);
    }
    else if (result == NFD_ERROR)
    {
        std::cerr << "Error: " << NFD_GetError() << '\n';
    }
}

int main()
{
    if (!cv::ocl::haveOpenCL())
    {
        std::cerr<<"OpenCL not available";
    }

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
                    img = multiThreadedCpuOriginalTime.clone();
                    singleThreadedCpuTime = originalSingleThreadedCpuTime;
                    multiThreadedCpuTime = originalMultiThreadedCpuTime;
                    gpuTime = originalGpuTime;
                    uploadTexture(img, texture);
                }

                if (ImGui::MenuItem("Grayscale") && img.channels() >= 3)
                {
                    cv::UMat gpuImg;
                    cv::Mat singleCpuImg;

                    img.copyTo(singleCpuImg);
                    img.copyTo(gpuImg);

                    auto start = std::chrono::steady_clock::now();
                    cv::cvtColor(img, img, img.channels() == 3 ? cv::COLOR_RGB2GRAY
                        : cv::COLOR_RGBA2GRAY);
                    auto end = std::chrono::steady_clock::now();
                    multiThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                    cv::setNumThreads(1);
                    start = std::chrono::steady_clock::now();
                    cv::cvtColor(singleCpuImg, singleCpuImg, singleCpuImg.channels() == 3 ? cv::COLOR_RGB2GRAY
                        : cv::COLOR_RGBA2GRAY);
                    end = std::chrono::steady_clock::now();
                    singleThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();
                    cv::setNumThreads(TOTAL_THREADS);

                    start = std::chrono::steady_clock::now();
                    cv::cvtColor(gpuImg, gpuImg, gpuImg.channels() == 3 ? cv::COLOR_RGB2GRAY
                        : cv::COLOR_RGBA2GRAY);
                    end = std::chrono::steady_clock::now();
                    gpuTime = std::chrono::duration<double, std::milli>(end - start).count();

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

            char singleThreadedCpuTimeText[32];
            snprintf(singleThreadedCpuTimeText, sizeof(singleThreadedCpuTimeText), "%.2f ms", singleThreadedCpuTime);

            char multiThreadedCpuTimeText[32];
            snprintf(multiThreadedCpuTimeText, sizeof(multiThreadedCpuTimeText), "%.2f ms", multiThreadedCpuTime);

            char gpuTimeText[32];
            snprintf(gpuTimeText, sizeof(gpuTimeText), "%.2f ms", gpuTime);

            const float columnWidth = ImGui::CalcTextSize("Speedup vs. single-threaded CPU").x;
            const float metricsY = contentTop + contentMax.y - contentMin.y - statsHeight;
            const float gapBetweenColumns = 50.0f;

            const float firstColumnX = 0.0f;
            const float secondColumnX = firstColumnX + ImGui::CalcTextSize("Single-threaded CPU Time").x + gapBetweenColumns;
            const float thirdColumnX = secondColumnX + ImGui::CalcTextSize("Multithreaded CPU Time").x + gapBetweenColumns;
            const float fourthColumnX = thirdColumnX + ImGui::CalcTextSize("GPU Time").x + gapBetweenColumns;
            const float fifthColumnX = fourthColumnX + ImGui::CalcTextSize("Speedup vs. single-threaded CPU").x + gapBetweenColumns;

            ImGui::SetCursorPos(ImVec2(firstColumnX, metricsY));
            ImGui::Text("Single-threaded CPU Time");

            ImGui::SameLine();
            ImGui::SetCursorPos(ImVec2(secondColumnX, metricsY));
            ImGui::Text("Multithreaded CPU Time");

            ImGui::SameLine();
            ImGui::SetCursorPosX(thirdColumnX);
            ImGui::Text("GPU Time");

            ImGui::SameLine();
            ImGui::SetCursorPosX(fourthColumnX);
            ImGui::Text("Speedup vs. single-threaded CPU");

            ImGui::SameLine();
            ImGui::SetCursorPosX(fifthColumnX);
            ImGui::Text("Speedup vs. multithreaded CPU");

            ImGui::SetCursorPos(ImVec2(firstColumnX, metricsY + ImGui::GetTextLineHeightWithSpacing()));
            ImGui::TextUnformatted(singleThreadedCpuTimeText);

            ImGui::SameLine();
            ImGui::SetCursorPosX(secondColumnX);
            ImGui::TextUnformatted(multiThreadedCpuTimeText);

            ImGui::SameLine();
            ImGui::SetCursorPosX(thirdColumnX);
            ImGui::TextUnformatted(gpuTimeText);

            ImGui::SameLine();
            ImGui::SetCursorPosX(fourthColumnX);
            ImGui::Text("%.2fx", singleThreadedCpuTime / gpuTime);

            ImGui::SameLine();
            ImGui::SetCursorPosX(fifthColumnX);
            ImGui::Text("%.2fx", multiThreadedCpuTime / gpuTime);
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
