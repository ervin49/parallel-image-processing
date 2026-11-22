#include <GLFW/glfw3.h>
#include <opencv2/core/ocl.hpp>
#include <chrono>
#include <clocale>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "external/nfd/src/include/nfd.h"
#include "imgui.h"
#include "imgui_internal.h"

namespace
{
    cv::Mat multiThreadedCpuOriginalImg;
    double originalSingleThreadedCpuTime, originalMultiThreadedCpuTime, originalGpuTime, singleThreadedCpuTime,
           multiThreadedCpuTime, gpuTime;
    std::string currentImagePath;
}

const int TOTAL_THREADS = cv::getNumThreads();

static void uploadTexture(const cv::Mat& m, GLuint& texture)
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
        constexpr GLint swizzle[4] = {GL_RED, GL_RED, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, src.cols, src.rows, 0, GL_RED, GL_UNSIGNED_BYTE, src.data);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, src.cols, src.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, src.data);
    }
}

template <typename Image>
static void applyAdjustments(const Image& source, Image& result, int brightness, int contrast, int saturation,
                             int warmth, bool invertColors)
{
    const double alpha = 1.0 + contrast / 100.0;
    const double beta = brightness + 128.0 * (1.0 - alpha);
    source.convertTo(result, -1, alpha, beta);

    if (result.channels() == 3 && saturation > 0)
    {
        Image hsv;
        cv::cvtColor(result, hsv, cv::COLOR_RGB2HSV);
        std::vector<Image> channels;
        cv::split(hsv, channels);
        channels[1].convertTo(channels[1], -1, 1.0 + saturation / 100.0);
        cv::merge(channels, hsv);
        cv::cvtColor(hsv, result, cv::COLOR_HSV2RGB);
    }

    if (result.channels() == 3 && warmth > 0)
    {
        std::vector<Image> channels;
        cv::split(result, channels);
        channels[0].convertTo(channels[0], -1, 1.0, warmth);
        channels[2].convertTo(channels[2], -1, 1.0, -warmth);
        cv::merge(channels, result);
    }

    if (invertColors)
        cv::bitwise_not(result, result);
}

static bool saveImage(const std::string& path, const cv::Mat& image)
{
    try
    {
        cv::Mat encodedImage;
        if (image.channels() == 3)
            cv::cvtColor(image, encodedImage, cv::COLOR_RGB2BGR);
        else if (image.channels() == 4)
            cv::cvtColor(image, encodedImage, cv::COLOR_RGBA2BGRA);
        else
            encodedImage = image;

        return cv::imwrite(path, encodedImage);
    }
    catch (const cv::Exception& error)
    {
        std::cerr << "Failed to save image: " << error.what() << '\n';
        return false;
    }
}

static void displayImage(cv::Mat& img, cv::Mat& brightnessBaseImg, int& brightnessValue, int& contrastValue,
                         int& saturationValue, int& warmthValue, bool& invertColors, GLuint& texture,
                         std::string& currentImagePath)
{
    nfdu8char_t* selectedPath = nullptr;
    constexpr nfdu8filteritem_t filters[1] = {{.name = "Images", .spec = "png,jpg,jpeg,bmp,tiff"}};
    nfdopendialogu8args_t args = {.filterList = nullptr};
    args.filterList = filters;
    args.filterCount = 1;

    const nfdresult_t result = NFD_OpenDialogU8_With(&selectedPath, &args);
    if (result == NFD_OKAY)
    {
        cv::Mat multiCpuLoaded = cv::imread(selectedPath), singleCpuLoaded;
        cv::UMat gpuLoaded;
        multiCpuLoaded.copyTo(singleCpuLoaded);
        multiCpuLoaded.copyTo(gpuLoaded);
        if (multiCpuLoaded.empty())
        {
            NFD_FreePathU8(selectedPath);
            std::cerr << "Failed to load image\n";
            return;
        }
        currentImagePath = selectedPath;
        NFD_FreePathU8(selectedPath);

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

        multiThreadedCpuOriginalImg = multiCpuLoaded;
        img = multiThreadedCpuOriginalImg.clone();
        brightnessBaseImg = img.clone();
        brightnessValue = 0;
        contrastValue = 0;
        saturationValue = 0;
        warmthValue = 0;
        invertColors = false;

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
        std::cerr << "OpenCL not available";
    }

    NFD_Init();
    // Native dialog initialization may adopt a locale that formats OpenCL
    // kernel constants with commas, which makes generated kernels invalid.
    std::setlocale(LC_NUMERIC, "C");

    if (!glfwInit())
        return 1;

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

    GLuint texture = 0, sidebarTexture = 0;
    int brightnessValue = 0;
    int contrastValue = 0;
    int saturationValue = 0;
    int warmthValue = 0;
    bool invertColors = false;
    const char* sidebarIconPath = ASSETS_DIR "/hide-sidebar-icon.png";
    cv::Mat sidebarIcon = std::filesystem::exists(sidebarIconPath)
                              ? cv::imread(sidebarIconPath, cv::IMREAD_UNCHANGED)
                              : cv::Mat{};

    cv::Mat img;
    cv::Mat brightnessBaseImg;
    if (!sidebarIcon.empty())
    {
        if (sidebarIcon.channels() == 4) cv::cvtColor(sidebarIcon, sidebarIcon, cv::COLOR_BGRA2RGBA);
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
                    displayImage(img, brightnessBaseImg, brightnessValue, contrastValue, saturationValue,
                                 warmthValue, invertColors, texture, currentImagePath);
                }
                ImGui::BeginDisabled(img.empty());
                if (ImGui::MenuItem("Save", "Ctrl+S"))
                {
                    if (currentImagePath.empty())
                    {
                        std::cerr << "No file path is set; use Save As first.\n";
                    }
                    else if (!saveImage(currentImagePath, img))
                        std::cerr << "Failed to save image.\n";
                }
                ImGui::EndDisabled();
                ImGui::BeginDisabled(img.empty());
                if (ImGui::MenuItem("Save as..."))
                {
                    constexpr nfdu8filteritem_t filters[1] = {{.name = "Images", .spec = "png,jpg,jpeg,bmp,tiff"}};
                    nfdsavedialogu8args_t args = {.filterList = nullptr};
                    args.filterList = filters;
                    args.filterCount = 1;

                    nfdu8char_t* selectedPath = nullptr;
                    const nfdresult_t result = NFD_SaveDialogU8_With(&selectedPath, &args);
                    if (result == NFD_OKAY)
                    {
                        const std::string savePath = selectedPath;
                        NFD_FreePathU8(selectedPath);
                        if (saveImage(savePath, img))
                            currentImagePath = savePath;
                        else
                            std::cerr << "Failed to save image.\n";
                    }
                    else if (result == NFD_ERROR)
                    {
                        std::cerr << "Save dialog error: " << NFD_GetError() << '\n';
                    }
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
                displayImage(img, brightnessBaseImg, brightnessValue, contrastValue, saturationValue, warmthValue,
                             invertColors, texture, currentImagePath);
            }

            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(3);
        }

        if (!img.empty())
        {
            cv::UMat gpuImg;
            cv::Mat singleCpuImg;

            img.copyTo(singleCpuImg);
            img.copyTo(gpuImg);
            float sidebarWidth = 0.0f;
            if (isSidebarOpen)
            {
                sidebarWidth = std::clamp(ImGui::GetWindowSize().x * 0.2f, 200.0f, 250.0f);
            }

            // sidebar
            if (isSidebarOpen)
            {
                const ImVec2 mainWindowPos = ImGui::GetWindowPos();
                const ImVec2 mainWindowSize = ImGui::GetWindowSize();
                ImGui::SetNextWindowSize(ImVec2(sidebarWidth, mainWindowSize.y));
                ImGui::SetNextWindowPos(ImVec2(mainWindowPos.x + mainWindowSize.x - sidebarWidth,
                                               mainWindowPos.y));
                ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.145f, 0.145f, 0.149f, 1.0f));
                ImGui::Begin("Sidebar", &isSidebarOpen, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
                ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(0.35f, 0.35f, 0.35f, 1));
                ImGui::PushStyleColor(ImGuiCol_TabSelected, ImVec4(0.45f, 0.45f, 0.45f, 1));
                ImGui::PushStyleColor(ImGuiCol_TabSelectedOverline, ImVec4(0.45f, 0.45f, 0.45f, 1));
                ImGui::PushStyleColor(ImGuiCol_TabDimmed, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_TabDimmedSelected, ImVec4(0.45f, 0.45f, 0.45f, 1));
                ImGui::PushStyleColor(ImGuiCol_TabDimmedSelectedOverline, ImVec4(0.45f, 0.45f, 0.45f, 1));
                if (ImGui::BeginTabBar("tab_bar"))
                {
                    if (ImGui::BeginTabItem("Filters"))
                    {
                        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.35f, 0.35f, 0.35f, 1));
                        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.45f, 0.45f, 0.45f, 1));
                        if (ImGui::MenuItem("Original"))
                        {
                            img = multiThreadedCpuOriginalImg.clone();
                            singleCpuImg = multiThreadedCpuOriginalImg.clone();
                            multiThreadedCpuOriginalImg.copyTo(gpuImg);
                            singleThreadedCpuTime = originalSingleThreadedCpuTime;
                            multiThreadedCpuTime = originalMultiThreadedCpuTime;
                            gpuTime = originalGpuTime;
                            brightnessValue = 0;
                            contrastValue = 0;
                            saturationValue = 0;
                            warmthValue = 0;
                            invertColors = false;
                            brightnessBaseImg = img.clone();
                            uploadTexture(img, texture);
                        }

                        if (ImGui::MenuItem("Grayscale") && img.channels() >= 3)
                        {
                            auto start = std::chrono::steady_clock::now();
                            cv::cvtColor(img, img, img.channels() == 3
                                                       ? cv::COLOR_RGB2GRAY
                                                       : cv::COLOR_RGBA2GRAY);
                            auto end = std::chrono::steady_clock::now();
                            multiThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            cv::setNumThreads(1);
                            start = std::chrono::steady_clock::now();
                            cv::cvtColor(singleCpuImg, singleCpuImg, singleCpuImg.channels() == 3
                                                                         ? cv::COLOR_RGB2GRAY
                                                                         : cv::COLOR_RGBA2GRAY);
                            end = std::chrono::steady_clock::now();
                            singleThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();
                            cv::setNumThreads(TOTAL_THREADS);

                            start = std::chrono::steady_clock::now();
                            cv::cvtColor(gpuImg, gpuImg, gpuImg.channels() == 3
                                                             ? cv::COLOR_RGB2GRAY
                                                             : cv::COLOR_RGBA2GRAY);
                            cv::ocl::finish();
                            end = std::chrono::steady_clock::now();
                            gpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            uploadTexture(img, texture);
                            brightnessBaseImg = img.clone();
                            brightnessValue = 0;
                            contrastValue = 0;
                            saturationValue = 0;
                            warmthValue = 0;
                            invertColors = false;
                        }
                        if (ImGui::MenuItem("Gaussian Blur"))
                        {
                            auto start = std::chrono::steady_clock::now();
                            cv::GaussianBlur(img, img, cv::Size(9, 9), 0);
                            auto end = std::chrono::steady_clock::now();
                            multiThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            cv::setNumThreads(1);
                            start = std::chrono::steady_clock::now();
                            cv::GaussianBlur(singleCpuImg, singleCpuImg, cv::Size(9, 9), 0);
                            end = std::chrono::steady_clock::now();
                            singleThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();
                            cv::setNumThreads(TOTAL_THREADS);

                            start = std::chrono::steady_clock::now();
                            cv::GaussianBlur(gpuImg, gpuImg, cv::Size(9, 9), 0);
                            cv::ocl::finish();
                            end = std::chrono::steady_clock::now();
                            gpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            uploadTexture(img, texture);
                            brightnessBaseImg = img.clone();
                            brightnessValue = 0;
                            contrastValue = 0;
                            saturationValue = 0;
                            warmthValue = 0;
                            invertColors = false;
                        }
                        if (ImGui::MenuItem("Median Blur"))
                        {
                            auto start = std::chrono::steady_clock::now();
                            cv::medianBlur(img, img, 5);
                            auto end = std::chrono::steady_clock::now();
                            multiThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            cv::setNumThreads(1);
                            start = std::chrono::steady_clock::now();
                            cv::medianBlur(singleCpuImg, singleCpuImg, 5);
                            end = std::chrono::steady_clock::now();
                            singleThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();
                            cv::setNumThreads(TOTAL_THREADS);

                            start = std::chrono::steady_clock::now();
                            cv::medianBlur(gpuImg, gpuImg, 5);
                            cv::ocl::finish();
                            end = std::chrono::steady_clock::now();
                            gpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            uploadTexture(img, texture);
                            brightnessBaseImg = img.clone();
                            brightnessValue = 0;
                            contrastValue = 0;
                            saturationValue = 0;
                            warmthValue = 0;
                            invertColors = false;
                        }

                        if (ImGui::MenuItem("Sharpening"))
                        {
                            cv::Matx33f kernel = {
                                0.0f, -1.0f, 0.0f,
                                -1.0f, 5.0f, -1.0f,
                                0.0f, -1.0f, 0.0f
                            };
                            auto start = std::chrono::steady_clock::now();
                            cv::filter2D(img, img, -1, kernel);
                            auto end = std::chrono::steady_clock::now();
                            multiThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            cv::setNumThreads(1);
                            start = std::chrono::steady_clock::now();
                            cv::filter2D(singleCpuImg, singleCpuImg, -1, kernel);
                            end = std::chrono::steady_clock::now();
                            singleThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();
                            cv::setNumThreads(TOTAL_THREADS);

                            start = std::chrono::steady_clock::now();
                            cv::UMat gpuSharpened;
                            cv::filter2D(gpuImg, gpuSharpened, -1, kernel);
                            cv::ocl::finish();
                            end = std::chrono::steady_clock::now();
                            gpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            uploadTexture(img, texture);
                            brightnessBaseImg = img.clone();
                            brightnessValue = 0;
                            contrastValue = 0;
                            saturationValue = 0;
                            warmthValue = 0;
                            invertColors = false;
                        }
                        ImGui::PopStyleColor(2);
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Adjustments"))
                    {
                        ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(0.35f, 0.35f, 0.35f, 1.0f));
                        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(0.45f, 0.45f, 0.45f, 1.0f));
                        ImGui::PushItemWidth(-1.0f);
                        ImGui::TextUnformatted("Brightness");
                        bool adjustmentsChanged = ImGui::SliderInt("##Brightness", &brightnessValue, 0, 100);
                        ImGui::Spacing();
                        ImGui::TextUnformatted("Contrast");
                        adjustmentsChanged |= ImGui::SliderInt("##Contrast", &contrastValue, 0, 100);
                        ImGui::Spacing();
                        ImGui::TextUnformatted("Saturation");
                        adjustmentsChanged |= ImGui::SliderInt("##Saturation", &saturationValue, 0, 100);
                        ImGui::Spacing();
                        ImGui::TextUnformatted("Warmth");
                        adjustmentsChanged |= ImGui::SliderInt("##Warmth", &warmthValue, 0, 100);
                        ImGui::Spacing();
                        adjustmentsChanged |= ImGui::Checkbox("Invert Colors", &invertColors);
                        ImGui::PopItemWidth();
                        ImGui::PopStyleColor(2);
                        if (adjustmentsChanged)
                        {
                            auto start = std::chrono::steady_clock::now();
                            applyAdjustments(brightnessBaseImg, img, brightnessValue, contrastValue, saturationValue,
                                             warmthValue, invertColors);
                            auto end = std::chrono::steady_clock::now();
                            multiThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            cv::setNumThreads(1);
                            start = std::chrono::steady_clock::now();
                            applyAdjustments(brightnessBaseImg, singleCpuImg, brightnessValue, contrastValue,
                                             saturationValue, warmthValue, invertColors);
                            end = std::chrono::steady_clock::now();
                            singleThreadedCpuTime = std::chrono::duration<double, std::milli>(end - start).count();
                            cv::setNumThreads(TOTAL_THREADS);

                            cv::UMat gpuBaseImg;
                            brightnessBaseImg.copyTo(gpuBaseImg);
                            start = std::chrono::steady_clock::now();
                            applyAdjustments(gpuBaseImg, gpuImg, brightnessValue, contrastValue, saturationValue,
                                             warmthValue, invertColors);
                            cv::ocl::finish();
                            end = std::chrono::steady_clock::now();
                            gpuTime = std::chrono::duration<double, std::milli>(end - start).count();

                            uploadTexture(img, texture);
                        }
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
                ImGui::PopStyleColor(7);
                ImGui::End();
                ImGui::PopStyleColor(1);
            }

            // image view
            constexpr float padding = 20.0f;
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

            // center image in the area above the metrics.
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
            snprintf(singleThreadedCpuTimeText, sizeof(singleThreadedCpuTimeText), "%.4f ms", singleThreadedCpuTime);

            char multiThreadedCpuTimeText[32];
            snprintf(multiThreadedCpuTimeText, sizeof(multiThreadedCpuTimeText), "%.4f ms", multiThreadedCpuTime);

            char gpuTimeText[32];
            snprintf(gpuTimeText, sizeof(gpuTimeText), "%.4f ms", gpuTime);

            const float metricsY = contentTop + contentMax.y - contentMin.y - statsHeight;
            const float metricsWidth = std::max(0.0f, available.x);
            char cpu1Metric[48];
            snprintf(cpu1Metric, sizeof(cpu1Metric), "CPU 1T %s", singleThreadedCpuTimeText);
            char cpuMtMetric[48];
            snprintf(cpuMtMetric, sizeof(cpuMtMetric), "CPU MT %s", multiThreadedCpuTimeText);
            char gpuMetric[48];
            snprintf(gpuMetric, sizeof(gpuMetric), "GPU %s", gpuTimeText);
            char speedupText[64];
            snprintf(speedupText, sizeof(speedupText), "Speedup (1T / MT) %.2fx / %.2fx",
                     singleThreadedCpuTime / gpuTime, multiThreadedCpuTime / gpuTime);

            const float textWidths[] = {
                ImGui::CalcTextSize(cpu1Metric).x,
                ImGui::CalcTextSize(cpuMtMetric).x,
                ImGui::CalcTextSize(gpuMetric).x,
                ImGui::CalcTextSize(speedupText).x,
            };
            const float totalTextWidth = textWidths[0] + textWidths[1] + textWidths[2] + textWidths[3];
            const float separatorWidth = ImGui::CalcTextSize("|").x;
            const float gapWidth = std::max(0.0f, (metricsWidth - totalTextWidth - separatorWidth * 3.0f) / 6.0f);
            const float segmentWidths[] = {textWidths[0], textWidths[1], textWidths[2], textWidths[3]};
            const char* segmentTexts[] = {cpu1Metric, cpuMtMetric, gpuMetric, speedupText};

            float metricsX = contentLeft + padding * 0.5f;
            ImGui::SetCursorPos(ImVec2(metricsX, metricsY));
            for (int i = 0; i < 4; ++i)
            {
                ImGui::TextUnformatted(segmentTexts[i]);
                metricsX += segmentWidths[i];
                if (i < 3)
                {
                    metricsX += gapWidth;
                    ImGui::SetCursorPos(ImVec2(metricsX, metricsY));
                    ImGui::TextUnformatted("|");
                    metricsX += separatorWidth + gapWidth;
                    ImGui::SetCursorPos(ImVec2(metricsX, metricsY));
                }
            }
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

    // clean up
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
