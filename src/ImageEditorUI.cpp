#include "ImageEditorUI.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Benchmark.h"
#include "EditableImage.h"
#include "EditorUiState.h"
#include "ImageDisplayer.h"
#include "ImageProcessor.h"
#include "nfd.h"

ImageEditorUI::ImageEditorUI(EditableImage& image, EditorUiState& state, ImageProcessor& processor,
                             Benchmark& benchmark, GLuint& texture)
    : image(image), state(state), processor(processor), benchmark(benchmark), texture(texture)
{
    loadSidebarIcon();
}

ImageEditorUI::~ImageEditorUI()
{
    if (sidebarTexture != 0)
        glDeleteTextures(1, &sidebarTexture);
}

void ImageEditorUI::loadSidebarIcon()
{
    const std::string iconPath = ASSETS_DIR "/hide-sidebar-icon.png";
    if (!std::filesystem::exists(iconPath))
        return;

    cv::Mat icon = cv::imread(iconPath, cv::IMREAD_UNCHANGED);
    if (icon.empty())
        return;

    if (icon.channels() == 4)
        cv::cvtColor(icon, icon, cv::COLOR_BGRA2RGBA);
    else if (icon.channels() == 3)
        cv::cvtColor(icon, icon, cv::COLOR_BGR2RGBA);
    else if (icon.channels() == 1)
        cv::cvtColor(icon, icon, cv::COLOR_GRAY2RGBA);
    else
        return;

    ImageDisplayer::uploadTexture(icon, sidebarTexture);
}

void ImageEditorUI::openImage()
{
    nfdu8char_t* selectedPath = nullptr;
    constexpr nfdu8filteritem_t filters[1] = {{.name = "Images", .spec = "png,jpg,jpeg,bmp,tiff"}};
    nfdopendialogu8args_t args = {.filterList = filters, .filterCount = 1};
    const nfdresult_t result = NFD_OpenDialogU8_With(&selectedPath, &args);
    if (result != NFD_OKAY)
    {
        if (result == NFD_ERROR)
            std::cerr << "Open dialog error: " << NFD_GetError() << '\n';
        return;
    }

    const std::string path = selectedPath;
    NFD_FreePathU8(selectedPath);
    if (!image.load(path, benchmark))
    {
        std::cerr << "Failed to load image: " << path << '\n';
        return;
    }

    state.brightness = 0;
    state.contrast = 0;
    state.saturation = 0;
    state.warmth = 0;
    state.invertColors = false;
    ImageDisplayer::uploadTexture(image.getImage(), texture);
}

void ImageEditorUI::draw(const ImVec2& displaySize, ImFont* font, GLFWwindow* window)
{
    auto measureAndSetImage = [&](const cv::Mat& source, auto operation)
    {
        cv::Mat processedImage;
        benchmark.run(source, operation, processedImage);
        image.setImage(processedImage);
    };

    ImGui::PushFont(font);
    ImGui::SetNextWindowSize(ImVec2(displaySize.x, displaySize.y));
    bool open = true;
    if (ImGui::BeginMainMenuBar())
    {
        state.mainMenuBarHeight = ImGui::GetWindowSize().y;
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.35f, 0.35f, 0.35f, 1));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.45f, 0.45f, 0.45f, 1));
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open", "Ctrl+O"))
            {
                openImage();
            }
            ImGui::BeginDisabled(image.getImage().empty());
            if (ImGui::MenuItem("Save", "Ctrl+S"))
            {
                if (image.getPath().empty())
                {
                    std::cerr << "No file path is set; use Save As first.\n";
                }
                else if (!image.save())
                    std::cerr << "Failed to save image.\n";
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(image.getImage().empty());
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
                    if (!image.saveAs(savePath))
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

        if (!image.getImage().empty())
        {
            ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 30);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.35f, 0.35f, 1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.45f, 0.45f, 1));
            if (ImGui::ImageButton("sidebar_toggle",
                                   static_cast<ImTextureID>(static_cast<intptr_t>(sidebarTexture)), ImVec2(20, 20)))
            {
                state.sidebarOpen = !state.sidebarOpen;
            }
            ImGui::PopStyleColor(3);
        }
        ImGui::EndMainMenuBar();
    }

    ImGui::SetNextWindowPos(ImVec2(0.0f, state.mainMenuBarHeight));
    ImGui::SetNextWindowSize(ImVec2(displaySize.x, displaySize.y - state.mainMenuBarHeight));
    ImGui::Begin("Parallel Image Processing", &open,
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_NoBringToFrontOnFocus);

    if (image.getImage().empty())
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
            openImage();
        }

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(3);
    }

    if (!image.getImage().empty())
    {
        float sidebarWidth = 0.0f;
        if (state.sidebarOpen)
        {
            sidebarWidth = std::clamp(ImGui::GetWindowSize().x * 0.2f, 200.0f, 250.0f);
        }

        // sidebar
        if (state.sidebarOpen)
        {
            const ImVec2 mainWindowPos = ImGui::GetWindowPos();
            const ImVec2 mainWindowSize = ImGui::GetWindowSize();
            ImGui::SetNextWindowSize(ImVec2(sidebarWidth, mainWindowSize.y));
            ImGui::SetNextWindowPos(ImVec2(mainWindowPos.x + mainWindowSize.x - sidebarWidth,
                                           mainWindowPos.y));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.145f, 0.145f, 0.149f, 1.0f));
            ImGui::Begin("Sidebar", &state.sidebarOpen, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
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
                        image.restoreOriginal();
                        benchmark.setLastResult(benchmark.getOriginalResult());
                        state.brightness = 0;
                        state.contrast = 0;
                        state.saturation = 0;
                        state.warmth = 0;
                        state.invertColors = false;
                        ImageDisplayer::uploadTexture(image.getImage(), texture);
                    }

                    if (ImGui::MenuItem("Grayscale") && image.getImage().channels() >= 3)
                    {
                        measureAndSetImage(image.getImage(), [this](const auto& source, auto& destination)
                        {
                            processor.toGrayscale(source, destination);
                        });
                        ImageDisplayer::uploadTexture(image.getImage(), texture);
                        image.setProcessingBaseImage(image.getImage());
                        state.brightness = 0;
                        state.contrast = 0;
                        state.saturation = 0;
                        state.warmth = 0;
                        state.invertColors = false;
                    }
                    if (ImGui::MenuItem("Gaussian Blur"))
                    {
                        measureAndSetImage(image.getImage(), [this](const auto& source, auto& destination)
                        {
                            processor.gaussianBlur(source, destination);
                        });
                        ImageDisplayer::uploadTexture(image.getImage(), texture);
                        image.setProcessingBaseImage(image.getImage());
                        state.brightness = 0;
                        state.contrast = 0;
                        state.saturation = 0;
                        state.warmth = 0;
                        state.invertColors = false;
                    }
                    if (ImGui::MenuItem("Median Blur"))
                    {
                        measureAndSetImage(image.getImage(), [this](const auto& source, auto& destination)
                        {
                            processor.medianBlur(source, destination);
                        });
                        ImageDisplayer::uploadTexture(image.getImage(), texture);
                        image.setProcessingBaseImage(image.getImage());
                        state.brightness = 0;
                        state.contrast = 0;
                        state.saturation = 0;
                        state.warmth = 0;
                        state.invertColors = false;
                    }

                    if (ImGui::MenuItem("Sharpening"))
                    {
                        measureAndSetImage(image.getImage(), [this](const auto& source, auto& destination)
                        {
                            processor.sharpen(source, destination);
                        });
                        ImageDisplayer::uploadTexture(image.getImage(), texture);
                        image.setProcessingBaseImage(image.getImage());
                        state.brightness = 0;
                        state.contrast = 0;
                        state.saturation = 0;
                        state.warmth = 0;
                        state.invertColors = false;
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
                    bool adjustmentsChanged = ImGui::SliderInt("##Brightness", &state.brightness, 0, 100);
                    ImGui::Spacing();
                    ImGui::TextUnformatted("Contrast");
                    adjustmentsChanged |= ImGui::SliderInt("##Contrast", &state.contrast, 0, 100);
                    ImGui::Spacing();
                    ImGui::TextUnformatted("Saturation");
                    adjustmentsChanged |= ImGui::SliderInt("##Saturation", &state.saturation, 0, 100);
                    ImGui::Spacing();
                    ImGui::TextUnformatted("Warmth");
                    adjustmentsChanged |= ImGui::SliderInt("##Warmth", &state.warmth, 0, 100);
                    ImGui::Spacing();
                    adjustmentsChanged |= ImGui::Checkbox("Invert Colors", &state.invertColors);
                    ImGui::PopItemWidth();
                    ImGui::PopStyleColor(2);
                    if (adjustmentsChanged)
                    {
                        measureAndSetImage(image.getProcessingBaseImage(),
                            [this, brightness = state.brightness, contrast = state.contrast,
                             saturation = state.saturation, warmth = state.warmth,
                             invertColors = state.invertColors](const auto& source, auto& destination)
                            {
                                processor.applyAdjustments(source, destination, brightness, contrast,
                                                                saturation, warmth, invertColors);
                            });
                        ImageDisplayer::uploadTexture(image.getImage(), texture);
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

        const float aspectRatio = static_cast<float>(image.getImage().cols) / static_cast<float>(image.getImage().rows);

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
        snprintf(singleThreadedCpuTimeText, sizeof(singleThreadedCpuTimeText), "%.4f ms", benchmark.getLastResult().singleThreadedCpuMs);

        char multiThreadedCpuTimeText[32];
        snprintf(multiThreadedCpuTimeText, sizeof(multiThreadedCpuTimeText), "%.4f ms", benchmark.getLastResult().multiThreadedCpuMs);

        char gpuTimeText[32];
        snprintf(gpuTimeText, sizeof(gpuTimeText), "%.4f ms", benchmark.getLastResult().gpuMs);

        const float metricsY = contentTop + contentMax.y - contentMin.y - statsHeight;
        const float metricsWidth = std::max(0.0f, available.x);
        char cpu1Metric[48];
        snprintf(cpu1Metric, sizeof(cpu1Metric), "CPU 1T %s", singleThreadedCpuTimeText);
        char cpuMtMetric[48];
        snprintf(cpuMtMetric, sizeof(cpuMtMetric), "CPU MT %s", multiThreadedCpuTimeText);
        char gpuMetric[48];
        snprintf(gpuMetric, sizeof(gpuMetric), "GPU %s", gpuTimeText);
        char speedupText[64];
        const BenchmarkResult& latestResult = benchmark.getLastResult();
        const double singleThreadedGpuSpeedup = latestResult.gpuMs > 0.0
                                                    ? latestResult.singleThreadedCpuMs / latestResult.gpuMs
                                                    : 0.0;
        const double multiThreadedGpuSpeedup = latestResult.gpuMs > 0.0
                                                   ? latestResult.multiThreadedCpuMs / latestResult.gpuMs
                                                   : 0.0;
        snprintf(speedupText, sizeof(speedupText), "Speedup (1T / MT) %.2fx / %.2fx",
                 singleThreadedGpuSpeedup, multiThreadedGpuSpeedup);

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
}
