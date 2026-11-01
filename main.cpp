#include <iostream>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"

#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include "external/nfd/src/include/nfd.h"

void openImage(cv::Mat& img, GLuint& texture)
{

    NFD_Init();
    nfdu8char_t* outPath;
    constexpr nfdu8filteritem_t filters[1] = {{.name = "Images", .spec = "png,jpg,jpeg,bmp,tiff"}};
    nfdopendialogu8args_t args = {.filterList = nullptr};
    args.filterList = filters;
    args.filterCount = 1;
    const nfdresult_t result = NFD_OpenDialogU8_With(&outPath, &args);
    if (result == NFD_OKAY)
    {
        img = cv::imread(outPath);
        if (!img.empty())
        {
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            if (texture != 0)
            {
                glDeleteTextures(1, &texture);
            }

            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE,
                         img.data);
        }
        NFD_FreePathU8(outPath);
    }
    else if (result == NFD_CANCEL)
    {
        puts("User pressed cancel.");
    }
    else
    {
        printf("Error: %s\n", NFD_GetError());
    }

    NFD_Quit();
}

int main()
{
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

    bool cpuCheck = true, gpuCheck = false;
    const ImGuiIO& io = ImGui::GetIO();
    ImFont* font = io.Fonts->AddFontFromFileTTF(ASSETS_DIR "/fonts/ARIAL.TTF");
    cv::Mat img;
    GLuint texture = 0;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0.11f,0.11f,0.11f,1.0f));
        ImGui::NewFrame();
        ImGui::PushFont(font);

        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y));
        bool open = true;
        ImGui::SetNextWindowPos(ImVec2(0.0f, ImGui::GetFrameHeight()));
        ImGui::Begin("Parallel Image Processing", &open,
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

        if (img.empty())
        {
            float fontHeight = ImGui::GetTextLineHeight();
            float fontSize = ImGui::CalcTextSize("No image loaded").x;
            ImGui::SetCursorPos(ImVec2(ImGui::GetWindowSize().x / 2 - fontSize / 2, ImGui::GetWindowSize().y / 2 - fontHeight * 1.5f));
            ImGui::Text("No image loaded");
            fontSize = ImGui::CalcTextSize("Open an image to start processing").x;
            ImGui::SetCursorPos(ImVec2(ImGui::GetWindowSize().x / 2 - fontSize / 2, ImGui::GetWindowSize().y / 2 ));
            ImGui::Text("Open an image to start processing.");
            fontSize = ImGui::CalcTextSize("Open Image").x;
            ImGui::SetCursorPos(ImVec2(ImGui::GetWindowSize().x / 2 - fontSize / 2, ImGui::GetWindowSize().y / 2 + fontHeight * 1.5f));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 14.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 6.0f));
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,        ImVec4(0.35f, 0.35f, 0.35f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,        ImVec4(0.45f, 0.45f, 0.45f, 1.0f));
            if (ImGui::Button("Open Image"))
            {
                openImage(img, texture);
            }
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(3);
        }
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Open", "Ctrl+O"))
                {
                    openImage(img, texture);
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
                    exit(0);
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
                if (ImGui::MenuItem("CPU Mode", nullptr, &cpuCheck))
                {
                    cpuCheck = true;
                    gpuCheck = false;
                }
                if (ImGui::MenuItem("GPU Mode", nullptr, &gpuCheck))
                {
                    gpuCheck = true;
                    cpuCheck = false;
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Filters"))
            {
                if (ImGui::MenuItem("Grayscale"))
                {
                    if (!img.empty())
                    {
                        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
                        if (texture != 0)
                        {
                            glDeleteTextures(1, &texture);
                        }

                        glGenTextures(1, &texture);
                        glBindTexture(GL_TEXTURE_2D, texture);

                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
                        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);


                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, img.cols, img.rows, 0, GL_RED, GL_UNSIGNED_BYTE,
                                     img.data);
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (!img.empty())
        {
            bool is_open = true;
            ImGui::Begin("Image", &is_open, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);

            ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(texture)),
                         ImVec2(static_cast<float>(img.cols), static_cast<float>(img.rows)));

            ImGui::End();
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
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
