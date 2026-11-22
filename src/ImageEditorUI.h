#pragma once

#include <GLFW/glfw3.h>
#include "imgui.h"

class Benchmark;
class EditableImage;
class EditorUiState;
class ImageProcessor;

class ImageEditorUI
{
public:
    ImageEditorUI(EditableImage& image, EditorUiState& state, ImageProcessor& processor, Benchmark& benchmark,
                  GLuint& texture);
    ~ImageEditorUI();

    void draw(const ImVec2& displaySize, ImFont* font, GLFWwindow* window);

private:
    EditableImage& image;
    EditorUiState& state;
    ImageProcessor& processor;
    Benchmark& benchmark;
    GLuint& texture;
    GLuint sidebarTexture = 0;

    void loadSidebarIcon();
    void openImage();
};
