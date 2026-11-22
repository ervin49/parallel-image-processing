#pragma once

#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>

class ImageDisplayer
{
public:
    static void uploadTexture(const cv::Mat& image, GLuint& texture);
};
