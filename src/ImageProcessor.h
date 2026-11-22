#pragma once

#include <opencv2/core.hpp>

class ImageProcessor
{
public:
    void toGrayscale(cv::InputArray source, cv::OutputArray result) const;
    void gaussianBlur(cv::InputArray source, cv::OutputArray result) const;
    void medianBlur(cv::InputArray source, cv::OutputArray result) const;
    void sharpen(cv::InputArray source, cv::OutputArray result) const;

    void applyAdjustments(const cv::Mat& source, cv::Mat& result, int brightness, int contrast, int saturation,
                          int warmth, bool invertColors) const;
    void applyAdjustments(const cv::UMat& source, cv::UMat& result, int brightness, int contrast, int saturation,
                          int warmth, bool invertColors) const;
};
