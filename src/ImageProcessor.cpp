#include "ImageProcessor.h"

#include <opencv2/imgproc.hpp>
#include <vector>

void ImageProcessor::toGrayscale(cv::InputArray source, cv::OutputArray result) const
{
    cv::cvtColor(source, result, source.channels() == 3 ? cv::COLOR_RGB2GRAY : cv::COLOR_RGBA2GRAY);
}

void ImageProcessor::gaussianBlur(cv::InputArray source, cv::OutputArray result) const
{
    cv::GaussianBlur(source, result, cv::Size(9, 9), 0);
}

void ImageProcessor::medianBlur(cv::InputArray source, cv::OutputArray result) const
{
    cv::medianBlur(source, result, 5);
}

void ImageProcessor::sharpen(cv::InputArray source, cv::OutputArray result) const
{
    const cv::Matx33f kernel = {
        0.0f, -1.0f, 0.0f,
        -1.0f, 5.0f, -1.0f,
        0.0f, -1.0f, 0.0f
    };
    cv::filter2D(source, result, -1, kernel);
}

namespace
{
    template <typename Image>
    void applyAdjustmentsImpl(const Image& source, Image& result, int brightness, int contrast, int saturation,
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
}

void ImageProcessor::applyAdjustments(const cv::Mat& source, cv::Mat& result, int brightness, int contrast,
                                      int saturation, int warmth, bool invertColors) const
{
    applyAdjustmentsImpl(source, result, brightness, contrast, saturation, warmth, invertColors);
}

void ImageProcessor::applyAdjustments(const cv::UMat& source, cv::UMat& result, int brightness, int contrast,
                                      int saturation, int warmth, bool invertColors) const
{
    applyAdjustmentsImpl(source, result, brightness, contrast, saturation, warmth, invertColors);
}
