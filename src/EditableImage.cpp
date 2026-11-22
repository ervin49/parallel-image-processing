#include "EditableImage.h"
#include "Benchmark.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

bool EditableImage::load(const std::string& path)
{
    cv::Mat loadedImage = cv::imread(path);

    if (loadedImage.empty())
        return false;

    if (loadedImage.channels() == 3)
        cv::cvtColor(loadedImage, loadedImage, cv::COLOR_BGR2RGB);
    else if (loadedImage.channels() == 4)
        cv::cvtColor(loadedImage, loadedImage, cv::COLOR_BGRA2RGBA);

    loadedImage.copyTo(currentImage);
    loadedImage.copyTo(originalImage);
    loadedImage.copyTo(processingBaseImage);
    this->path = path;
    return true;
}

bool EditableImage::load(const std::string& path, Benchmark& benchmark)
{
    const cv::Mat sourceImage = cv::imread(path);
    if (sourceImage.empty())
        return false;

    cv::Mat loadedImage;
    benchmark.run(sourceImage, [](const auto& source, auto& destination)
    {
        cv::cvtColor(source, destination, cv::COLOR_BGR2RGB);
    }, loadedImage);

    if (loadedImage.empty())
        return false;

    loadedImage.copyTo(currentImage);
    loadedImage.copyTo(originalImage);
    loadedImage.copyTo(processingBaseImage);
    this->path = path;
    benchmark.setOriginalResult(benchmark.getLastResult());
    return true;
}

bool EditableImage::save() const
{
    if (currentImage.empty() || path.empty())
        return false;

    try
    {
        cv::Mat encodedImage;
        if (currentImage.channels() == 3)
            cv::cvtColor(currentImage, encodedImage, cv::COLOR_RGB2BGR);
        else if (currentImage.channels() == 4)
            cv::cvtColor(currentImage, encodedImage, cv::COLOR_RGBA2BGRA);
        else
            encodedImage = currentImage;

        return cv::imwrite(path, encodedImage);
    }
    catch (const cv::Exception& error)
    {
        std::cerr << "Failed to save image: " << error.what() << '\n';
        return false;
    }
}

bool EditableImage::saveAs(const std::string& path)
{
    if (currentImage.empty() || path.empty())
        return false;

    try
    {
        cv::Mat encodedImage;
        if (currentImage.channels() == 3)
            cv::cvtColor(currentImage, encodedImage, cv::COLOR_RGB2BGR);
        else if (currentImage.channels() == 4)
            cv::cvtColor(currentImage, encodedImage, cv::COLOR_RGBA2BGRA);
        else
            encodedImage = currentImage;

        if (!cv::imwrite(path, encodedImage))
            return false;

        this->path = path;
        return true;
    }
    catch (const cv::Exception& error)
    {
        std::cerr << "Failed to save image: " << error.what() << '\n';
        return false;
    }
}

void EditableImage::restoreOriginal()
{
    originalImage.copyTo(currentImage);
    originalImage.copyTo(processingBaseImage);
}
