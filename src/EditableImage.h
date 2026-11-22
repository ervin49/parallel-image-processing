#pragma once
#include <opencv2/core.hpp>
#include <string>

class Benchmark;

class EditableImage
{
public:
    EditableImage() = default;

    bool load(const std::string& path);
    bool load(const std::string& path, Benchmark& benchmark);
    bool save() const;
    bool saveAs(const std::string& path);
    void restoreOriginal();

    cv::Mat& getImage()
    {
        return currentImage;
    }
    const cv::Mat& getImage() const
    {
        return currentImage;
    }
    const cv::Mat& getOriginalImage() const
    {
        return originalImage;
    }

    const cv::Mat& getProcessingBaseImage() const
    {
        return processingBaseImage;
    }

    void setProcessingBaseImage(const cv::Mat& image)
    {
        image.copyTo(processingBaseImage);
    }

    void setImage(const cv::Mat& image)
    {
        image.copyTo(this->currentImage);
    }

    const std::string& getPath() const
    {
        return path;
    }

    void setPath(const std::string& path)
    {
        this->path = path;
    }

private:
    cv::Mat currentImage;
    cv::Mat originalImage;
    cv::Mat processingBaseImage;
    std::string path;

};
