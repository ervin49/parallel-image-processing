#include "EditableImage.h"
#include "ImageProcessor.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
    void require(const bool condition, const std::string& message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    bool imagesEqual(const cv::Mat& first, const cv::Mat& second)
    {
        return first.size() == second.size() && first.type() == second.type() &&
               cv::countNonZero(first.reshape(1) != second.reshape(1)) == 0;
    }

    class TemporaryDirectory
    {
    public:
        TemporaryDirectory()
        {
            const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
            path = std::filesystem::temp_directory_path() /
                   ("parallel-image-processing-tests-" + std::to_string(timestamp));
            std::filesystem::create_directories(path);
        }

        ~TemporaryDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }

        std::filesystem::path path;
    };

    void testImageProcessorFilters()
    {
        ImageProcessor processor;
        cv::Mat source(11, 11, CV_8UC3, cv::Scalar(40, 40, 40));
        source.at<cv::Vec3b>(5, 5) = cv::Vec3b(100, 140, 180);
        cv::Mat result;

        processor.toGrayscale(source, result);
        require(result.size() == source.size(), "grayscale should preserve image dimensions");
        require(result.channels() == 1, "grayscale should produce one channel");
        require(result.at<unsigned char>(0, 0) == 40, "grayscale should convert RGB pixels correctly");
        require(result.at<unsigned char>(5, 5) == 133, "grayscale should convert RGB pixels correctly");

        cv::Mat rgba(1, 1, CV_8UC4, cv::Scalar(20, 40, 60, 255));
        processor.toGrayscale(rgba, result);
        require(result.at<unsigned char>(0, 0) == 36, "grayscale should convert RGBA pixels correctly");

        processor.gaussianBlur(source, result);
        require(result.size() == source.size() && result.type() == source.type(),
                "Gaussian blur should preserve dimensions and type");
        require(result.at<cv::Vec3b>(5, 5)[0] < source.at<cv::Vec3b>(5, 5)[0],
                "Gaussian blur should soften a bright detail");
        require(result.at<cv::Vec3b>(5, 6)[0] > source.at<cv::Vec3b>(5, 6)[0],
                "Gaussian blur should spread a bright detail to nearby pixels");

        processor.medianBlur(source, result);
        require(result.size() == source.size() && result.type() == source.type(),
                "median blur should preserve dimensions and type");
        require(result.at<cv::Vec3b>(5, 5) == cv::Vec3b(40, 40, 40),
                "median blur should remove an isolated bright detail");

        processor.sharpen(source, result);
        require(result.size() == source.size() && result.type() == source.type(),
                "sharpen should preserve dimensions and type");
        require(result.at<cv::Vec3b>(5, 5)[0] > source.at<cv::Vec3b>(5, 5)[0],
                "sharpen should increase contrast at a bright detail");

        cv::UMat sourceUmat;
        source.copyTo(sourceUmat);
        cv::UMat umatResult;
        processor.toGrayscale(sourceUmat, umatResult);
        cv::Mat umatResultMat;
        umatResult.copyTo(umatResultMat);
        processor.toGrayscale(source, result);
        require(imagesEqual(result, umatResultMat), "grayscale should give matching Mat and UMat results");
        processor.gaussianBlur(sourceUmat, umatResult);
        umatResult.copyTo(umatResultMat);
        processor.gaussianBlur(source, result);
        require(imagesEqual(result, umatResultMat), "Gaussian blur should give matching Mat and UMat results");
        processor.medianBlur(sourceUmat, umatResult);
        umatResult.copyTo(umatResultMat);
        processor.medianBlur(source, result);
        require(imagesEqual(result, umatResultMat), "median blur should give matching Mat and UMat results");
        processor.sharpen(sourceUmat, umatResult);
        umatResult.copyTo(umatResultMat);
        processor.sharpen(source, result);
        require(imagesEqual(result, umatResultMat), "sharpen should give matching Mat and UMat results");
    }

    void testAdjustmentsUseTheProvidedBaseImage()
    {
        ImageProcessor processor;
        const cv::Mat base(1, 1, CV_8UC3, cv::Scalar(50, 60, 70));
        cv::Mat adjusted;

        processor.applyAdjustments(base, adjusted, 10, 0, 0, 0, false);
        require(adjusted.at<cv::Vec3b>(0, 0) == cv::Vec3b(60, 70, 80),
                "brightness should be applied to the supplied base image");

        processor.applyAdjustments(base, adjusted, 20, 0, 0, 0, false);
        require(adjusted.at<cv::Vec3b>(0, 0) == cv::Vec3b(70, 80, 90),
                "changing brightness should recompute from the base, not accumulate");

        processor.applyAdjustments(base, adjusted, 0, 0, 0, 0, false);
        require(imagesEqual(base, adjusted), "zero adjustments should reproduce the base image");

        processor.applyAdjustments(base, adjusted, 0, 100, 0, 0, false);
        require(adjusted.at<cv::Vec3b>(0, 0) == cv::Vec3b(0, 0, 12),
                "contrast should increase distance from the midpoint");

        processor.applyAdjustments(base, adjusted, 0, 0, 100, 0, false);
        cv::Mat baseHsv;
        cv::Mat adjustedHsv;
        cv::cvtColor(base, baseHsv, cv::COLOR_RGB2HSV);
        cv::cvtColor(adjusted, adjustedHsv, cv::COLOR_RGB2HSV);
        require(adjustedHsv.at<cv::Vec3b>(0, 0)[1] > baseHsv.at<cv::Vec3b>(0, 0)[1],
                "saturation should increase color saturation");

        processor.applyAdjustments(base, adjusted, 0, 0, 0, 10, false);
        require(adjusted.at<cv::Vec3b>(0, 0) == cv::Vec3b(60, 60, 60),
                "warmth should increase red and reduce blue");

        processor.applyAdjustments(base, adjusted, 0, 0, 0, 0, true);
        require(adjusted.at<cv::Vec3b>(0, 0) == cv::Vec3b(205, 195, 185),
                "invert should complement each color channel");

        cv::UMat baseUmat;
        cv::UMat adjustedUmat;
        base.copyTo(baseUmat);
        processor.applyAdjustments(baseUmat, adjustedUmat, 10, 20, 30, 10, true);
        cv::Mat adjustedFromUmat;
        adjustedUmat.copyTo(adjustedFromUmat);
        processor.applyAdjustments(base, adjusted, 10, 20, 30, 10, true);
        require(imagesEqual(adjusted, adjustedFromUmat),
                "adjustments should give matching Mat and UMat results");

        const cv::Mat grayBase(1, 1, CV_8UC1, cv::Scalar(64));
        processor.applyAdjustments(grayBase, adjusted, 10, 0, 100, 100, true);
        require(adjusted.at<unsigned char>(0, 0) == 181,
                "grayscale adjustments should apply brightness and inversion and skip color-only controls");
    }

    void testEditableImageLoadSaveAndRestore()
    {
        TemporaryDirectory temporaryDirectory;
        const std::filesystem::path imagePath = temporaryDirectory.path / "sample.png";

        cv::Mat original(3, 4, CV_8UC3);
        for (int row = 0; row < original.rows; ++row)
        {
            for (int column = 0; column < original.cols; ++column)
                original.at<cv::Vec3b>(row, column) = cv::Vec3b(row * 20, column * 30, (row + column) * 15);
        }

        EditableImage image;
        image.setImage(original);
        require(image.saveAs(imagePath.string()), "Save As should write the image");
        require(image.getPath() == imagePath.string(), "Save As should update the document path");

        EditableImage loadedImage;
        require(loadedImage.load(imagePath.string()), "EditableImage should load a saved image");
        require(imagesEqual(original, loadedImage.getImage()), "load/save should preserve RGB pixel values");
        require(imagesEqual(original, loadedImage.getOriginalImage()), "loaded pixels should be retained as original");

        cv::Mat changed = original.clone();
        changed.setTo(cv::Scalar(200, 150, 100));
        loadedImage.setImage(changed);
        loadedImage.restoreOriginal();
        require(imagesEqual(original, loadedImage.getImage()), "restoreOriginal should restore loaded pixels");
        require(imagesEqual(original, loadedImage.getProcessingBaseImage()),
                "restoreOriginal should also reset the processing base");

        changed.setTo(cv::Scalar(10, 20, 30));
        loadedImage.setImage(changed);
        require(loadedImage.save(), "Save should write to the document's current path");

        EditableImage savedAgain;
        require(savedAgain.load(imagePath.string()), "saved image should remain loadable");
        require(imagesEqual(changed, savedAgain.getImage()), "Save should persist the current pixels");
    }
}

int main()
{
    try
    {
        testImageProcessorFilters();
        testAdjustmentsUseTheProvidedBaseImage();
        testEditableImageLoadSaveAndRestore();
        std::cout << "All image unit tests passed.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Image unit test failed: " << error.what() << '\n';
        return 1;
    }
}
