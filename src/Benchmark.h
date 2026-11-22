#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

#include <algorithm>
#include <chrono>
#include <vector>

struct BenchmarkResult
{
    double singleThreadedCpuMs = 0.0;
    double multiThreadedCpuMs = 0.0;
    double gpuMs = 0.0;
    bool gpuPathMeasured = false;
};

class Benchmark
{
public:
    explicit Benchmark(const int repetitions = 5)
        : repetitions(std::max(1, repetitions))
    {
    }

    const BenchmarkResult& getLastResult() const
    {
        return lastResult;
    }

    void setLastResult(const BenchmarkResult& result)
    {
        lastResult = result;
    }

    const BenchmarkResult& getOriginalResult() const
    {
        return originalResult;
    }

    void setOriginalResult(const BenchmarkResult& result)
    {
        originalResult = result;
    }

    template <typename Operation>
    BenchmarkResult run(const cv::Mat& image, Operation operation)
    {
        cv::Mat output;
        return run(image, operation, output);
    }

    template <typename Operation>
    BenchmarkResult run(const cv::Mat& image, Operation operation, cv::Mat& output)
    {
        BenchmarkResult result;
        if (image.empty())
        {
            lastResult = result;
            return result;
        }

        const int originalThreadCount = cv::getNumThreads();

        cv::setNumThreads(1);
        cv::Mat singleThreadedOutput;
        result.singleThreadedCpuMs = measure([&]
        {
            operation(image, singleThreadedOutput);
        }, false);

        cv::setNumThreads(originalThreadCount);
        cv::Mat multiThreadedOutput;
        result.multiThreadedCpuMs = measure([&]
        {
            operation(image, multiThreadedOutput);
        }, false);
        multiThreadedOutput.copyTo(output);

        if (cv::ocl::haveOpenCL() && cv::ocl::useOpenCL())
        {
            cv::UMat gpuImage;
            cv::UMat gpuOutput;
            image.copyTo(gpuImage);

            result.gpuMs = measure([&]
            {
                operation(gpuImage, gpuOutput);
            }, true);
            result.gpuPathMeasured = true;
        }

        lastResult = result;
        return result;
    }

private:
    int repetitions;
    BenchmarkResult lastResult;
    BenchmarkResult originalResult;

    template <typename Operation>
    double measure(Operation operation, const bool synchronizeOpenCL) const
    {
        using Clock = std::chrono::steady_clock;

        operation(); // Warm up the path; this excludes first-use setup from the samples.
        if (synchronizeOpenCL)
            cv::ocl::finish();

        std::vector<double> samples;
        samples.reserve(repetitions);
        for (int i = 0; i < repetitions; ++i)
        {
            if (synchronizeOpenCL)
                cv::ocl::finish();

            const auto start = Clock::now();
            operation();
            if (synchronizeOpenCL)
                cv::ocl::finish();
            const auto end = Clock::now();

            samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        std::ranges::sort(samples);
        return samples[samples.size() / 2];
    }
};
