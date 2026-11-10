#include <common.hpp>
#include <guided_filter_naive.hpp>

#include <stdexcept>
#include <iostream>
#include <omp.h>

float get_mean(int i, int j, const cv::Mat &input, Configuration &config)
{
    const float *input_buff = input.ptr<float>(i, j);
    float acum = 0;

    int start_row = std::max(i - config.guide_size, 0);
    int start_col = std::max(j - config.guide_size, 0);
    int end_row = std::min(i + config.guide_size, input.rows - 1);
    int end_col = std::min(j + config.guide_size, input.cols - 1);

    for (int p = start_row; p <= end_row; p++)
    {
        for (int q = start_col; q <= end_col; q++)
        {
            acum += input.at<float>(p, q);
        }
    }

    return acum / ((end_row - start_row + 1) * (end_col - start_col + 1));
}

void fit_window(int i, int j, cv::Mat &input, cv::Mat &guide, cv::Mat &output, cv::Mat &guide_guide, cv::Mat &input_guide, Configuration &config)
{

    float mean_input = get_mean(i, j, input, config);
    float mean_guide = get_mean(i, j, guide, config);
    float mean_mul = get_mean(i, j, input_guide, config);
    float sigma = get_mean(i, j, guide_guide, config) - mean_guide * mean_guide;

    float a = (mean_mul - mean_input * mean_guide) / (sigma + config.eps);
    float b = mean_input - a * mean_guide;

    int start_row = std::max(i - config.guide_size, 0);
    int start_col = std::max(j - config.guide_size, 0);
    int end_row = std::min(i + config.guide_size, input.rows - 1);
    int end_col = std::min(j + config.guide_size, input.cols - 1);

    for (int p = start_row; p <= end_row; p++)
    {
        for (int q = start_col; q <= end_col; q++)
        {
            output.at<float>(p, q) += a * guide.at<float>(p, q) + b;
        }
    }
}

void rescale(cv::Mat &output, Configuration &config)
{
    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            int start_row = std::max(i - config.guide_size, 0);
            int start_col = std::max(j - config.guide_size, 0);
            int end_row = std::min(i + config.guide_size, output.rows - 1);
            int end_col = std::min(j + config.guide_size, output.cols - 1);

            output.at<float>(i, j) /= (end_row - start_row + 1) * (end_col - start_col + 1);
        }
    }
}

void guided_filter(cv::Mat &input, cv::Mat &guide, cv::Mat &output, Configuration &config)
{
    if ((input.rows != guide.rows) || (input.cols != guide.cols))
        throw std::length_error("Size of the guide does not match the input size");
    
    // cv::copyMakeBorder(input, input, config.guide_size, config.guide_size, config.guide_size, config.guide_size, cv::BORDER_CONSTANT, cv::Scalar(0));
    // cv::copyMakeBorder(output, output, config.guide_size, config.guide_size, config.guide_size, config.guide_size, cv::BORDER_CONSTANT, cv::Scalar(0));
    // cv::copyMakeBorder(guide, guide, config.guide_size, config.guide_size, config.guide_size, config.guide_size, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::Mat input_guide = input.mul(guide);
    cv::Mat guide_guide = guide.mul(guide);
    
    for (int i = config.guide_size + 3; i < input.rows - config.guide_size - 3; i++)
    {
        for (int j = config.guide_size + 3; j < input.cols - config.guide_size - 3; j++)
        {
            fit_window(i, j, input, guide, output, guide_guide, input_guide, config);
        }
    }

    rescale(output, config);
}