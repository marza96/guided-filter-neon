#include <common.hpp>
#include <stdexcept>
#include <iostream>

#ifndef GUIDED_FILTER_NEON_HPP
#define GUIDED_FILTER_NEON_HPP


struct CacheData_ {
    int offset;
    float32x4_t buff_a;
    float32x4_t buff_b;
};

using CacheData = CacheData_;


float32x4_t fetch_buff_cached(CacheData &cache_data, const float *buff, int p, int q, int rows, int cols, bool reset);

float32x4_t fetch_buff(const float *buff, int p, int q, int rows, int cols);

float32x4_t get_mean_neon(int i, int j, const cv::Mat &input, Configuration &config);

void fit_window_neon(int i, int j, cv::Mat &input, cv::Mat &guide, cv::Mat &output, cv::Mat &guide_guide, cv::Mat &input_guide,Configuration &config);

void rescale_neon(cv::Mat &output, Configuration &config);

void guided_filter_neon(cv::Mat &input,cv::Mat &guide, cv::Mat &output, Configuration &config);

void multiply_neon(cv::Mat &first, cv::Mat &second, cv::Mat &output);

void write_buff(float32x4_t buff_neon, float *buff, int p, int q, int rows, int cols);

#endif
