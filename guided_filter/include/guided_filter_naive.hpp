#include <stdexcept>
#include <iostream>
#include <common.hpp>


#ifndef GUIDED_FILTER_NAIVE_HPP
#define GUIDED_FILTER_NAIVE_HPP

float get_mean(int i, int j, const cv::Mat &input, Configuration &config);

void fit_window(int i, int j, cv::Mat &input, cv::Mat &guide, cv::Mat &output, cv::Mat &guide_guide, cv::Mat &input_guide, Configuration &config);

void rescale(cv::Mat &output, Configuration &config);

void guided_filter(cv::Mat &input,cv::Mat &guide, cv::Mat &output, Configuration &config);


#endif

