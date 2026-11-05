#include <common.hpp>
#include <guided_filter_neon.hpp>
#include <guided_filter_naive.hpp>

#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <guide_image> <input_image>\n";
        return 1;
    }

    const std::string guide_path = argv[1];
    const std::string input_path = argv[2];

    cv::Mat guide_u8 = cv::imread(guide_path);
    if (guide_u8.empty())
    {
        std::cerr << "Failed to load guide image: " << guide_path << "\n";
        return 1;
    }

    cv::Mat input_u8 = cv::imread(input_path);
    if (input_u8.empty())
    {
        std::cerr << "Failed to load input image: " << input_path << "\n";
        return 1;
    }

    Configuration config{2, 1e-10};

    cv::cvtColor(guide_u8, guide_u8, cv::COLOR_BGR2GRAY);
    cv::cvtColor(input_u8, input_u8, cv::COLOR_BGR2GRAY);

    cv::Mat input, guide;
    input_u8.convertTo(input, CV_32F, 1.0 / 255.0);
    guide_u8.convertTo(guide, CV_32F, 1.0 / 255.0);

    cv::Mat input_neon = input.clone();
    cv::Mat guide_neon = guide.clone();
 
    auto start = std::chrono::steady_clock::now();
    cv::Mat output_neon = cv::Mat::zeros(input.rows, input.cols, input.type());
    guided_filter_neon(input_neon, guide_neon, output_neon, config);
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << us << " neon us\n";

    start = std::chrono::steady_clock::now();
    cv::Mat output = cv::Mat::zeros(input.rows, input.cols, input.type());
    guided_filter(input, guide, output, config);
    end = std::chrono::steady_clock::now();
    us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << us << " naive us\n";

    cv::Mat output_u8;
    cv::Mat output_neon_u8;
    output.convertTo(output_u8, CV_8U, 255.0);
    output_neon.convertTo(output_neon_u8, CV_8U, 255.0);

    cv::imwrite("output.png", output_u8);
    cv::imwrite("output_neon.png", output_neon_u8);

    return 0;
}