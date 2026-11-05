#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

#include <guided_filter_naive.hpp>
#include <common.hpp>


TEST(GuidedFilterNaive, RescalingTestNaive) {
    cv::Mat test_image{cv::Mat::ones(10, 10, CV_32F)};
    cv::Mat reference_image{cv::Mat::ones(10, 10, CV_32F)};
    Configuration config{2, 1e-15}; 

    cv::filter2D(
        test_image, test_image, -1, 
        cv::Mat::ones(5, 5, CV_32F), cv::Point(-1, -1), 
        0, cv::BORDER_CONSTANT);

    rescale(test_image, config);
    
    for (int i = 0; i < test_image.rows; i++) {
        for (int j = 0; j < test_image.cols; j++) {
            EXPECT_EQ(test_image.at<float>(i, j), reference_image.at<float>(i, j));
        }
    }
}


struct WindowCenter_ {
    int x;
    int y;
};
using WindowCenter = WindowCenter_;


class GuidedFilterNaiveP : public testing::TestWithParam<WindowCenter> {};


TEST_P(GuidedFilterNaiveP, GetMeanTestNaive) { 
    cv::Mat test_image{cv::Mat::ones(5, 5, CV_32F)};
    Configuration config{4, 1e-15}; 

    WindowCenter center = GetParam();

    float mean = get_mean(center.x, center.y, test_image, config);

    EXPECT_EQ(mean, 1);
}


TEST_P(GuidedFilterNaiveP, FitWindow) {
    cv::Mat test_input{cv::Mat::ones(5, 5, CV_32F)};
    cv::Mat test_guide{cv::Mat::ones(5, 5, CV_32F)};
    cv::Mat test_output{cv::Mat::zeros(5, 5, CV_32F)};

    Configuration config{4, 0.0}; 

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j ++) {
            test_input.at<float>(i, j) = float(i + j);
            test_guide.at<float>(i, j) = 0.5 * float(i + j) - 1.0;
        }
    }

    WindowCenter center = GetParam();
    fit_window(center.x, center.y, test_input, test_guide, test_output, config);
    
    for (int i = 0; i < center.x - config.guide_size + test_input.rows; i++) {
        for (int j = 0; j < center.y - config.guide_size + test_input.cols; j ++) {
            EXPECT_EQ(test_output.at<float>(i, j), test_input.at<float>(i, j));
        }
    }
}


INSTANTIATE_TEST_SUITE_P(
    WindowTestsParametrized,
    GuidedFilterNaiveP,
    testing::Values(
        WindowCenter{1, 1}, 
        WindowCenter{2, 2}
    )
);
