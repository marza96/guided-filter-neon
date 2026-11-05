#include <gtest/gtest.h>

#include <guided_filter_naive.hpp>
#include <guided_filter_neon.hpp>
#include <common.hpp>
#include <chrono>

TEST(GuidedFilterNeon, WriteBuffTestNeon)
{
    float buffer[4] = {1.0, 2.0, 3.0, 4.0};

    float ref_buffer_a[25] = {
        0.0, 1.0, 2.0, 3.0, 4.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
    };
    float out_buffer_a[25] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
    };
    
    write_buff(vld1q_f32(buffer), out_buffer_a, 0, 1, 5, 5);

    for (int i = 0; i < 25; i++){
        EXPECT_NEAR(out_buffer_a[i], ref_buffer_a[i], 1e-10);
    }

    float ref_buffer_b[25] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 2.0, 3.0, 
    };
    float out_buffer_b[25] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
    };
    
    write_buff(vld1q_f32(buffer), out_buffer_b, 4, 2, 5, 5);

    for (int i = 0; i < 25; i++){
        EXPECT_NEAR(out_buffer_b[i], ref_buffer_b[i], 1e-10);
    }


    float ref_buffer_c[25] = {
        2.0, 3.0, 4.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
    };
    float out_buffer_c[25] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
    };
    
    write_buff(vld1q_f32(buffer), out_buffer_c, 0, -1, 5, 5);

    for (int i = 0; i < 25; i++){
        EXPECT_NEAR(out_buffer_c[i], ref_buffer_c[i], 1e-10);
    }


    float ref_buffer_d[25] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 2.0, 3.0, 
        1.0, 2.0, 3.0, 4.0, 0.0, 
    };
    float out_buffer_d[25] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
    };
    
    write_buff(vld1q_f32(buffer), out_buffer_d, 3, 2, 5, 5);
    write_buff(vld1q_f32(buffer), out_buffer_d, 4, 0, 5, 5);

    for (int i = 0; i < 25; i++){
        EXPECT_NEAR(out_buffer_d[i], ref_buffer_d[i], 1e-10);
    }


    // float ref_buffer_e[25] = {
    //     0.0, 0.0, 0.0, 0.0, 0.0, 
    //     0.0, 0.0, 0.0, 0.0, 0.0, 
    //     0.0, 0.0, 0.0, 0.0, 0.0, 
    //     0.0, 0.0, 1.0, 2.0, 3.0, 
    //     0.0, 0.0, 0.0, 0.0, 0.0, 
    // };
    // float out_buffer_e[25] = {
    //     0.0, 0.0, 0.0, 0.0, 0.0, 
    //     0.0, 0.0, 0.0, 0.0, 0.0, 
    //     0.0, 0.0, 0.0, 0.0, 0.0, 
    //     0.0, 0.0, 0.0, 0.0, 0.0, 
    //     0.0, 0.0, 0.0, 0.0, 0.0, 
    // };
    
    // write_buff(vld1q_f32(buffer), out_buffer_e, 3, 2, 5, 5);

    // for (int i = 0; i < 25; i++){
    //     EXPECT_NEAR(out_buffer_e[i], ref_buffer_e[i], 1e-10);
    // }
}


TEST(GuidedFilterNeon, FetchBuffCachedTestNeon)
{
    constexpr int SIZE = 8;
    float buffer[SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    float ref_buffer_a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    CacheData cache_data;
    bool reset = true;
    for (int k = -4; k < 8; k++) {
        if (k > -4)
            reset = false;

        float32x4_t result_cached_neon = fetch_buff_cached(cache_data, buffer, 0, k, 1, SIZE, reset);
        float out_buffer_cached[4];
        vst1q_f32(out_buffer_cached, result_cached_neon);
        
        float32x4_t result_neon = fetch_buff(buffer, 0, k, 1, SIZE);
        float out_buffer[4];
        vst1q_f32(out_buffer, result_neon);

        for (int i = 0; i < 4; i++)
        {   
            std::cout << out_buffer[i] << " ";
            EXPECT_NEAR(out_buffer[i], out_buffer_cached[i], 1e-10);
        }
        std::cout << std::endl;
    }
}


TEST(GuidedFilterNeon, FetchBuffTestNeon)
{
    constexpr int SIZE = 5;
    float buffer[SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0};

    float ref_buffer_a[4] = {1.0, 2.0, 3.0, 4.0};
    float32x4_t result_a_neon = fetch_buff(buffer, 0, 0, 1, SIZE);
    float out_buffer_a[4];
    vst1q_f32(out_buffer_a, result_a_neon);

    for (int i = 0; i < 4; i++)
    {
        EXPECT_NEAR(out_buffer_a[i], ref_buffer_a[i], 1e-10);
    }

    float ref_buffer_b[4] = {3.0, 4.0, 5.0, 0.0};
    float32x4_t result_b_neon = fetch_buff(buffer, 0, 2, 1, SIZE);
    float out_buffer_b[4];
    vst1q_f32(out_buffer_b, result_b_neon);

    for (int i = 0; i < 4; i++)
    {
        EXPECT_NEAR(out_buffer_b[i], ref_buffer_b[i], 1e-10);
    }

    float ref_buffer_c[4] = {0.0, 0.0, 1.0, 2.0};
    float32x4_t result_c_neon = fetch_buff(buffer, 0, -2, 1, SIZE);
    float out_buffer_c[4];
    vst1q_f32(out_buffer_c, result_c_neon);

    for (int i = 0; i < 4; i++)
    {
        EXPECT_NEAR(out_buffer_c[i], ref_buffer_c[i], 1e-10);
    }

    float buffer_mat[25] = {
        1.0,  2.0,  3.0,  4.0,  5.0, 
        6.0,  7.0,  8.0,  9.0,  10.0, 
        11.0, 12.0, 13.0, 14.0, 15.0, 
        16.0, 17.0, 18.0, 19.0, 20.0, 
        21.0, 22.0, 23.0, 24.0, 25.0, 
    };


    float ref_buffer_d[4] = {0.0, 0.0, 1.0, 2.0};
    float32x4_t result_d_neon = fetch_buff(buffer_mat, 0, -2, SIZE, SIZE);
    float out_buffer_d[4];
    vst1q_f32(out_buffer_d, result_d_neon);

    for (int i = 0; i < 4; i++)
    {
        EXPECT_NEAR(out_buffer_d[i], ref_buffer_d[i], 1e-10);
    }


    float ref_buffer_e[4] = {0.0, 0.0, 0.0, 0.0};
    float32x4_t result_e_neon = fetch_buff(buffer_mat, 1, -4, SIZE, SIZE);
    float out_buffer_e[4];
    vst1q_f32(out_buffer_e, result_e_neon);

    for (int i = 0; i < 4; i++)
    {
        EXPECT_NEAR(out_buffer_e[i], ref_buffer_e[i], 1e-10);
    }
}

TEST(GuidedFilterNeon, RescaleTestNeon)
{
    cv::Mat test_image_neon{cv::Mat::ones(1001, 1001, CV_32F)};
    cv::Mat test_image_naive{cv::Mat::ones(1001, 1001, CV_32F)};
    Configuration config{2, 1e-15};

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j ++) {
            test_image_naive.at<float>(i, j) = float(i + j);
            test_image_neon.at<float>(i, j) = float(i + j);
        }
    }

    cv::filter2D(
        test_image_naive, test_image_naive, -1,
        cv::Mat::ones(5, 5, CV_32F), cv::Point(-1, -1),
        0, cv::BORDER_CONSTANT);

    cv::filter2D(
        test_image_neon, test_image_neon, -1,
        cv::Mat::ones(5, 5, CV_32F), cv::Point(-1, -1),
        0, cv::BORDER_CONSTANT);

    auto start = std::chrono::steady_clock::now();
    rescale(test_image_naive, config);
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "timing: " << us << " rescale naive us\n";

    start = std::chrono::steady_clock::now();
    rescale_neon(test_image_neon, config);
    end = std::chrono::steady_clock::now();
    us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "timing: " << us << " rescale neon us\n";

    for (int i = 0; i < test_image_naive.rows; i++)
    {
        for (int j = 0; j < test_image_naive.cols; j++)
        {
            EXPECT_NEAR(test_image_naive.at<float>(i, j), test_image_neon.at<float>(i, j), 1e-10);
        }
    }
}

TEST(GuidedFilterNeonP, GetMeanTestNeon)
{   
    int SIZE = 10;
    cv::Mat test_image{cv::Mat::ones(SIZE, SIZE, CV_32F)};
    Configuration config{6, 0};

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            test_image.at<float>(i, j) = i + j;
        }
    }

    float mean_buff[4];

    auto start = std::chrono::steady_clock::now();
    get_mean(2, 0, test_image, config);
    get_mean(2, 1, test_image, config);
    get_mean(2, 2, test_image, config);
    get_mean(2, 3, test_image, config);
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "timing: " << ns << " get_mean naive ns\n";

    int I = 2;
    int J = 0;
    start = std::chrono::steady_clock::now();

    static float32x4_t zero_neon = vdupq_n_f32(0.0);
    static float32x4_t guide_size_neon = vdupq_n_f32(float(config.guide_size));

    float32x4_t j_offset_neon{0.0f, 1.0f, 2.0f, 3.0f};
    float32x4_t i_neon = vdupq_n_f32(I);
    float32x4_t j_neon = vdupq_n_f32(J) + j_offset_neon;

    float32x4_t start_row_neon = vmaxq_f32(i_neon - guide_size_neon, zero_neon);
    float32x4_t start_col_neon = vmaxq_f32(j_neon - guide_size_neon, zero_neon);
    float32x4_t end_row_neon = vminq_f32(i_neon + guide_size_neon, vdupq_n_f32(float(test_image.rows - 1)));
    float32x4_t end_col_neon = vminq_f32(j_neon + guide_size_neon, vdupq_n_f32(float(test_image.cols - 1)));

    float32x4_t mean_neon = get_mean_neon(I, J, test_image, config) / ((end_row_neon - start_row_neon + 1) * (end_col_neon - start_col_neon + 1));;
    vst1q_f32(mean_buff, mean_neon);
    end = std::chrono::steady_clock::now();
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "timing: " << ns << " get_mean neon ns\n";

    EXPECT_NEAR(get_mean(I, J, test_image, config), mean_buff[0], 1e-6);
    EXPECT_NEAR(get_mean(I, J + 1, test_image, config), mean_buff[1], 1e-6);
    EXPECT_NEAR(get_mean(I, J + 2, test_image, config), mean_buff[2], 1e-6);
    EXPECT_NEAR(get_mean(I, J + 3, test_image, config), mean_buff[3], 1e-6);
}


TEST(GuidedFilterNeon, FitWindowNeon) {
    int CENTER_X = 2;
    int CENTER_Y = 2;
    int SIZE = 500;

    cv::Mat test_input{cv::Mat::ones(SIZE, SIZE, CV_32F)};
    cv::Mat test_guide{cv::Mat::ones(SIZE, SIZE, CV_32F)};
    cv::Mat test_output{cv::Mat::zeros(SIZE, SIZE, CV_32F)};
    cv::Mat dummy_output{cv::Mat::zeros(SIZE, SIZE, CV_32F)};
    cv::Mat test_output_neon{cv::Mat::zeros(SIZE, SIZE, CV_32F)};

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            test_input.at<float>(i, j) = i + j;
            test_guide.at<float>(i, j) = 2 * (i + j) + 1;
            
            test_output.at<float>(i, j) = 2.0;
            test_output_neon.at<float>(i, j) = 2.0;
        }
    }

    Configuration config{2, 1e-10}; 

    fit_window(CENTER_X, CENTER_Y, test_input, test_guide, dummy_output, config);
    auto start = std::chrono::steady_clock::now();
    fit_window(CENTER_X, CENTER_Y, test_input, test_guide, test_output, config);
    fit_window(CENTER_X, CENTER_Y + 1, test_input, test_guide, test_output, config);
    fit_window(CENTER_X, CENTER_Y + 2, test_input, test_guide, test_output, config);
    fit_window(CENTER_X, CENTER_Y + 3, test_input, test_guide, test_output, config);
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "timing: " << ns << " fit_window naive ns\n";

    cv::Mat input_guide = test_input.mul(test_guide);
    cv::Mat guide_guide = test_guide.mul(test_guide);
    fit_window_neon(CENTER_X, CENTER_Y, test_input, test_guide, dummy_output, guide_guide, input_guide, config);
    start = std::chrono::steady_clock::now();
    fit_window_neon(CENTER_X, CENTER_Y, test_input, test_guide, test_output_neon, guide_guide, input_guide, config);
    end = std::chrono::steady_clock::now();
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "timing: " << ns << " fit_window neon ns\n";

    for (int i = 0; i < CENTER_X - config.guide_size + test_input.rows; i++) {
        for (int j = 0; j < CENTER_Y - config.guide_size + test_input.cols; j ++) {
            EXPECT_NEAR(test_output.at<float>(i, j), test_output_neon.at<float>(i, j), 1e-5);
        }
    }

}
