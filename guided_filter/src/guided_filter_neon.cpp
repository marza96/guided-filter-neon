#include <common.hpp>
#include <guided_filter_neon.hpp>
#include <guided_filter_naive.hpp>

#include <stdexcept>
#include <iostream>

#include <arm_neon.h>
#include <omp.h>

void write_buff(float32x4_t buff_neon, float *buff, int p, int q, int rows, int cols)
{   
    if ((q + 4 <= cols) && (q >= 0)) {
        vst1q_f32(&buff[p * cols + q], buff_neon);

        return;
    }

    if (q < 0)
    {
        float tmp_buff[4] = {0.0, 0.0, 0.0, 0.0};

        vst1q_f32(tmp_buff, buff_neon);

        for (int k = -q; k < 4; k++)
        {
            buff[p * cols + q + k] = tmp_buff[k];
        }

        return;
    }


    float tmp_buff[4] = {0.0, 0.0, 0.0, 0.0};

    vst1q_f32(tmp_buff, buff_neon);

    for (int k = q; k < cols; k++)
    {
        buff[p * cols + k] = tmp_buff[k - q];
    }

    return;
}

float32x4_t fetch_buff_cached(CacheData &cache_data, const float *buff, int p, int q, int rows, int cols, bool reset) {

    if (reset){
        cache_data.offset = q;
        cache_data.buff_a = fetch_buff(buff, p, q, rows, cols);
        cache_data.buff_b = fetch_buff(buff, p, q + 4, rows, cols);
        
        return cache_data.buff_a;
    }

    if ((q - cache_data.offset) % 4 == 0) {
        cache_data.offset = q;
        cache_data.buff_a = cache_data.buff_b;
        cache_data.buff_b = fetch_buff(buff, p, q + 4, rows, cols);
    }

    switch (q - cache_data.offset)
    {
        case 3:
            return vextq_f32(cache_data.buff_a, cache_data.buff_b, 3);
        case 2:
            return vextq_f32(cache_data.buff_a, cache_data.buff_b, 2);
        case 1:
            return vextq_f32(cache_data.buff_a, cache_data.buff_b, 1);
        case 0:
            return cache_data.buff_a;
    }
}


float32x4_t fetch_buff(const float *buff, int p, int q, int rows, int cols)
{   
    if ((q + 4 <= cols) && (q >= 0))
        return vld1q_f32(&buff[p * cols + q]);

    if (q < 0)
    {
        switch (q)
        {
        case -1:
            // Shift right
            return vextq_f32(vdupq_n_f32(0.0), vld1q_f32(&buff[p * cols]), 3);
        case -2:
            return vextq_f32(vdupq_n_f32(0.0), vld1q_f32(&buff[p * cols]), 2);
        case -3:
            return vextq_f32(vdupq_n_f32(0.0), vld1q_f32(&buff[p * cols]), 1);
        default:
            return vdupq_n_f32(0.0);
        }
    }

    switch (q + 4 - cols)
    {
    case 1:
        // Shift left
        return vextq_f32(vld1q_f32(&buff[p * cols + q - 1]), vdupq_n_f32(0.0), 1);
    case 2:
        return vextq_f32(vld1q_f32(&buff[p * cols + q - 2]), vdupq_n_f32(0.0), 2);
    case 3:
        return vextq_f32(vld1q_f32(&buff[p * cols + q - 3]), vdupq_n_f32(0.0), 3);
    default:
        return vdupq_n_f32(0.0);
    }
}

float32x4_t get_mean_neon(int i, int j, const cv::Mat &input, Configuration &config)
{
    const float *input_buff = input.ptr<float>(0);

    float32x4_t acum_neon = vdupq_n_f32(0.0);

    int start_row = std::max(i - config.guide_size, 0);
    int end_row = std::min(i + config.guide_size, input.rows - 1);

    CacheData cache_data;

    for (int p = start_row; p <= end_row; p++)
    {
        for (int q = j - config.guide_size; q <= j + config.guide_size; q++)
        {
            acum_neon += fetch_buff_cached(cache_data, input_buff, p, q, input.rows, input.cols, q ==  j - config.guide_size);
        }
    }

    return acum_neon;
}

void fit_window_neon(int i, int j, cv::Mat &input, cv::Mat &guide, cv::Mat &output, cv::Mat &guide_guide, cv::Mat &input_guide, Configuration &config)
{
    float32x4_t zero_neon = vdupq_n_f32(0.0);
    float32x4_t guide_size_neon = vdupq_n_f32(float(config.guide_size));
    float32x4_t j_offset_neon{0.0f, 1.0f, 2.0f, 3.0f};

    float32x4_t i_neon = vdupq_n_f32(i);
    float32x4_t j_neon = vdupq_n_f32(j) + j_offset_neon;

    float32x4_t start_row_neon = vmaxq_f32(i_neon - guide_size_neon, zero_neon);
    float32x4_t start_col_neon = vmaxq_f32(j_neon - guide_size_neon, zero_neon);
    float32x4_t end_row_neon = vminq_f32(i_neon + guide_size_neon, vdupq_n_f32(float(input.rows - 1)));
    float32x4_t end_col_neon = vminq_f32(j_neon + guide_size_neon, vdupq_n_f32(float(input.cols - 1)));

    float32x4_t mean_input_neon = vdupq_n_f32(0.0);
    float32x4_t mean_guide_neon = vdupq_n_f32(0.0);
    float32x4_t mean_mul_neon = vdupq_n_f32(0.0);
    float32x4_t sigma_neon = vdupq_n_f32(0.0);

    int start_row = std::max(i - config.guide_size, 0);
    int end_row = std::min(i + config.guide_size, input.rows - 1);
    int start_col = std::max(-3, j - config.guide_size);
    int end_col = std::min(j + config.guide_size, input.cols - 1);

    float *input_buff = input.ptr<float>(0);
    float *guide_buff = guide.ptr<float>(0);
    float *input_guide_buff = input_guide.ptr<float>(0);
    float *guide_guide_buff = guide_guide.ptr<float>(0);

    for (int q = start_col; q <= end_col; q++)
    {
        for (int p = start_row; p <= end_row; p++)
        {   
            mean_input_neon += vld1q_f32(&input_buff[p * input.cols + q]);//  fetch_buff(input_buff, p, q, input.rows, input.cols);
            mean_guide_neon += vld1q_f32(&guide_buff[p * input.cols + q]);// fetch_buff(guide_buff, p, q, input.rows, input.cols);
            sigma_neon += vld1q_f32(&guide_guide_buff[p * input.cols + q]); //fetch_buff(guide_guide_buff, p, q, input.rows, input.cols);
            mean_mul_neon += vld1q_f32(&input_guide_buff[p * input.cols + q]); //fetch_buff(input_guide_buff, p, q, input.rows, input.cols);
        }
    }

    mean_input_neon /= ((end_row_neon - start_row_neon + 1) * (end_col_neon - start_col_neon + 1));
    mean_guide_neon /= ((end_row_neon - start_row_neon + 1) * (end_col_neon - start_col_neon + 1));
    mean_mul_neon /= ((end_row_neon - start_row_neon + 1) * (end_col_neon - start_col_neon + 1));
    sigma_neon = sigma_neon / ((end_row_neon - start_row_neon + 1) * (end_col_neon - start_col_neon + 1)) - mean_guide_neon * mean_guide_neon;

    float32x4_t a_neon = (mean_mul_neon - mean_input_neon * mean_guide_neon) / (sigma_neon + config.eps);
    float32x4_t b_neon = mean_input_neon - a_neon * mean_guide_neon;

    float *output_buff = output.ptr<float>(0);

    for (int q = start_col; q <= end_col; q++)
    {
        for (int p = start_row; p <= end_row; p++)
        {
            float32x4_t guide_neon = vld1q_f32(&guide_buff[p * input.cols + q]);// fetch_buff(guide_buff, p, q, input.rows, input.cols);
            float32x4_t output_current_neon = vld1q_f32(&output_buff[p * input.cols + q]); // fetch_buff(output_buff, p, q, input.rows, input.cols);

            float32x4_t output_sum_neon = output_current_neon + a_neon * guide_neon + b_neon;

            vst1q_f32(&output_buff[p * input.cols + q], output_sum_neon);
        }
    }
}

void rescale_neon(cv::Mat &output, Configuration &config)
{
    float *output_buff = output.ptr<float>(0);
    float res_buff[4] = {0.0, 0.0, 0.0, 0.0};
    float i_buff[4] = {0.0, 0.0, 0.0, 0.0};
    float j_buff[4] = {0.0, 1.0, 2.0, 3.0};

    float32x4_t guide_size_neon = vdupq_n_f32(float(config.guide_size));
    float32x4_t zero_neon = vdupq_n_f32(0.0);
    float32x4_t cols_neon = vdupq_n_f32(float(output.cols - 1));
    float32x4_t rows_neon = vdupq_n_f32(float(output.rows - 1));

    float32x4_t i_neon = vld1q_f32(i_buff);
    float32x4_t j_neon = vld1q_f32(j_buff);
    float32x4_t k_neon = vld1q_f32(j_buff);

    for (int k = 0; k < output.rows * output.cols; k += 4)
    {
        float32x4_t output_neon = vld1q_f32(&output_buff[k]);
        float32x4_t start_row_neon = vmaxq_f32(i_neon - guide_size_neon, zero_neon);
        float32x4_t start_col_neon = vmaxq_f32(j_neon - guide_size_neon, zero_neon);
        float32x4_t end_row_neon = vminq_f32(i_neon + guide_size_neon, rows_neon);
        float32x4_t end_col_neon = vminq_f32(j_neon + guide_size_neon, cols_neon);

        output_neon /= (end_row_neon - start_row_neon + 1) * (end_col_neon - start_col_neon + 1);

        if (output.rows * output.cols - k + 1 <= 4)
        {
            vst1q_f32(res_buff, output_neon);

            for (int q = 0; q < 4; q++)
            {
                if (k + q >= output.rows * output.cols)
                    break;

                output_buff[k + q] = res_buff[q];
            }

            return;
        }

        i_neon = vrndmq_f32((k_neon + 4) / (rows_neon + 1));
        j_neon = (k_neon + 4) - vrndmq_f32((k_neon + 4) / (cols_neon + 1)) * (cols_neon + 1);
        k_neon = k_neon + 4;

        vst1q_f32(&output_buff[k], output_neon);
    }
}

void guided_filter_neon(cv::Mat &input,cv::Mat &guide, cv::Mat &output, Configuration &config)
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
        for (int j = config.guide_size + 3; j < input.cols - config.guide_size - 3; j += 4)
        {
            if (j + 4 < input.cols - config.guide_size - 3) {
                fit_window_neon(i, j, input, guide, output, guide_guide, input_guide, config);

                continue;
            }

            for (int k = j; k < input.cols - config.guide_size - 3; k++) 
                fit_window(i, k, input, guide, output, guide_guide, input_guide, config);
        }
    }

    rescale_neon(output, config);
}