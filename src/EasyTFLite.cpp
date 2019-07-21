//
// Created by Armando Herrera on 2019-07-20.
//

#include "EasyTFLite.h"
#include <opencv2/imgproc.hpp>

std::vector<float *> EasyTFLite::run_inference_ptrs(const cv::Mat &image) {
    // Assuming a scale between -1 and 1
    std::function<float(unsigned char)> scale_func = [](unsigned char x) -> float {
        return static_cast<float>(x) / 127.5 - 1;
    };
    return run_inference_ptrs(image, scale_func);
}

std::vector<float *>
EasyTFLite::run_inference_ptrs(const cv::Mat &image, const std::function<float(unsigned char)> &scale_func) {
    std::vector<int> it = input_tensors();
    std::vector<int> ot = output_tensors();
    int input_index = it[0];

    // Assuming a one input model
    if (it.size() != 1)
        LOG(FATAL) << "Error: OpenCV's Mat inferencing can only be done on models with one input.\n";
    // Assuming a Rank 4 input tensor for the model with the following [1, x, y, c]
    std::vector<int> dims = get_tensor_dims(input_index);
    if (dims.size() != 4)
        LOG(FATAL) << "Error: OpenCV's Mat inferencing requires a model with a Rank 4 input tensor.\n";

    // Resize image
    cv::Mat resized_image;
    cv::Size target_size(dims[1], dims[2]);
    cv::resize(image, resized_image, target_size);

    // Get the number of elements in image
    int total_elements = resized_image.channels() * resized_image.rows * resized_image.cols;
    const unsigned char *resized_image_ptr = resized_image.data;

    // Where converted data will be stored
    std::vector<float> float_data;

    // Run Scale and convert function
    std::transform(resized_image_ptr, resized_image_ptr + total_elements, std::back_inserter(float_data),
                   scale_func);

    // Fill input tensor
    fill_tensor<float>(float_data.data(), input_index);

    // Invoke model
    invoke();

    return get_output_tensor_ptrs<float>();
}

std::vector<float *>
EasyTFLite::run_inference_ptrs(const cv::Mat &image,
                               const std::function<std::vector<float>(cv::Mat)> &preprocess_func) {
    std::vector<int> it = input_tensors();
    std::vector<int> ot = output_tensors();
    int input_index = it[0];

    // Assuming a one input model
    if (it.size() != 1)
        LOG(FATAL) << "Error: OpenCV's Mat inferencing can only be done on models with one input.\n";
    // Assuming a Rank 4 input tensor for the model with the following [1, x, y, c]
    std::vector<int> dims = get_tensor_dims(input_index);
    if (dims.size() != 4)
        LOG(FATAL) << "Error: OpenCV's Mat inferencing requires a model with a Rank 4 input tensor.\n";

    // Resize image
    cv::Mat resized_image;
    cv::Size target_size(dims[1], dims[2]);
    cv::resize(image, resized_image, target_size);

    // Apply preprocess function
    std::vector<float> float_data = preprocess_func(resized_image);

    // Fill input tensor
    fill_tensor<float>(float_data.data(), input_index);

    // Invoke model
    invoke();

    return get_output_tensor_ptrs<float>();
}