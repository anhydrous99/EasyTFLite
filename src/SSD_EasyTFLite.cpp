//
// Created by armandoh on 7/24/19.
//

#include "SSD_EasyTFLite.h"

SSD_EasyTFLite::SSD_EasyTFLite(const boost::filesystem::path &model_path) : EasyTFLite(model_path) {
    int input_index = input_tensors()[0];
    TfLiteType type = interpreter->tensor(input_index)->type;
    if (type == kTfLiteFloat32)
        quant_model = false;
    else if (type == kTfLiteUInt8)
        quant_model = true;
    else
        LOG(FATAL) << "Error: cannot handle input type " << type << " yet\n";
}

std::array<Eigen::Tensor<float, 2>, 4> SSD_EasyTFLite::run_inference(const cv::Mat &input_image) {
    // Get size of input image
    cv::Size input_image_size = input_image.size();

    int input_index = input_tensors()[0];

    // Resize image
    cv::Mat resized_image;
    std::vector<int> dims = get_tensor_dims(input_index);
    cv::Size target_size(dims[1], dims[2]);
    cv::resize(resized_image, resized_image, target_size);

    // Run inference
    std::vector<float *> output_tensors;
    if (quant_model) {
        std::vector<uint8_t *> itp(1);
        itp[0] = resized_image.data;;
        output_tensors = run_inference_ptrs<uint8_t, float>(itp);
    } else
        output_tensors = run_inference_ptrs(resized_image);

    auto output_tensor_indexes = input_tensors();

    int location_size = get_tensor_element_count(output_tensor_indexes[0]) / 4;
    int classes_score_size = get_tensor_element_count(output_tensor_indexes[1]);

    // Allocate output tensors
    Eigen::Tensor<float, 2> locations(location_size, 4);
    Eigen::Tensor<float, 2> classes(1, classes_score_size);
    Eigen::Tensor<float, 2> scores(1, classes_score_size);
    Eigen::Tensor<float, 2> n_detections(1, 1);

    // Copy data from model
    for (int i = 0; i < location_size; i++) {
        // Scale location data from between 0 and 1 to between 0 and the dimensions of the input image
        int j = i * 4;
        locations(i, 0) = output_tensors[0][j] * input_image_size.height;
        locations(i, 1) = output_tensors[0][j + 1] * input_image_size.width;
        locations(i, 2) = output_tensors[0][j + 2] * input_image_size.height;
        locations(i, 3) = output_tensors[0][j + 3] * input_image_size.width;
    }
    for (int i = 0; i < classes_score_size; i++) {
        classes(1, i) = output_tensors[1][i];
        scores(1, i) = output_tensors[0][i];
    }
    n_detections(0, 0) = output_tensors[3][0];


    // create and return output arrays of tensors
    std::array<Eigen::Tensor<float, 2>, 4> output = {locations, classes, scores, n_detections};
    return output;
}
