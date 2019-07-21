//
// Created by aherrera on 7/17/19.
//

#include "TFLite.h"
#include "gtest/gtest.h"

#include <array>
#include <string>
#include <fstream>
#include <memory>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

namespace {


    ////////////// Tests to make sure the input tensors remain allocated over life of object //////////////
    TEST(TFLiteTest, MultiInput_SingleOutput_PointerFill_Test) {
        TFLite tflite(boost::filesystem::path("../../tests/test-models/multi_input_single_output.tflite"));

        // Get indexes of input tensors
        std::vector<int> input_tensor_indexes = tflite.input_tensors();

        // Saving added tensors
        std::vector<std::vector<float>> added_tensors;

        // create random number generator
        boost::random::random_device d;
        boost::random::mt19937 gen(d());
        boost::normal_distribution<float> dist(0.0, 1.0);

        // Fill tensors with random numbers while storing them
        for (int tensor_index : input_tensor_indexes) {
            // Get number of elements in tensor
            int tensor_size = tflite.get_tensor_element_count(tensor_index);
            // Create empty vector with the size of the tensor
            std::vector<float> random_input(tensor_size);
            // Fill the vector with random numbers
            std::generate(random_input.begin(), random_input.end(), [&]() { return dist(gen); });
            // Save generated tensor
            added_tensors.push_back(random_input);
            // Fill tensor
            tflite.fill_tensor(random_input.data(), tensor_index);
        }

        tflite.invoke();

        int i = 0;
        for (int tensor_index : input_tensor_indexes) {
            // Get pointer to data
            auto *data_ptr = tflite.get_tensor_ptr<float>(tensor_index);
            // Get number of elements in tensor
            int tensor_size = tflite.get_tensor_element_count(tensor_index);
            for (int j = 0; j < tensor_size; j++) {
                ASSERT_FLOAT_EQ(data_ptr[j], added_tensors[i][j]);
            }
            i++;
        }
    }

    TEST(TFLiteTest, SingleInput_MultiOutput_PointerFill_Test) {
        TFLite tflite(boost::filesystem::path("../../tests/test-models/single_input_multi_output.tflite"));

        // Get indexes of input tensors
        std::vector<int> input_tensor_indexes = tflite.input_tensors();

        // Saving added tensors
        std::vector<std::vector<float>> added_tensors;

        // create random number generator
        boost::random::random_device d;
        boost::random::mt19937 gen(d());
        boost::normal_distribution<float> dist(0.0, 1.0);

        // Fill tensors with random numbers while storing them
        for (int tensor_index : input_tensor_indexes) {
            // Get number of elements in tensor
            int tensor_size = tflite.get_tensor_element_count(tensor_index);
            // Create empty vector with the size of the tensor
            std::vector<float> random_input(tensor_size);
            // Fill the vector with random numbers
            std::generate(random_input.begin(), random_input.end(), [&]() { return dist(gen); });
            // Save generated tensor
            added_tensors.push_back(random_input);
            // Fill tensor
            tflite.fill_tensor(random_input.data(), tensor_index);
        }

        tflite.invoke();

        int i = 0;
        for (int tensor_index : input_tensor_indexes) {
            // Get pointer to data
            auto *data_ptr = tflite.get_tensor_ptr<float>(tensor_index);
            // Get number of elements in tensor
            int tensor_size = tflite.get_tensor_element_count(tensor_index);
            for (int j = 0; j < tensor_size; j++) {
                ASSERT_FLOAT_EQ(data_ptr[j], added_tensors[i][j]);
            }
            i++;
        }
    }

    TEST(TFLiteTest, SingleVolumeInput_PointerFill_Test) {
        TFLite tflite(boost::filesystem::path("../../tests/test-models/single_volume_input.tflite"));

        // Get indexes of input tensors
        std::vector<int> input_tensor_indexes = tflite.input_tensors();

        // Saving added tensors
        std::vector<std::vector<float>> added_tensors;

        // create random number generator
        boost::random::random_device d;
        boost::random::mt19937 gen(d());
        boost::normal_distribution<float> dist(0.0, 1.0);

        // Fill tensors with random numbers while storing them
        for (int tensor_index : input_tensor_indexes) {
            // Get number of elements in tensor
            int tensor_size = tflite.get_tensor_element_count(tensor_index);
            // Create empty vector with the size of the tensor
            std::vector<float> random_input(tensor_size);
            // Fill the vector with random numbers
            std::generate(random_input.begin(), random_input.end(), [&]() { return dist(gen); });
            // Save generated tensor
            added_tensors.push_back(random_input);
            // Fill tensor
            tflite.fill_tensor(random_input.data(), tensor_index);
        }

        tflite.invoke();

        int i = 0;
        for (int tensor_index : input_tensor_indexes) {
            // Get pointer to data
            auto *data_ptr = tflite.get_tensor_ptr<float>(tensor_index);
            // Get number of elements in tensor
            int tensor_size = tflite.get_tensor_element_count(tensor_index);
            for (int j = 0; j < tensor_size; j++) {
                ASSERT_FLOAT_EQ(data_ptr[j], added_tensors[i][j]);
            }
            i++;
        }
    }

    ////////////// Tests to make sure calculation are done correctly //////////////
    TEST(TFLiteTest, SingleInput_MultiOutput_CorrectCalculation_Test) {
        // Expected output data
        std::array<float, 6> output1 = {-0.14983515, 0.47272223, -0.73745316, 0.46977115, -0.07364011, 0.26235366};
        std::array<float, 6> output2 = {0.11423676, -0.04815429, -0.52054065, -1.1527455, 0.12045179, -0.06280062};

        // Allocating input container & init with zeros
        std::array<float, 4096> input = {0.0};

        // Grab input data
        std::ifstream input_data_file("../../tests/random-data.txt");
        if (input_data_file.is_open()) {
            std::string line;
            int i = 0;
            while (getline(input_data_file, line)) {
                input[i] = std::stof(line);
                i++;
            }
            input_data_file.close();
        }

        // Create model
        TFLite tflite(boost::filesystem::path("../../tests/test-models/single_input_multi_output.tflite"));

        // Get indexes of input and output tensors
        std::vector<int> input_tensor_indexes = tflite.input_tensors();

        // Fill tensor
        tflite.fill_tensor(input.data(), input_tensor_indexes[0]);

        // Invoke
        tflite.invoke();

        // Get output tensors
        std::vector<int> output_tensor_indexes = tflite.output_tensors();
        auto *output1_inter = tflite.get_tensor_ptr<float>(output_tensor_indexes[0]);
        auto *output2_inter = tflite.get_tensor_ptr<float>(output_tensor_indexes[1]);

        float abs_error = 0.00001;

        for (int i = 0; i < 6; i++)
            ASSERT_NEAR(output1_inter[i], output1[i], abs_error);
        for (int i = 0; i < 6; i++)
            ASSERT_NEAR(output2_inter[i], output2[i], abs_error);
    }

    TEST(TFLiteTest, MultiInput_SimpleOutput_CorrectCalculation_Test) {
        // Expected output data
        std::array<float, 6> output = {-0.68169963, -0.54100305, 1.3366573, -1.2651198, 0.755826, -0.27840295};

        // Allocating input container & init with zeros
        std::array<float, 4096> input = {0.0};

        // Grab input data
        std::ifstream input_data_file("../../tests/random-data.txt");
        if (input_data_file.is_open()) {
            std::string line;
            int i = 0;
            while (getline(input_data_file, line)) {
                input[i] = std::stof(line);
                i++;
            }
            input_data_file.close();
        }

        // Create model
        TFLite tflite(boost::filesystem::path("../../tests/test-models/multi_input_single_output.tflite"));

        // Get indexes of input and output tensors
        std::vector<int> input_tensor_indexes = tflite.input_tensors();

        // Fill tensor
        // fill_tensor copies input's data, so we can use it for both inputs
        tflite.fill_tensor(input.data(), input_tensor_indexes[0]);
        tflite.fill_tensor(input.data(), input_tensor_indexes[1]);

        // Invoke
        tflite.invoke();

        // Get output tensors
        std::vector<int> output_tensor_indexes = tflite.output_tensors();
        auto *output_inter = tflite.get_tensor_ptr<float>(output_tensor_indexes[0]);

        float abs_error = 0.00001;

        for (int i = 0; i < 6; i++)
            ASSERT_NEAR(output_inter[i], output[i], abs_error);

    }

    TEST(TFLiteTest, SingleVolumeInput_CorrectCalculation_Test) {
        // Expected output data
        std::array<float, 10> output = {0.21751887, 0.7512632, 0.13513072, 0.12721045, 0.923916, -0.9657576, -0.3309331,
                                        0.2791444, 1.629914, 2.21699};

        // Allocating input container & init with zeros
        // Dims are 1x16x16x16x1=4096
        std::array<float, 4096> input = {0.0};

        // Grab input data
        std::ifstream input_data_file("../../tests/random-data.txt");
        if (input_data_file.is_open()) {
            std::string line;
            int i = 0;
            while (getline(input_data_file, line)) {
                input[i] = std::stof(line);
                i++;
            }
            input_data_file.close();
        }

        // Create model
        TFLite tflite(boost::filesystem::path("../../tests/test-models/single_volume_input.tflite"));

        // Get indexes of input and output tensors
        std::vector<int> input_tensor_indexes = tflite.input_tensors();

        // Fill tensor
        tflite.fill_tensor(input.data(), input_tensor_indexes[0]);

        // Invoke
        tflite.invoke();

        // Get output tensors
        std::vector<int> output_tensor_indexes = tflite.output_tensors();
        auto *output_inter = tflite.get_tensor_ptr<float>(output_tensor_indexes[0]);

        float abs_error = 0.00001;

        for (int i = 0; i < 10; i++)
            ASSERT_NEAR(output_inter[i], output[i], abs_error);

    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging(argv[0]);
    return RUN_ALL_TESTS();
}