//
// Created by Armando Herrera on 2019-07-22.
//

#include <array>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <EasyTFLite.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char **argv) {
    // Init google logging
    google::InitGoogleLogging(argv[0]);

    // Get paths
    fs::path project_path(fs::current_path().parent_path());
    fs::path source(project_path.string() + "/examples/galaxyclassification/110887.jpg");
    fs::path model_path(project_path.string() + "/examples/galaxyclassification/galaxmobilenet.tflite");

    // Image scale function
    std::function<float(unsigned char)> scale_func = [](unsigned char x) -> float {
        return x / 255.;
    };

    // Get and parse arguments
    po::options_description description("An EasyTFLite Image Classification example");
    description.add_options()
            ("model", po::value<fs::path>(&model_path)->default_value(model_path),
             "Path to tensorFlow lite flatbuffer model")
            ("image", po::value<fs::path>(&source)->default_value(source),
             "Path to galaxy image to classify")
            ("help", "Produce help message");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, description), vm);
    if (vm.count("help")) {
        std::cout << description << "\n";
        return 0;
    }

    // Show selected arguments
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Image path: " << source << std::endl;

    // Get image
    cv::Mat image = cv::imread(source.string());

    // Init model
    EasyTFLite model(model_path);

    // Run inference
    float *raw_output = model.run_inference_ptrs(image, scale_func)[0];

    // Get Data - model's output is 37 floats between 0 and 1
    std::array<float, 37> output = {0};
    std::copy(raw_output, raw_output + 37, output.begin());

    // Interpret model output
    // The first float indicates the potential of the galaxy being rounded
    // The second float indicates the potential of the galaxy being a disk
    // The third float indicates the potential of the galaxy being a artifact
    // Other floats - TODO
    //  https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview/the-galaxy-zoo-decision-tree
    std::cout << "The galaxy is ";
    if (output[2] > std::max(output[0], output[1])) {
        std::cout << "an artifact\n";
        return 0;
    } else if (output[0] > output[1]) {
        std::cout << "rounded ";
    } else {
        std::cout << "disk shaped ";
    }

    std::cout << '\n';

    return 0;
}
