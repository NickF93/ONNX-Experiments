#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <random>
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

std::vector<std::string> get_image_paths(const std::string& directory, int batch_size) {
    std::vector<std::string> image_paths;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            image_paths.push_back(entry.path().string());
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(image_paths.begin(), image_paths.end(), gen);

    if (batch_size < image_paths.size()) {
        image_paths.resize(batch_size);
    }

    return image_paths;
}

std::vector<float> preprocessImage(const cv::Mat& img, int input_width, int input_height, const std::string& save_path) {
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(input_width, input_height));
    
    // Ensure the image is in RGB format
    cv::Mat rgb_img;
    cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);
    
    rgb_img.convertTo(rgb_img, CV_32F, 1.0 / 255);  // Convert to float32 and scale to [0, 1]

    // Save the preprocessed image
    cv::Mat save_img;
    save_img = rgb_img * 255;  // Convert back to [0, 255] for saving
    save_img.convertTo(save_img, CV_8UC3);  // Convert to 8-bit for saving
    cv::cvtColor(save_img, save_img, cv::COLOR_RGB2BGR);  // Convert back to BGR before saving
    cv::imwrite(save_path, save_img);

    std::vector<float> input_tensor_values;
    input_tensor_values.assign((float*)rgb_img.datastart, (float*)rgb_img.dataend);
    return input_tensor_values;
}

cv::Mat convertOutputToImage(float* output_data, int width, int height) {
    cv::Mat output_img(height, width, CV_32FC3, output_data);
    output_img = output_img * 255;  // Convert back to [0, 255]
    output_img.convertTo(output_img, CV_8UC3);  // Convert to 8-bit
    cv::cvtColor(output_img, output_img, cv::COLOR_RGB2BGR);  // Convert back to BGR
    return output_img;
}

cv::Mat createImageGrid(const std::vector<cv::Mat>& original_images, const std::vector<cv::Mat>& reconstructed_images, int rows, int cols, int img_width, int img_height) {
    int grid_width = cols * img_width;
    int grid_height = 2 * rows * img_height;

    cv::Mat grid(grid_height, grid_width, original_images[0].type(), cv::Scalar(255, 255, 255));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            if (index < original_images.size()) {
                // Resize each image to the specified size before placing it into the grid
                cv::Mat original_resized, reconstructed_resized;
                cv::resize(original_images[index], original_resized, cv::Size(img_width, img_height));
                cv::resize(reconstructed_images[index], reconstructed_resized, cv::Size(img_width, img_height));

                cv::Mat original_roi = grid(cv::Rect(j * img_width, i * 2 * img_height, img_width, img_height));
                cv::Mat reconstructed_roi = grid(cv::Rect(j * img_width, (i * 2 + 1) * img_height, img_width, img_height));

                original_resized.copyTo(original_roi);
                reconstructed_resized.copyTo(reconstructed_roi);
            }
        }
    }

    return grid;
}

int main(int argc, char* argv[]) {
    int batch_size = (argc > 1) ? std::stoi(argv[1]) : 8;
    std::string directory = "/tmp/dataset/test/GOOD";
    std::vector<std::string> image_paths = get_image_paths(directory, batch_size);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const char* model_path = "/tmp/unet_model.onnx";
    Ort::Session session(env, model_path, session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_node_dims = input_tensor_info.GetShape();  // Model expects this shape

    std::cout << "Expected input dimensions: ";
    for (auto dim : input_node_dims) std::cout << dim << " ";
    std::cout << std::endl;

    std::vector<float> input_tensor_values;
    std::vector<cv::Mat> original_images;
    std::vector<cv::Mat> reconstructed_images;

    for (size_t i = 0; i < image_paths.size(); ++i) {
        cv::Mat img = cv::imread(image_paths[i]);
        if (img.empty()) {
            std::cerr << "Could not open or find the image: " << image_paths[i] << std::endl;
            return -1;
        }
        original_images.push_back(img.clone());  // Save the original image for display later
        std::string save_path = "/tmp/cpponnx_out/input_image" + std::to_string(i) + ".png";
        auto img_preprocessed = preprocessImage(img, 224, 224, save_path);  // Resize to 224x224 and save
        input_tensor_values.insert(input_tensor_values.end(), img_preprocessed.begin(), img_preprocessed.end());
    }

    std::cout << "Total input tensor values size: " << input_tensor_values.size() << std::endl;
    std::cout << "Expected total size: " << batch_size * 224 * 224 * 3 << std::endl;

    if (input_tensor_values.size() != batch_size * 224 * 224 * 3) {
        std::cerr << "Error: Input tensor size mismatch!" << std::endl;
        return -1;
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<int64_t> input_shape = {batch_size, 224, 224, 3};  // Batch size, Height, Width, Channels (NHWC)
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    fs::create_directory("/tmp/cpponnx_out");
    for (int i = 0; i < batch_size; ++i) {
        std::string output_path = "/tmp/cpponnx_out/output_image" + std::to_string(i) + ".png";
        cv::Mat output_img = convertOutputToImage(output_data + i * 224 * 224 * 3, 224, 224);
        reconstructed_images.push_back(output_img);  // Save reconstructed image for display later
        cv::imwrite(output_path, output_img);
    }

    int rows = std::ceil(std::sqrt(batch_size));
    int cols = rows;

    cv::Mat grid = createImageGrid(original_images, reconstructed_images, rows, cols, 224, 224);
    cv::imshow("Original and Reconstructed Images", grid);
    cv::waitKey(0);  // Wait indefinitely for a key press before closing the window

    std::cout << "Inference complete, grid displayed and images saved." << std::endl;
    return 0;
}
