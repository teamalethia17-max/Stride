#include <iostream>
#include <string>
#include <vector>

#include "stride/tensor.hpp"
#include "stride/test_model.hpp"

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cout << "stride: CPU-first LLM inference engine\n";
            std::cout << "Usage: stride COMMAND [OPTIONS]\n";
            std::cout << "  --help\n  --version\n  test-model\n";
            return 0;
        }

        const std::string cmd = argv[1];
        if (cmd == "--help" || cmd == "-h") {
            std::cout << "stride: CPU-first LLM inference engine\n";
            std::cout << "Usage: stride COMMAND [OPTIONS]\n";
            std::cout << "  --help\n  --version\n  test-model\n";
            return 0;
        }
        if (cmd == "--version" || cmd == "-v") {
            std::cout << "stride " << STRIDE_VERSION_STRING << "\n";
            return 0;
        }
        if (cmd == "test-model") {
            auto model = stride::TinyModel::make();
            stride::Tensor input = stride::Tensor::from_f32({3}, {1.0f, 2.0f, 3.0f});
            stride::Tensor output = model.forward(input);
            std::cout << "test-model: input [1, 2, 3]\n";
            std::cout << "test-model: output [" << output.data()[0] << "]\n";
            return 0;
        }

        std::cerr << "Unknown command: " << cmd << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
