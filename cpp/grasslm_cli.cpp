#include <grasslm/generator.h>
#include <grasslm/model.h>
#include <grasslm/tokenizer.h>
#include <grasslm/weight_loader.h>

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: grasslm_cli <model.grasslm> [vocab.txt]" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string vocab_path = (argc >= 3) ? argv[2] : "vocab.txt";

    // Load model weights
    grasslm::WeightLoader loader;
    if (!loader.load(model_path)) {
        std::cerr << "Error: failed to load model from " << model_path << std::endl;
        return 1;
    }

    // Initialize model
    grasslm::GrassLMModel model;
    if (!model.load(loader)) {
        std::cerr << "Error: failed to initialize model" << std::endl;
        return 1;
    }

    // Load tokenizer
    grasslm::Tokenizer tokenizer;
    if (!tokenizer.load(vocab_path)) {
        std::cerr << "Error: failed to load vocabulary from " << vocab_path << std::endl;
        return 1;
    }

    // Create generator
    grasslm::Generator generator(model, tokenizer);
    grasslm::GenerationConfig config;
    config.max_tokens = 128;
    config.temperature = 0.8f;
    config.top_p = 0.9f;

    // Interactive prompt loop
    std::cout << "GrassLM loaded. Type a prompt (or 'quit' to exit):" << std::endl;

    std::string line;
    while (true) {
        std::cout << "\n> ";
        if (!std::getline(std::cin, line)) break;
        if (line == "quit" || line == "exit") break;
        if (line.empty()) continue;

        generator.generate_stream(line, config, [](const std::string& token) {
            std::cout << token << std::flush;
        });
        std::cout << std::endl;
    }

    return 0;
}
