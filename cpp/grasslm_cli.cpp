#include <grasslm/generator.h>
#include <grasslm/model.h>
#include <grasslm/tokenizer.h>
#include <grasslm/weight_loader.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

struct CliArgs {
    std::string model_path;
    std::string vocab_path = "vocab.txt";
    std::string prompt;           // empty = interactive mode
    int max_tokens = 128;
    float temperature = 0.8f;
    int top_k = 50;
    float top_p = 0.9f;
    int seed = -1;                // -1 = random
    bool greedy = false;
    bool show_help = false;
};

static void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " <model.grasslm> [options]\n"
        << "\nOptions:\n"
        << "  --vocab <path>       Vocabulary file (default: vocab.txt)\n"
        << "  --prompt <text>      Single-shot mode: generate from prompt and exit\n"
        << "  --max_tokens <n>     Maximum tokens to generate (default: 128)\n"
        << "  --temperature <f>    Sampling temperature (default: 0.8)\n"
        << "  --top_k <n>          Top-k filtering (default: 50, 0=disabled)\n"
        << "  --top_p <f>          Nucleus sampling threshold (default: 0.9)\n"
        << "  --greedy             Use greedy decoding (argmax)\n"
        << "  --seed <n>           Random seed for reproducibility\n"
        << "  --help               Show this message\n";
}

static CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;

    if (argc < 2) {
        args.show_help = true;
        return args;
    }

    // First positional argument is the model path
    int i = 1;
    if (argv[1][0] != '-') {
        args.model_path = argv[1];
        i = 2;
    }

    for (; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
        } else if (arg == "--vocab" && i + 1 < argc) {
            args.vocab_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            args.max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            args.temperature = std::atof(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            args.top_k = std::atoi(argv[++i]);
        } else if (arg == "--top_p" && i + 1 < argc) {
            args.top_p = std::atof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = std::atoi(argv[++i]);
        } else if (arg == "--greedy") {
            args.greedy = true;
        } else if (args.model_path.empty()) {
            args.model_path = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            args.show_help = true;
        }
    }

    return args;
}

static void print_model_info(const grasslm::ModelConfig& config) {
    std::cout << "Model configuration:\n"
              << "  Layers:       " << config.n_layers << "\n"
              << "  d_model:      " << config.d_model << "\n"
              << "  d_reduce:     " << config.d_reduce << "\n"
              << "  d_ff:         " << config.d_ff << "\n"
              << "  Vocab size:   " << config.vocab_size << "\n"
              << "  Max seq len:  " << config.max_seq_len << "\n"
              << "  Dtype:        " << (config.dtype == 0 ? "float32" : "float16") << "\n";
}

static void run_generation(grasslm::Generator& generator,
                           const std::string& prompt,
                           const grasslm::GenerationConfig& config) {
    int token_count = 0;
    auto t_start = std::chrono::steady_clock::now();

    generator.generate_stream(prompt, config, [&](const std::string& token) {
        std::cout << token << std::flush;
        token_count++;
    });

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double tokens_per_sec = (elapsed_ms > 0) ? (token_count * 1000.0 / elapsed_ms) : 0;

    std::cout << "\n\n--- " << token_count << " tokens in "
              << (elapsed_ms / 1000.0) << "s ("
              << tokens_per_sec << " tok/s) ---\n";
}

int main(int argc, char* argv[]) {
    CliArgs args = parse_args(argc, argv);

    if (args.show_help || args.model_path.empty()) {
        print_usage(argv[0]);
        return args.show_help ? 0 : 1;
    }

    // Load model weights
    std::cout << "Loading model: " << args.model_path << "\n";
    grasslm::WeightLoader loader;
    if (!loader.load(args.model_path)) {
        std::cerr << "Error: failed to load model from " << args.model_path << "\n";
        return 1;
    }

    print_model_info(loader.config());

    // Initialize model
    grasslm::GrassLMModel model;
    if (!model.load(loader)) {
        std::cerr << "Error: failed to initialize model\n";
        return 1;
    }

    // Load tokenizer
    std::cout << "Loading vocabulary: " << args.vocab_path << "\n";
    grasslm::Tokenizer tokenizer;
    if (!tokenizer.load(args.vocab_path)) {
        std::cerr << "Error: failed to load vocabulary from " << args.vocab_path << "\n";
        return 1;
    }
    std::cout << "Vocabulary size: " << tokenizer.vocab_size() << "\n";

    // Create generator
    grasslm::Generator generator(model, tokenizer);
    grasslm::GenerationConfig config;
    config.max_tokens = args.max_tokens;
    config.temperature = args.greedy ? 0.0f : args.temperature;
    config.top_k = args.top_k;
    config.top_p = args.top_p;

    std::string mode_str = args.greedy
        ? "greedy"
        : "sampling (T=" + std::to_string(args.temperature)
            + ", top_k=" + std::to_string(args.top_k)
            + ", top_p=" + std::to_string(args.top_p) + ")";
    std::cout << "Generation mode: " << mode_str << "\n";

    // Single-shot mode
    if (!args.prompt.empty()) {
        std::cout << "Prompt: \"" << args.prompt << "\"\n";
        std::cout << std::string(60, '-') << "\n";
        run_generation(generator, args.prompt, config);
        return 0;
    }

    // Interactive mode
    std::cout << "\nGrassLM ready. Type a prompt (or 'quit' to exit):\n";

    std::string line;
    while (true) {
        std::cout << "\n> ";
        if (!std::getline(std::cin, line)) break;
        if (line == "quit" || line == "exit") break;
        if (line.empty()) continue;

        run_generation(generator, line, config);
    }

    return 0;
}
