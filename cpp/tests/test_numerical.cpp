#include <grasslm/model.h>
#include <grasslm/tensor.h>
#include <grasslm/weight_loader.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace grasslm;

// ---------------------------------------------------------------------------
// Binary file readers for test data exported by python/export_test_data.py
// ---------------------------------------------------------------------------

/// Read a tensor from binary format: uint32 ndim, uint32 shape[], float32 data[].
static Tensor read_tensor(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open tensor file: " + path);
    }

    uint32_t ndim;
    if (std::fread(&ndim, sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("Failed to read ndim from: " + path);
    }

    std::vector<int> shape(ndim);
    int numel = 1;
    for (uint32_t d = 0; d < ndim; d++) {
        uint32_t dim_size;
        if (std::fread(&dim_size, sizeof(uint32_t), 1, f) != 1) {
            std::fclose(f);
            throw std::runtime_error("Failed to read shape from: " + path);
        }
        shape[d] = static_cast<int>(dim_size);
        numel *= shape[d];
    }

    std::vector<float> data(numel);
    if (std::fread(data.data(), sizeof(float), numel, f) !=
        static_cast<size_t>(numel)) {
        std::fclose(f);
        throw std::runtime_error("Failed to read data from: " + path);
    }

    std::fclose(f);
    return Tensor(shape, std::move(data));
}

/// Read token IDs from binary format: uint32 length, int32 ids[].
static std::vector<int> read_token_ids(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open token IDs file: " + path);
    }

    uint32_t length;
    if (std::fread(&length, sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("Failed to read length from: " + path);
    }

    std::vector<int> ids(length);
    if (std::fread(ids.data(), sizeof(int32_t), length, f) !=
        static_cast<size_t>(length)) {
        std::fclose(f);
        throw std::runtime_error("Failed to read IDs from: " + path);
    }

    std::fclose(f);
    return ids;
}

// ---------------------------------------------------------------------------
// Helper: compute max absolute error between two tensors
// ---------------------------------------------------------------------------

static float max_abs_error(const Tensor& a, const Tensor& b) {
    EXPECT_EQ(a.numel(), b.numel());
    float max_err = 0.0f;
    for (int i = 0; i < a.numel(); i++) {
        float err = std::fabs(a.data()[i] - b.data()[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static bool file_exists(const std::string& path) {
    std::ifstream ifs(path);
    return ifs.good();
}

// ---------------------------------------------------------------------------
// Test data directory — set via TEST_DATA_DIR env var or default path
// ---------------------------------------------------------------------------

static std::string get_test_data_dir() {
    const char* env = std::getenv("GRASSLM_TEST_DATA_DIR");
    if (env) return std::string(env);
    return "test_data";
}

// Tolerance for float32 numerical parity
static constexpr float TOLERANCE = 1e-4f;

// ---------------------------------------------------------------------------
// Test fixture: loads model and reference data once for all tests
// ---------------------------------------------------------------------------

class NumericalParityTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        data_dir_ = get_test_data_dir();

        std::string model_path = data_dir_ + "/model.grasslm";
        if (!file_exists(model_path)) {
            skip_ = true;
            return;
        }

        // Load model
        if (!loader_.load(model_path)) {
            skip_ = true;
            return;
        }
        if (!model_.load(loader_)) {
            skip_ = true;
            return;
        }

        // Load input token IDs
        token_ids_ = read_token_ids(data_dir_ + "/input_ids.bin");

        // Run C++ forward pass with debug info
        result_ = model_.forward_debug(token_ids_);

        skip_ = false;
    }

    void SetUp() override {
        if (skip_) {
            GTEST_SKIP() << "Test data not found at: " << data_dir_
                         << ". Run python/export_test_data.py first.";
        }
    }

    static std::string data_dir_;
    static bool skip_;
    static WeightLoader loader_;
    static GrassLMModel model_;
    static std::vector<int> token_ids_;
    static ForwardDebugResult result_;
};

std::string NumericalParityTest::data_dir_;
bool NumericalParityTest::skip_ = true;
WeightLoader NumericalParityTest::loader_;
GrassLMModel NumericalParityTest::model_;
std::vector<int> NumericalParityTest::token_ids_;
ForwardDebugResult NumericalParityTest::result_;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(NumericalParityTest, EmbeddingParity) {
    std::string path = data_dir_ + "/embed_output.bin";
    ASSERT_TRUE(file_exists(path)) << "Missing: " << path;

    Tensor py_embed = read_tensor(path);
    const Tensor& cpp_embed = result_.embed_output;

    // Shape check
    ASSERT_EQ(py_embed.ndim(), cpp_embed.ndim());
    for (int d = 0; d < py_embed.ndim(); d++) {
        EXPECT_EQ(py_embed.size(d), cpp_embed.size(d));
    }

    float err = max_abs_error(py_embed, cpp_embed);
    EXPECT_LT(err, TOLERANCE)
        << "Embedding max abs error: " << err;
}

TEST_F(NumericalParityTest, BlockOutputParity) {
    int n_layers = static_cast<int>(loader_.config().n_layers);

    for (int i = 0; i < n_layers; i++) {
        std::string path = data_dir_ + "/block_" + std::to_string(i) + "_output.bin";
        ASSERT_TRUE(file_exists(path)) << "Missing: " << path;

        Tensor py_block = read_tensor(path);
        const Tensor& cpp_block = result_.block_outputs[i];

        // Shape check
        ASSERT_EQ(py_block.ndim(), cpp_block.ndim())
            << "Block " << i << " ndim mismatch";
        for (int d = 0; d < py_block.ndim(); d++) {
            EXPECT_EQ(py_block.size(d), cpp_block.size(d))
                << "Block " << i << " dim " << d << " mismatch";
        }

        float err = max_abs_error(py_block, cpp_block);
        EXPECT_LT(err, TOLERANCE)
            << "Block " << i << " max abs error: " << err;
    }
}

TEST_F(NumericalParityTest, FinalNormParity) {
    std::string path = data_dir_ + "/final_norm_output.bin";
    ASSERT_TRUE(file_exists(path)) << "Missing: " << path;

    Tensor py_norm = read_tensor(path);
    const Tensor& cpp_norm = result_.final_norm_output;

    ASSERT_EQ(py_norm.ndim(), cpp_norm.ndim());
    for (int d = 0; d < py_norm.ndim(); d++) {
        EXPECT_EQ(py_norm.size(d), cpp_norm.size(d));
    }

    float err = max_abs_error(py_norm, cpp_norm);
    EXPECT_LT(err, TOLERANCE)
        << "Final layernorm max abs error: " << err;
}

TEST_F(NumericalParityTest, LogitsParity) {
    std::string path = data_dir_ + "/logits.bin";
    ASSERT_TRUE(file_exists(path)) << "Missing: " << path;

    Tensor py_logits = read_tensor(path);
    const Tensor& cpp_logits = result_.logits;

    ASSERT_EQ(py_logits.ndim(), cpp_logits.ndim());
    for (int d = 0; d < py_logits.ndim(); d++) {
        EXPECT_EQ(py_logits.size(d), cpp_logits.size(d));
    }

    float err = max_abs_error(py_logits, cpp_logits);
    EXPECT_LT(err, TOLERANCE)
        << "Logits max abs error: " << err;
}

TEST_F(NumericalParityTest, ArgmaxAgreement) {
    // Verify that the argmax token prediction agrees between Python and C++
    std::string path = data_dir_ + "/logits.bin";
    ASSERT_TRUE(file_exists(path)) << "Missing: " << path;

    Tensor py_logits = read_tensor(path);
    const Tensor& cpp_logits = result_.logits;

    int L = py_logits.size(0);
    int V = py_logits.size(1);

    for (int t = 0; t < L; t++) {
        // Find argmax in Python logits
        int py_argmax = 0;
        float py_max = py_logits(t, 0);
        for (int v = 1; v < V; v++) {
            if (py_logits(t, v) > py_max) {
                py_max = py_logits(t, v);
                py_argmax = v;
            }
        }

        // Find argmax in C++ logits
        int cpp_argmax = 0;
        float cpp_max = cpp_logits(t, 0);
        for (int v = 1; v < V; v++) {
            if (cpp_logits(t, v) > cpp_max) {
                cpp_max = cpp_logits(t, v);
                cpp_argmax = v;
            }
        }

        EXPECT_EQ(py_argmax, cpp_argmax)
            << "Argmax mismatch at position " << t
            << ": Python=" << py_argmax << " C++=" << cpp_argmax;
    }
}

TEST_F(NumericalParityTest, ErrorAccumulation) {
    // Track how error grows through layers to detect drift
    int n_layers = static_cast<int>(loader_.config().n_layers);

    float prev_err = 0.0f;

    // Embedding error
    {
        std::string path = data_dir_ + "/embed_output.bin";
        if (file_exists(path)) {
            Tensor py = read_tensor(path);
            prev_err = max_abs_error(py, result_.embed_output);
        }
    }

    // Per-layer error (should not grow unreasonably)
    for (int i = 0; i < n_layers; i++) {
        std::string path = data_dir_ + "/block_" + std::to_string(i) + "_output.bin";
        if (!file_exists(path)) continue;

        Tensor py = read_tensor(path);
        float err = max_abs_error(py, result_.block_outputs[i]);

        // Error should stay within tolerance even if it grows slightly per layer
        EXPECT_LT(err, TOLERANCE)
            << "Layer " << i << " error " << err
            << " exceeds tolerance (prev layer error: " << prev_err << ")";

        prev_err = err;
    }
}
