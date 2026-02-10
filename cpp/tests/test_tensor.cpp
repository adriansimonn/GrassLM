#include <grasslm/tensor.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace grasslm;

TEST(TensorTest, DefaultConstruction) {
    Tensor t;
    EXPECT_EQ(t.ndim(), 0);
    EXPECT_EQ(t.numel(), 0);
}

TEST(TensorTest, ShapeConstruction) {
    Tensor t({3, 4});
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.size(0), 3);
    EXPECT_EQ(t.size(1), 4);
    EXPECT_EQ(t.numel(), 12);
}

TEST(TensorTest, DataConstruction) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t({2, 3}, data);
    EXPECT_FLOAT_EQ(t(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(t(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(t(1, 2), 6.0f);
}

TEST(TensorTest, FillAndZeros) {
    Tensor t({2, 3});
    t.fill(5.0f);
    for (int i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(t(i), 5.0f);
    }
    t.zeros();
    for (int i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(t(i), 0.0f);
    }
}

TEST(TensorTest, Gemv) {
    // A = [[1, 2], [3, 4]], x = [1, 1] -> y = [3, 7]
    Tensor A({2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor x({2}, std::vector<float>{1, 1});
    Tensor y = A.gemv(x);
    EXPECT_EQ(y.ndim(), 1);
    EXPECT_EQ(y.size(0), 2);
    EXPECT_FLOAT_EQ(y(0), 3.0f);
    EXPECT_FLOAT_EQ(y(1), 7.0f);
}

TEST(TensorTest, Gemm) {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = [[19, 22], [43, 50]]
    Tensor A({2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor B({2, 2}, std::vector<float>{5, 6, 7, 8});
    Tensor C = A.gemm(B);
    EXPECT_FLOAT_EQ(C(0, 0), 19.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 22.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 43.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 50.0f);
}

TEST(TensorTest, ElementWiseAdd) {
    Tensor a({3}, std::vector<float>{1, 2, 3});
    Tensor b({3}, std::vector<float>{4, 5, 6});
    Tensor c = a + b;
    EXPECT_FLOAT_EQ(c(0), 5.0f);
    EXPECT_FLOAT_EQ(c(1), 7.0f);
    EXPECT_FLOAT_EQ(c(2), 9.0f);
}

TEST(TensorTest, ElementWiseMultiply) {
    Tensor a({3}, std::vector<float>{1, 2, 3});
    Tensor b({3}, std::vector<float>{4, 5, 6});
    Tensor c = a * b;
    EXPECT_FLOAT_EQ(c(0), 4.0f);
    EXPECT_FLOAT_EQ(c(1), 10.0f);
    EXPECT_FLOAT_EQ(c(2), 18.0f);
}

TEST(TensorTest, ScalarMultiply) {
    Tensor a({3}, std::vector<float>{1, 2, 3});
    Tensor b = a * 2.0f;
    EXPECT_FLOAT_EQ(b(0), 2.0f);
    EXPECT_FLOAT_EQ(b(1), 4.0f);
    EXPECT_FLOAT_EQ(b(2), 6.0f);
}

TEST(TensorTest, Sigmoid) {
    Tensor a({3}, std::vector<float>{0.0f, 1.0f, -1.0f});
    Tensor b = a.sigmoid();
    EXPECT_NEAR(b(0), 0.5f, 1e-6f);
    EXPECT_NEAR(b(1), 1.0f / (1.0f + std::exp(-1.0f)), 1e-6f);
    EXPECT_NEAR(b(2), 1.0f / (1.0f + std::exp(1.0f)), 1e-6f);
}

TEST(TensorTest, Gelu) {
    Tensor a({1}, std::vector<float>{0.0f});
    Tensor b = a.gelu();
    EXPECT_NEAR(b(0), 0.0f, 1e-6f);
}

TEST(TensorTest, Clamp) {
    Tensor a({5}, std::vector<float>{-2, -1, 0, 1, 2});
    Tensor b = a.clamp(-1.0f, 1.0f);
    EXPECT_FLOAT_EQ(b(0), -1.0f);
    EXPECT_FLOAT_EQ(b(1), -1.0f);
    EXPECT_FLOAT_EQ(b(2), 0.0f);
    EXPECT_FLOAT_EQ(b(3), 1.0f);
    EXPECT_FLOAT_EQ(b(4), 1.0f);
}

TEST(TensorTest, Reshape) {
    Tensor a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b = a.reshape({3, 2});
    EXPECT_EQ(b.size(0), 3);
    EXPECT_EQ(b.size(1), 2);
    EXPECT_FLOAT_EQ(b(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(b(2, 1), 6.0f);
}

TEST(TensorTest, Axpy) {
    Tensor y({3}, std::vector<float>{1, 2, 3});
    Tensor x({3}, std::vector<float>{4, 5, 6});
    y.axpy(2.0f, x);  // y = 2*x + y
    EXPECT_FLOAT_EQ(y(0), 9.0f);
    EXPECT_FLOAT_EQ(y(1), 12.0f);
    EXPECT_FLOAT_EQ(y(2), 15.0f);
}

TEST(TensorTest, CopySemantics) {
    Tensor a({2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor b = a;
    b(0, 0) = 99.0f;
    EXPECT_FLOAT_EQ(a(0, 0), 1.0f);  // a unchanged
    EXPECT_FLOAT_EQ(b(0, 0), 99.0f);
}

TEST(TensorTest, ToString) {
    Tensor t({2, 3});
    std::string s = t.to_string();
    EXPECT_NE(s.find("2"), std::string::npos);
    EXPECT_NE(s.find("3"), std::string::npos);
}

// --- Tests for newly implemented methods ---

TEST(TensorTest, LayerNorm) {
    // 2D tensor: 2 rows of 3 values
    Tensor x({2, 3}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor gamma({3}, std::vector<float>{1.0f, 1.0f, 1.0f});
    Tensor beta({3}, std::vector<float>{0.0f, 0.0f, 0.0f});

    Tensor result = x.layernorm(gamma, beta);
    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 3);

    // Row 0: mean=2, var=2/3, std=sqrt(2/3+eps)
    // Normalized: (-1, 0, 1) / sqrt(2/3) ≈ (-1.2247, 0, 1.2247)
    EXPECT_NEAR(result(0, 0), -1.2247f, 1e-3f);
    EXPECT_NEAR(result(0, 1), 0.0f, 1e-3f);
    EXPECT_NEAR(result(0, 2), 1.2247f, 1e-3f);

    // Verify each row sums to ~0 (with gamma=1, beta=0)
    float sum0 = result(0, 0) + result(0, 1) + result(0, 2);
    float sum1 = result(1, 0) + result(1, 1) + result(1, 2);
    EXPECT_NEAR(sum0, 0.0f, 1e-5f);
    EXPECT_NEAR(sum1, 0.0f, 1e-5f);
}

TEST(TensorTest, LayerNormWithScaleShift) {
    Tensor x({1, 3}, std::vector<float>{1.0f, 2.0f, 3.0f});
    Tensor gamma({3}, std::vector<float>{2.0f, 2.0f, 2.0f});
    Tensor beta({3}, std::vector<float>{1.0f, 1.0f, 1.0f});

    Tensor result = x.layernorm(gamma, beta);
    // gamma * normalized + beta = 2 * normalized + 1
    // Row mean should be ~1.0 (beta)
    float mean = (result(0, 0) + result(0, 1) + result(0, 2)) / 3.0f;
    EXPECT_NEAR(mean, 1.0f, 1e-4f);
}

TEST(TensorTest, Softmax1D) {
    Tensor a({3}, std::vector<float>{1.0f, 2.0f, 3.0f});
    Tensor b = a.softmax(0);

    // All values should be positive and sum to 1
    float sum = b(0) + b(1) + b(2);
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    EXPECT_GT(b(2), b(1));
    EXPECT_GT(b(1), b(0));
}

TEST(TensorTest, Softmax2D) {
    Tensor a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b = a.softmax(-1);  // along last dim

    // Each row should sum to 1
    float sum0 = b(0, 0) + b(0, 1) + b(0, 2);
    float sum1 = b(1, 0) + b(1, 1) + b(1, 2);
    EXPECT_NEAR(sum0, 1.0f, 1e-6f);
    EXPECT_NEAR(sum1, 1.0f, 1e-6f);
}

TEST(TensorTest, SliceDim0) {
    Tensor a({4, 3}, std::vector<float>{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    });
    Tensor b = a.slice(0, 1, 3);  // rows 1..2
    EXPECT_EQ(b.size(0), 2);
    EXPECT_EQ(b.size(1), 3);
    EXPECT_FLOAT_EQ(b(0, 0), 4.0f);
    EXPECT_FLOAT_EQ(b(1, 2), 9.0f);
}

TEST(TensorTest, SliceDim1) {
    Tensor a({2, 4}, std::vector<float>{
        1, 2, 3, 4,
        5, 6, 7, 8
    });
    Tensor b = a.slice(1, 1, 3);  // cols 1..2
    EXPECT_EQ(b.size(0), 2);
    EXPECT_EQ(b.size(1), 2);
    EXPECT_FLOAT_EQ(b(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(b(0, 1), 3.0f);
    EXPECT_FLOAT_EQ(b(1, 0), 6.0f);
    EXPECT_FLOAT_EQ(b(1, 1), 7.0f);
}

TEST(TensorTest, Slice1D) {
    Tensor a({5}, std::vector<float>{10, 20, 30, 40, 50});
    Tensor b = a.slice(0, 1, 4);
    EXPECT_EQ(b.size(0), 3);
    EXPECT_FLOAT_EQ(b(0), 20.0f);
    EXPECT_FLOAT_EQ(b(2), 40.0f);
}

TEST(TensorTest, CatDim0) {
    Tensor a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b({1, 3}, std::vector<float>{7, 8, 9});
    Tensor c = Tensor::cat({a, b}, 0);
    EXPECT_EQ(c.size(0), 3);
    EXPECT_EQ(c.size(1), 3);
    EXPECT_FLOAT_EQ(c(2, 0), 7.0f);
}

TEST(TensorTest, CatDim1) {
    Tensor a({2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor b({2, 3}, std::vector<float>{5, 6, 7, 8, 9, 10});
    Tensor c = Tensor::cat({a, b}, 1);
    EXPECT_EQ(c.size(0), 2);
    EXPECT_EQ(c.size(1), 5);
    EXPECT_FLOAT_EQ(c(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(c(0, 2), 5.0f);
    EXPECT_FLOAT_EQ(c(1, 4), 10.0f);
}

TEST(TensorTest, Transpose) {
    Tensor a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b = a.transpose(0, 1);
    EXPECT_EQ(b.size(0), 3);
    EXPECT_EQ(b.size(1), 2);
    EXPECT_FLOAT_EQ(b(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(b(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(b(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(b(2, 1), 6.0f);
}

TEST(TensorTest, Norm2D) {
    Tensor a({2, 3}, std::vector<float>{3, 4, 0, 0, 5, 12});
    Tensor n = a.norm(-1);
    EXPECT_EQ(n.size(0), 2);
    EXPECT_EQ(n.size(1), 1);
    EXPECT_NEAR(n(0, 0), 5.0f, 1e-5f);    // sqrt(9+16)
    EXPECT_NEAR(n(1, 0), 13.0f, 1e-5f);   // sqrt(25+144)
}

TEST(TensorTest, Norm1D) {
    Tensor a({3}, std::vector<float>{3, 4, 0});
    Tensor n = a.norm(0);
    EXPECT_NEAR(n(0), 5.0f, 1e-5f);
}

TEST(TensorTest, LinearWithBias) {
    // x: (2, 3), W: (4, 3), b: (4,) -> result: (2, 4)
    // result = x @ W^T + b
    Tensor x({2, 3}, std::vector<float>{1, 0, 0, 0, 1, 0});
    Tensor W({4, 3}, std::vector<float>{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 1
    });
    Tensor b({4}, std::vector<float>{10, 20, 30, 40});

    Tensor result = x.linear(W, b);
    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 4);

    // Row 0: [1,0,0] @ W^T = [1,0,0,1] + [10,20,30,40] = [11,20,30,41]
    EXPECT_FLOAT_EQ(result(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 20.0f);
    EXPECT_FLOAT_EQ(result(0, 2), 30.0f);
    EXPECT_FLOAT_EQ(result(0, 3), 41.0f);

    // Row 1: [0,1,0] @ W^T = [0,1,0,1] + [10,20,30,40] = [10,21,30,41]
    EXPECT_FLOAT_EQ(result(1, 0), 10.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 21.0f);
    EXPECT_FLOAT_EQ(result(1, 2), 30.0f);
    EXPECT_FLOAT_EQ(result(1, 3), 41.0f);
}

TEST(TensorTest, LinearNoBias) {
    // x: (2, 3), W: (2, 3) -> result = x @ W^T, shape (2, 2)
    Tensor x({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor W({2, 3}, std::vector<float>{1, 0, 0, 0, 1, 0});

    Tensor result = x.linear(W);
    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 2);

    // Row 0: [1,2,3] @ [[1,0],[0,1],[0,0]] = [1, 2]
    EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 2.0f);
    // Row 1: [4,5,6] @ [[1,0],[0,1],[0,0]] = [4, 5]
    EXPECT_FLOAT_EQ(result(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 5.0f);
}

TEST(TensorTest, Linear1D) {
    // x: (3,), W: (2, 3), b: (2,) -> result: (2,)
    Tensor x({3}, std::vector<float>{1, 2, 3});
    Tensor W({2, 3}, std::vector<float>{1, 0, 0, 0, 1, 0});
    Tensor b({2}, std::vector<float>{10, 20});

    Tensor result = x.linear(W, b);
    EXPECT_EQ(result.size(0), 2);
    EXPECT_FLOAT_EQ(result(0), 11.0f);  // 1*1+0*2+0*3 + 10
    EXPECT_FLOAT_EQ(result(1), 22.0f);  // 0*1+1*2+0*3 + 20
}
