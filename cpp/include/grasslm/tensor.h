#pragma once

#include <Accelerate/Accelerate.h>
#include <Eigen/Core>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace grasslm {

/// Row-major float tensor with BLAS-accelerated operations via Apple Accelerate.
class Tensor {
public:
    // --- Construction ---
    Tensor();
    Tensor(std::vector<int> shape);
    Tensor(std::vector<int> shape, const float* data);
    Tensor(std::vector<int> shape, std::vector<float> data);

    // Copy and move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // --- Accessors ---
    const std::vector<int>& shape() const { return shape_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int size(int dim) const;
    int numel() const { return numel_; }
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    float& operator()(int i);
    float operator()(int i) const;
    float& operator()(int i, int j);
    float operator()(int i, int j) const;
    float& operator()(int i, int j, int k);
    float operator()(int i, int j, int k) const;

    // --- BLAS operations (via Accelerate) ---

    /// Matrix-vector multiply: y = A * x  (this is A, x is input)
    Tensor gemv(const Tensor& x) const;

    /// Matrix-matrix multiply: C = A * B  (this is A, B is input)
    Tensor gemm(const Tensor& B) const;

    /// y = alpha * x + y  (in-place on this)
    void axpy(float alpha, const Tensor& x);

    // --- Element-wise operations ---
    Tensor sigmoid() const;
    Tensor gelu() const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // element-wise
    Tensor operator*(float scalar) const;

    // --- Utilities ---
    Tensor layernorm(const Tensor& gamma, const Tensor& beta, float eps = 1e-5f) const;
    Tensor softmax(int dim = -1) const;
    Tensor slice(int dim, int start, int end) const;
    static Tensor cat(const std::vector<Tensor>& tensors, int dim);
    Tensor reshape(std::vector<int> new_shape) const;
    Tensor transpose(int dim0, int dim1) const;

    /// Linear layer: result = this @ weight^T + bias
    /// this: (M, K) or (K,), weight: (N, K), bias: (N,)
    /// Returns: (M, N) or (N,)
    Tensor linear(const Tensor& weight, const Tensor& bias) const;

    /// Linear layer without bias: result = this @ weight^T
    Tensor linear(const Tensor& weight) const;

    /// Fill all elements with a value
    void fill(float value);

    /// Fill with zeros
    void zeros();

    /// L2 norm along the last dimension, returns tensor with last dim = 1
    Tensor norm(int dim = -1) const;

    /// Clamp all elements to [min_val, max_val]
    Tensor clamp(float min_val, float max_val) const;

    /// Map Eigen matrix view (2D tensors only)
    Eigen::Map<Eigen::MatrixXf> as_eigen_matrix();
    Eigen::Map<const Eigen::MatrixXf> as_eigen_matrix() const;

    /// Debug string
    std::string to_string() const;

private:
    std::vector<int> shape_;
    int numel_ = 0;
    std::vector<float> data_;

    void compute_numel();
    int flat_index(int i, int j) const;
    int flat_index(int i, int j, int k) const;
};

}  // namespace grasslm
