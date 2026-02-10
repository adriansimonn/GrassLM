#include <grasslm/tensor.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace grasslm {

// --- Construction ---

Tensor::Tensor() : shape_(), numel_(0), data_() {}

Tensor::Tensor(std::vector<int> shape) : shape_(std::move(shape)) {
    compute_numel();
    data_.resize(numel_, 0.0f);
}

Tensor::Tensor(std::vector<int> shape, const float* data) : shape_(std::move(shape)) {
    compute_numel();
    data_.assign(data, data + numel_);
}

Tensor::Tensor(std::vector<int> shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
    compute_numel();
    if (static_cast<int>(data_.size()) != numel_) {
        throw std::invalid_argument("Data size does not match shape");
    }
}

Tensor::Tensor(const Tensor& other) = default;
Tensor::Tensor(Tensor&& other) noexcept = default;
Tensor& Tensor::operator=(const Tensor& other) = default;
Tensor& Tensor::operator=(Tensor&& other) noexcept = default;

// --- Accessors ---

int Tensor::size(int dim) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

float& Tensor::operator()(int i) {
    return data_[i];
}

float Tensor::operator()(int i) const {
    return data_[i];
}

float& Tensor::operator()(int i, int j) {
    return data_[flat_index(i, j)];
}

float Tensor::operator()(int i, int j) const {
    return data_[flat_index(i, j)];
}

float& Tensor::operator()(int i, int j, int k) {
    return data_[flat_index(i, j, k)];
}

float Tensor::operator()(int i, int j, int k) const {
    return data_[flat_index(i, j, k)];
}

// --- BLAS operations (via Apple Accelerate) ---

Tensor Tensor::gemv(const Tensor& x) const {
    // this: (M, N), x: (N,) -> result: (M,)
    if (ndim() != 2 || x.ndim() != 1) {
        throw std::invalid_argument("gemv requires 2D matrix and 1D vector");
    }
    int M = shape_[0];
    int N = shape_[1];
    if (x.size(0) != N) {
        throw std::invalid_argument("gemv dimension mismatch");
    }

    Tensor result({M});
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N,
                1.0f, data(), N,
                x.data(), 1,
                0.0f, result.data(), 1);
    return result;
}

Tensor Tensor::gemm(const Tensor& B) const {
    // this: (M, K), B: (K, N) -> result: (M, N)
    if (ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("gemm requires 2D matrices");
    }
    int M = shape_[0];
    int K = shape_[1];
    int N = B.shape()[1];
    if (B.shape()[0] != K) {
        throw std::invalid_argument("gemm dimension mismatch");
    }

    Tensor result({M, N});
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, data(), K,
                B.data(), N,
                0.0f, result.data(), N);
    return result;
}

void Tensor::axpy(float alpha, const Tensor& x) {
    if (numel_ != x.numel()) {
        throw std::invalid_argument("axpy size mismatch");
    }
    cblas_saxpy(numel_, alpha, x.data(), 1, data(), 1);
}

// --- Element-wise operations ---

Tensor Tensor::sigmoid() const {
    Tensor result(shape_);
    for (int i = 0; i < numel_; ++i) {
        result.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    return result;
}

Tensor Tensor::gelu() const {
    // Exact GELU using erf (matches PyTorch's nn.GELU default: approximate='none')
    Tensor result(shape_);
    constexpr float inv_sqrt2 = 0.7071067811865476f;  // 1/sqrt(2)
    for (int i = 0; i < numel_; ++i) {
        float x = data_[i];
        result.data_[i] = 0.5f * x * (1.0f + std::erf(x * inv_sqrt2));
    }
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (numel_ != other.numel_) {
        throw std::invalid_argument("Addition size mismatch");
    }
    Tensor result(shape_);
    for (int i = 0; i < numel_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (numel_ != other.numel_) {
        throw std::invalid_argument("Subtraction size mismatch");
    }
    Tensor result(shape_);
    for (int i = 0; i < numel_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (numel_ != other.numel_) {
        throw std::invalid_argument("Element-wise multiply size mismatch");
    }
    Tensor result(shape_);
    for (int i = 0; i < numel_; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < numel_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

// --- Utilities ---

Tensor Tensor::layernorm(const Tensor& gamma, const Tensor& beta, float eps) const {
    if (ndim() == 1) {
        int d = shape_[0];
        float mean = 0.0f;
        for (int j = 0; j < d; ++j) mean += data_[j];
        mean /= d;

        float var = 0.0f;
        for (int j = 0; j < d; ++j) {
            float diff = data_[j] - mean;
            var += diff * diff;
        }
        var /= d;
        float inv_std = 1.0f / std::sqrt(var + eps);

        Tensor result(shape_);
        for (int j = 0; j < d; ++j) {
            result.data_[j] = gamma.data_[j] * (data_[j] - mean) * inv_std + beta.data_[j];
        }
        return result;
    }

    if (ndim() != 2) {
        throw std::invalid_argument("layernorm requires 1D or 2D tensor");
    }

    int rows = shape_[0];
    int cols = shape_[1];
    Tensor result(shape_);

    for (int i = 0; i < rows; ++i) {
        const float* row = data_.data() + i * cols;
        float* out = result.data_.data() + i * cols;

        float mean = 0.0f;
        for (int j = 0; j < cols; ++j) mean += row[j];
        mean /= cols;

        float var = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= cols;
        float inv_std = 1.0f / std::sqrt(var + eps);

        for (int j = 0; j < cols; ++j) {
            out[j] = gamma.data_[j] * (row[j] - mean) * inv_std + beta.data_[j];
        }
    }
    return result;
}

Tensor Tensor::softmax(int dim) const {
    if (dim < 0) dim += ndim();

    if (ndim() == 1 && dim == 0) {
        Tensor result(shape_);
        float max_val = *std::max_element(data_.begin(), data_.end());
        float sum = 0.0f;
        for (int i = 0; i < numel_; ++i) {
            result.data_[i] = std::exp(data_[i] - max_val);
            sum += result.data_[i];
        }
        for (int i = 0; i < numel_; ++i) {
            result.data_[i] /= sum;
        }
        return result;
    }

    if (ndim() == 2 && dim == 1) {
        int rows = shape_[0];
        int cols = shape_[1];
        Tensor result(shape_);

        for (int i = 0; i < rows; ++i) {
            const float* row = data_.data() + i * cols;
            float* out = result.data_.data() + i * cols;

            float max_val = row[0];
            for (int j = 1; j < cols; ++j) {
                max_val = std::max(max_val, row[j]);
            }

            float sum = 0.0f;
            for (int j = 0; j < cols; ++j) {
                out[j] = std::exp(row[j] - max_val);
                sum += out[j];
            }
            for (int j = 0; j < cols; ++j) {
                out[j] /= sum;
            }
        }
        return result;
    }

    throw std::invalid_argument("softmax: unsupported tensor shape/dim combination");
}

Tensor Tensor::slice(int dim, int start, int end) const {
    if (dim < 0) dim += ndim();

    if (ndim() == 1 && dim == 0) {
        int len = end - start;
        Tensor result({len});
        std::copy(data_.begin() + start, data_.begin() + end, result.data_.begin());
        return result;
    }

    if (ndim() == 2) {
        int rows = shape_[0];
        int cols = shape_[1];

        if (dim == 0) {
            int new_rows = end - start;
            Tensor result({new_rows, cols});
            std::copy(data_.begin() + start * cols,
                      data_.begin() + end * cols,
                      result.data_.begin());
            return result;
        }
        if (dim == 1) {
            int new_cols = end - start;
            Tensor result({rows, new_cols});
            for (int i = 0; i < rows; ++i) {
                std::copy(data_.begin() + i * cols + start,
                          data_.begin() + i * cols + end,
                          result.data_.begin() + i * new_cols);
            }
            return result;
        }
    }

    throw std::invalid_argument("slice: unsupported tensor shape/dim");
}

Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) return Tensor();

    int nd = tensors[0].ndim();
    if (dim < 0) dim += nd;

    if (nd == 1 && dim == 0) {
        int total = 0;
        for (const auto& t : tensors) total += t.numel();
        Tensor result({total});
        int offset = 0;
        for (const auto& t : tensors) {
            std::copy(t.data(), t.data() + t.numel(), result.data() + offset);
            offset += t.numel();
        }
        return result;
    }

    if (nd == 2 && dim == 0) {
        int cols = tensors[0].shape()[1];
        int total_rows = 0;
        for (const auto& t : tensors) total_rows += t.shape()[0];
        Tensor result({total_rows, cols});
        int offset = 0;
        for (const auto& t : tensors) {
            int n = t.shape()[0] * cols;
            std::copy(t.data(), t.data() + n, result.data() + offset);
            offset += n;
        }
        return result;
    }

    if (nd == 2 && dim == 1) {
        int rows = tensors[0].shape()[0];
        int total_cols = 0;
        for (const auto& t : tensors) total_cols += t.shape()[1];
        Tensor result({rows, total_cols});
        for (int i = 0; i < rows; ++i) {
            int col_offset = 0;
            for (const auto& t : tensors) {
                int c = t.shape()[1];
                std::copy(t.data() + i * c,
                          t.data() + i * c + c,
                          result.data() + i * total_cols + col_offset);
                col_offset += c;
            }
        }
        return result;
    }

    throw std::invalid_argument("cat: unsupported configuration");
}

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    Tensor result(std::move(new_shape), data_);
    return result;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (ndim() != 2) {
        throw std::invalid_argument("transpose: only 2D tensors supported");
    }
    if (dim0 < 0) dim0 += 2;
    if (dim1 < 0) dim1 += 2;
    if (dim0 == dim1) return Tensor(shape_, data_);

    int rows = shape_[0];
    int cols = shape_[1];
    Tensor result({cols, rows});
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data_[j * rows + i] = data_[i * cols + j];
        }
    }
    return result;
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::zeros() {
    fill(0.0f);
}

Tensor Tensor::norm(int dim) const {
    if (dim < 0) dim += ndim();

    if (ndim() == 1 && dim == 0) {
        float sum = 0.0f;
        for (int i = 0; i < numel_; ++i) {
            sum += data_[i] * data_[i];
        }
        Tensor result({1});
        result.data_[0] = std::sqrt(sum);
        return result;
    }

    if (ndim() == 2 && dim == 1) {
        int rows = shape_[0];
        int cols = shape_[1];
        Tensor result({rows, 1});
        for (int i = 0; i < rows; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < cols; ++j) {
                float v = data_[i * cols + j];
                sum += v * v;
            }
            result.data_[i] = std::sqrt(sum);
        }
        return result;
    }

    throw std::invalid_argument("norm: unsupported tensor shape/dim combination");
}

Tensor Tensor::clamp(float min_val, float max_val) const {
    Tensor result(shape_);
    for (int i = 0; i < numel_; ++i) {
        result.data_[i] = std::min(max_val, std::max(min_val, data_[i]));
    }
    return result;
}

Eigen::Map<Eigen::MatrixXf> Tensor::as_eigen_matrix() {
    if (ndim() != 2) {
        throw std::invalid_argument("as_eigen_matrix requires 2D tensor");
    }
    return Eigen::Map<Eigen::MatrixXf>(data(), shape_[0], shape_[1]);
}

Eigen::Map<const Eigen::MatrixXf> Tensor::as_eigen_matrix() const {
    if (ndim() != 2) {
        throw std::invalid_argument("as_eigen_matrix requires 2D tensor");
    }
    return Eigen::Map<const Eigen::MatrixXf>(data(), shape_[0], shape_[1]);
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (int i = 0; i < ndim(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], numel=" << numel_ << ")";
    return oss.str();
}

// --- Linear ---

Tensor Tensor::linear(const Tensor& weight, const Tensor& bias) const {
    if (ndim() == 2) {
        // this: (M, K), weight: (N, K), bias: (N,) -> result: (M, N)
        int M = shape_[0];
        int K = shape_[1];
        int N = weight.shape()[0];

        Tensor result({M, N});
        // C = A * B^T  where A=this, B=weight
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K,
                    1.0f, data(), K,
                    weight.data(), K,
                    0.0f, result.data(), N);

        // Add bias to each row
        for (int i = 0; i < M; ++i) {
            cblas_saxpy(N, 1.0f, bias.data(), 1, result.data() + i * N, 1);
        }
        return result;
    }

    if (ndim() == 1) {
        // this: (K,), weight: (N, K), bias: (N,) -> result: (N,)
        int K = shape_[0];
        int N = weight.shape()[0];

        Tensor result({N});
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K,
                    1.0f, weight.data(), K,
                    data(), 1,
                    0.0f, result.data(), 1);
        cblas_saxpy(N, 1.0f, bias.data(), 1, result.data(), 1);
        return result;
    }

    throw std::invalid_argument("linear: requires 1D or 2D input");
}

Tensor Tensor::linear(const Tensor& weight) const {
    if (ndim() == 2) {
        int M = shape_[0];
        int K = shape_[1];
        int N = weight.shape()[0];

        Tensor result({M, N});
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K,
                    1.0f, data(), K,
                    weight.data(), K,
                    0.0f, result.data(), N);
        return result;
    }

    if (ndim() == 1) {
        int K = shape_[0];
        int N = weight.shape()[0];

        Tensor result({N});
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K,
                    1.0f, weight.data(), K,
                    data(), 1,
                    0.0f, result.data(), 1);
        return result;
    }

    throw std::invalid_argument("linear: requires 1D or 2D input");
}

// --- Private ---

void Tensor::compute_numel() {
    if (shape_.empty()) {
        numel_ = 0;
    } else {
        numel_ = 1;
        for (int s : shape_) {
            numel_ *= s;
        }
    }
}

int Tensor::flat_index(int i, int j) const {
    return i * shape_[1] + j;
}

int Tensor::flat_index(int i, int j, int k) const {
    return (i * shape_[1] + j) * shape_[2] + k;
}

}  // namespace grasslm
