#include <grasslm/weight_loader.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace grasslm {

// IEEE 754 half-precision to single-precision conversion
static float half_to_float(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x03FFu;

    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;  // +/- zero
        } else {
            // Subnormal: renormalize
            exponent = 1;
            while (!(mantissa & 0x0400u)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FFu;
            result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        result = sign | 0x7F800000u | (mantissa << 13);  // Inf/NaN
    } else {
        result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float f;
    std::memcpy(&f, &result, sizeof(float));
    return f;
}

bool WeightLoader::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    // Read 64-byte header
    uint8_t header[64];
    if (std::fread(header, 1, 64, f) != 64) {
        std::fclose(f);
        return false;
    }

    // Validate magic bytes "GRLM"
    if (std::memcmp(header, "GRLM", 4) != 0) {
        std::fclose(f);
        return false;
    }

    // Parse header fields (stored as little-endian uint32_t)
    // Layout: magic(4) version(4) n_layers(4) d_model(4) d_reduce(4)
    //         d_ff(4) vocab_size(4) max_seq_len(4) dtype(4) padding(28)
    const uint32_t* hp = reinterpret_cast<const uint32_t*>(header);
    // hp[0] = magic
    // hp[1] = version (read but not stored)
    config_.n_layers    = hp[2];
    config_.d_model     = hp[3];
    config_.d_reduce    = hp[4];
    config_.d_ff        = hp[5];
    config_.vocab_size  = hp[6];
    config_.max_seq_len = hp[7];
    config_.dtype       = hp[8];

    // Read weight table
    uint32_t num_weights;
    if (std::fread(&num_weights, sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    weights_.clear();
    weights_.reserve(num_weights);

    for (uint32_t w = 0; w < num_weights; ++w) {
        // Read weight name
        uint32_t name_len;
        if (std::fread(&name_len, sizeof(uint32_t), 1, f) != 1) {
            std::fclose(f);
            return false;
        }

        std::string name(name_len, '\0');
        if (std::fread(name.data(), 1, name_len, f) != name_len) {
            std::fclose(f);
            return false;
        }

        // Read shape
        uint32_t n_dims;
        if (std::fread(&n_dims, sizeof(uint32_t), 1, f) != 1) {
            std::fclose(f);
            return false;
        }

        std::vector<int> shape(n_dims);
        int numel = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint32_t dim_size;
            if (std::fread(&dim_size, sizeof(uint32_t), 1, f) != 1) {
                std::fclose(f);
                return false;
            }
            shape[d] = static_cast<int>(dim_size);
            numel *= shape[d];
        }

        // Read weight data
        std::vector<float> data(numel);
        if (config_.dtype == 0) {
            // float32: read directly
            if (std::fread(data.data(), sizeof(float), numel, f) !=
                static_cast<size_t>(numel)) {
                std::fclose(f);
                return false;
            }
        } else {
            // float16: read raw uint16 and convert to float32
            std::vector<uint16_t> raw(numel);
            if (std::fread(raw.data(), sizeof(uint16_t), numel, f) !=
                static_cast<size_t>(numel)) {
                std::fclose(f);
                return false;
            }
            for (int i = 0; i < numel; ++i) {
                data[i] = half_to_float(raw[i]);
            }
        }

        weights_[name] = Tensor(shape, std::move(data));
    }

    std::fclose(f);
    return true;
}

const Tensor& WeightLoader::get(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

}  // namespace grasslm
