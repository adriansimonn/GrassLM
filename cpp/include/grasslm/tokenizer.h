#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace grasslm {

/// WordPiece tokenizer compatible with BERT's vocab.txt format.
class Tokenizer {
public:
    Tokenizer() = default;

    /// Load vocabulary from a vocab.txt file (one token per line).
    bool load(const std::string& vocab_path);

    /// Encode text to token IDs.
    std::vector<int> encode(const std::string& text) const;

    /// Decode token IDs back to text.
    std::string decode(const std::vector<int>& token_ids) const;

    /// Special token IDs.
    int cls_id() const { return cls_id_; }
    int sep_id() const { return sep_id_; }
    int unk_id() const { return unk_id_; }
    int pad_id() const { return pad_id_; }

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

    /// Get the raw token string for a given ID (empty if out of range).
    const std::string& id_to_token(int id) const {
        static const std::string empty;
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return empty;
        return id_to_token_[id];
    }

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;

    int cls_id_ = 0;
    int sep_id_ = 0;
    int unk_id_ = 0;
    int pad_id_ = 0;

    /// Lowercase and strip accents from input text.
    std::string normalize(const std::string& text) const;

    /// Split text on whitespace.
    std::vector<std::string> split_whitespace(const std::string& text) const;

    /// WordPiece greedy longest-match segmentation of a single word.
    std::vector<std::string> wordpiece_tokenize(const std::string& word) const;
};

}  // namespace grasslm
