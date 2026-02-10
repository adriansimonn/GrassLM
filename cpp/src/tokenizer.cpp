#include <grasslm/tokenizer.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace grasslm {

bool Tokenizer::load(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) return false;

    token_to_id_.clear();
    id_to_token_.clear();

    std::string line;
    int id = 0;
    while (std::getline(file, line)) {
        // Strip trailing \r (Windows line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        id_to_token_.push_back(line);
        token_to_id_[line] = id;
        id++;
    }

    // Look up special token IDs
    auto find_special = [&](const std::string& token) -> int {
        auto it = token_to_id_.find(token);
        return (it != token_to_id_.end()) ? it->second : 0;
    };

    cls_id_ = find_special("[CLS]");
    sep_id_ = find_special("[SEP]");
    unk_id_ = find_special("[UNK]");
    pad_id_ = find_special("[PAD]");

    return !id_to_token_.empty();
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::string normalized = normalize(text);
    std::vector<std::string> words = split_whitespace(normalized);

    std::vector<int> token_ids;
    for (const auto& word : words) {
        std::vector<std::string> subwords = wordpiece_tokenize(word);
        for (const auto& sw : subwords) {
            auto it = token_to_id_.find(sw);
            if (it != token_to_id_.end()) {
                token_ids.push_back(it->second);
            } else {
                token_ids.push_back(unk_id_);
            }
        }
    }

    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (size_t i = 0; i < token_ids.size(); i++) {
        int id = token_ids[i];
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;

        const std::string& token = id_to_token_[id];

        // Skip special tokens
        if (token == "[CLS]" || token == "[SEP]" || token == "[PAD]" ||
            token == "[UNK]" || token == "[MASK]") {
            continue;
        }

        if (token.size() >= 2 && token[0] == '#' && token[1] == '#') {
            // Continuation subword: append without space, strip ## prefix
            result += token.substr(2);
        } else {
            // Regular token: add space separator (except for first token)
            if (!result.empty()) {
                result += ' ';
            }
            result += token;
        }
    }

    return result;
}

std::string Tokenizer::normalize(const std::string& text) const {
    std::string result;
    result.reserve(text.size());

    const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
    size_t len = text.size();
    size_t i = 0;

    while (i < len) {
        unsigned char c = bytes[i];

        if (c < 0x80) {
            // ASCII: lowercase
            result += static_cast<char>(std::tolower(c));
            i++;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < len) {
            // 2-byte UTF-8 sequence
            unsigned char c2 = bytes[i + 1];
            uint32_t codepoint = ((c & 0x1F) << 6) | (c2 & 0x3F);

            // Strip combining diacritical marks (U+0300 - U+036F)
            if (codepoint >= 0x0300 && codepoint <= 0x036F) {
                i += 2;
                continue;
            }

            // Decompose common accented Latin characters to their base form
            // This covers the most common accented characters in Latin-1 Supplement
            char base = 0;
            if (codepoint >= 0x00C0 && codepoint <= 0x00C5) base = 'a'; // À-Å
            else if (codepoint == 0x00C7) base = 'c';                   // Ç
            else if (codepoint >= 0x00C8 && codepoint <= 0x00CB) base = 'e'; // È-Ë
            else if (codepoint >= 0x00CC && codepoint <= 0x00CF) base = 'i'; // Ì-Ï
            else if (codepoint == 0x00D1) base = 'n';                   // Ñ
            else if (codepoint >= 0x00D2 && codepoint <= 0x00D6) base = 'o'; // Ò-Ö
            else if (codepoint >= 0x00D9 && codepoint <= 0x00DC) base = 'u'; // Ù-Ü
            else if (codepoint == 0x00DD) base = 'y';                   // Ý
            else if (codepoint >= 0x00E0 && codepoint <= 0x00E5) base = 'a'; // à-å
            else if (codepoint == 0x00E7) base = 'c';                   // ç
            else if (codepoint >= 0x00E8 && codepoint <= 0x00EB) base = 'e'; // è-ë
            else if (codepoint >= 0x00EC && codepoint <= 0x00EF) base = 'i'; // ì-ï
            else if (codepoint == 0x00F1) base = 'n';                   // ñ
            else if (codepoint >= 0x00F2 && codepoint <= 0x00F6) base = 'o'; // ò-ö
            else if (codepoint >= 0x00F9 && codepoint <= 0x00FC) base = 'u'; // ù-ü
            else if (codepoint == 0x00FD || codepoint == 0x00FF) base = 'y'; // ý,ÿ

            if (base != 0) {
                result += base;
            } else {
                // Keep the original bytes (already lowercase if applicable)
                result += static_cast<char>(c);
                result += static_cast<char>(c2);
            }
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < len) {
            // 3-byte UTF-8: pass through
            result += static_cast<char>(c);
            result += static_cast<char>(bytes[i + 1]);
            result += static_cast<char>(bytes[i + 2]);
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < len) {
            // 4-byte UTF-8: pass through
            result += static_cast<char>(c);
            result += static_cast<char>(bytes[i + 1]);
            result += static_cast<char>(bytes[i + 2]);
            result += static_cast<char>(bytes[i + 3]);
            i += 4;
        } else {
            // Invalid byte: skip
            i++;
        }
    }

    return result;
}

std::vector<std::string> Tokenizer::split_whitespace(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;

    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }

    return tokens;
}

std::vector<std::string> Tokenizer::wordpiece_tokenize(const std::string& word) const {
    // BERT-style WordPiece: greedy longest-match from left to right.
    // First subword is looked up as-is; subsequent subwords are prefixed with "##".
    std::vector<std::string> result;

    // Split word into individual characters (UTF-8 aware)
    std::vector<std::string> chars;
    const auto* bytes = reinterpret_cast<const unsigned char*>(word.data());
    size_t len = word.size();
    size_t i = 0;
    while (i < len) {
        unsigned char c = bytes[i];
        size_t char_len = 1;
        if ((c & 0x80) == 0)        char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        if (i + char_len > len) char_len = 1; // safety
        chars.push_back(word.substr(i, char_len));
        i += char_len;
    }

    if (chars.empty()) return result;

    size_t start = 0;
    while (start < chars.size()) {
        size_t end = chars.size();
        bool found = false;

        while (end > start) {
            // Build the candidate substring
            std::string candidate;
            for (size_t k = start; k < end; k++) {
                candidate += chars[k];
            }

            // Add ## prefix for continuation subwords
            if (start > 0) {
                candidate = "##" + candidate;
            }

            if (token_to_id_.count(candidate)) {
                result.push_back(candidate);
                found = true;
                start = end;
                break;
            }

            end--;
        }

        if (!found) {
            // Character not in vocab — emit [UNK] for the whole word
            result.clear();
            result.push_back("[UNK]");
            return result;
        }
    }

    return result;
}

}  // namespace grasslm
