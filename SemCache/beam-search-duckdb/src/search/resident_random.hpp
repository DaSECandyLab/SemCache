#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace semcache {
// Import Random(seed).getstate()[1] once; prediction is the only subsequent
// Python callback. CPython's getrandbits/_randbelow/sample decisions are kept.
class ResidentRandom {
public:
  explicit ResidentRandom(const std::vector<std::uint64_t> &state) {
    if (state.size() != 625 || state.back() > 624)
      throw std::invalid_argument("Invalid imported random state");
    for (std::size_t i = 0; i < 624; ++i) {
      if (state[i] > UINT32_MAX)
        throw std::invalid_argument("Invalid random word");
      words_[i] = static_cast<std::uint32_t>(state[i]);
    }
    position_ = state.back();
  }
  std::vector<std::size_t> sample(const std::vector<std::size_t> &population,
                                  std::size_t k) {
    const auto n = population.size();
    if (k > n)
      throw std::invalid_argument("Sample larger than population");
    std::vector<std::size_t> result;
    result.reserve(k);
    std::size_t setsize = 21;
    if (k > 5) {
      std::size_t power = 4;
      while (power < k * 3)
        power *= 4;
      setsize += power;
    }
    if (n <= setsize) {
      auto pool = population;
      for (std::size_t i = 0; i < k; ++i) {
        const auto j = below(n - i);
        result.push_back(pool[j]);
        pool[j] = pool[n - i - 1];
      }
    } else {
      std::unordered_set<std::size_t> selected;
      for (std::size_t i = 0; i < k; ++i) {
        auto j = below(n);
        while (selected.count(j))
          j = below(n);
        selected.insert(j);
        result.push_back(population[j]);
      }
    }
    return result;
  }

private:
  std::array<std::uint32_t, 624> words_{};
  std::size_t position_ = 624;
  std::uint32_t next() {
    if (position_ == 624) {
      for (std::size_t i = 0; i < 624; ++i) {
        const auto y =
            (words_[i] & 0x80000000U) | (words_[(i + 1) % 624] & 0x7fffffffU);
        words_[i] =
            words_[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1U) ? 0x9908b0dfU : 0U);
      }
      position_ = 0;
    }
    auto y = words_[position_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= y >> 18;
    return y;
  }
  std::size_t below(std::size_t n) {
    if (!n)
      throw std::invalid_argument("Empty random range");
    std::size_t bits = 0;
    for (auto v = n; v; v >>= 1)
      ++bits;
    std::uint64_t value;
    do {
      if (bits <= 32)
        value = next() >> (32 - bits);
      else
        value =
            next() | (static_cast<std::uint64_t>(next() >> (64 - bits)) << 32);
    } while (value >= n);
    return value;
  }
};
}
