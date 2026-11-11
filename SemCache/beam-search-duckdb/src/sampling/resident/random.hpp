#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace semcache::sampling::resident_detail {
// CPython's integer-seeded MT19937 and rejection randbelow. It preserves
// Random(seed).shuffle ordering, including quality groups created after replay.
class Random {
  std::array<std::uint32_t, 624> mt{};
  std::size_t index = 624;
  std::uint32_t word() {
    if (index >= 624) {
      for (std::size_t i = 0; i < 624; ++i) {
        std::uint32_t y =
            (mt[i] & 0x80000000U) | (mt[(i + 1) % 624] & 0x7fffffffU);
        mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0);
      }
      index = 0;
    }
    auto y = mt[index++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= y >> 18;
    return y;
  }

public:
  explicit Random(std::uint64_t seed) {
    mt[0] = 19650218U;
    for (std::size_t i = 1; i < 624; ++i)
      mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
    std::vector<std::uint32_t> key{static_cast<std::uint32_t>(seed)};
    if (seed >> 32)
      key.push_back(seed >> 32);
    std::size_t i = 1, j = 0;
    for (std::size_t k = 624; k; --k) {
      mt[i] =
          (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + key[j] + j;
      if (++i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
      if (++j >= key.size())
        j = 0;
    }
    for (std::size_t k = 623; k; --k) {
      mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - i;
      if (++i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
    }
    mt[0] = 0x80000000U;
  }
  std::size_t below(std::size_t n) {
    if (!n)
      throw std::invalid_argument("Random range must be positive");
    int bits = 0;
    for (auto v = n; v; v >>= 1)
      ++bits;
    std::uint64_t r;
    do {
      if (bits <= 32)
        r = word() >> (32 - bits);
      else {
        r = word();
        r |= std::uint64_t(word() >> (64 - bits)) << 32;
      }
    } while (r >= n);
    return r;
  }
  template <class T> void shuffle(std::vector<T> &rows) {
    for (std::size_t i = rows.size(); i > 1; --i)
      std::swap(rows[i - 1], rows[below(i)]);
  }
};
}
