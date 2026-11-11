#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace semcache::sampling::resident_detail {
// One bit per actual sample coordinate. Unknown and rejected are distinct;
// repeated coordinates keep bag multiplicity, while the request ledger shares
// model results by actual task/input identity.
class DecisionMask {
  std::size_t size_;
  std::vector<std::uint64_t> known_, accepted_;

public:
  explicit DecisionMask(std::size_t size)
      : size_(size), known_((size + 63) / 64), accepted_(known_.size()) {}
  void publish(std::size_t position, bool accepted) {
    if (position >= size_)
      throw std::out_of_range("Sample mask coordinate");
    const auto bit = std::uint64_t(1) << (position % 64);
    known_[position / 64] |= bit;
    if (accepted)
      accepted_[position / 64] |= bit;
    else
      accepted_[position / 64] &= ~bit;
  }
  bool known(std::size_t position) const {
    return position < size_ &&
           (known_[position / 64] & (std::uint64_t(1) << (position % 64)));
  }
  bool accepted(std::size_t position) const {
    return known(position) &&
           (accepted_[position / 64] & (std::uint64_t(1) << (position % 64)));
  }
  const std::vector<std::uint64_t> &known_words() const { return known_; }
  const std::vector<std::uint64_t> &accepted_words() const { return accepted_; }
  template <typename Visit> void each_known(const Visit &visit) const {
    for (std::size_t word = 0; word < known_.size(); ++word) {
      auto bits = known_[word];
      while (bits) {
        const auto index = static_cast<std::size_t>(__builtin_ctzll(bits));
        visit(word * 64 + index);
        bits &= bits - 1;
      }
    }
  }
  template <typename Visit> void each_accepted(const Visit &visit) const {
    for (std::size_t word = 0; word < accepted_.size(); ++word) {
      auto bits = accepted_[word];
      while (bits) {
        const auto index = static_cast<std::size_t>(__builtin_ctzll(bits));
        visit(word * 64 + index);
        bits &= bits - 1;
      }
    }
  }
};
}
