#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace semcache {
// Imported once under the GIL. Runtime additions retain Python's int/float
// distinction, including exact integer comparisons above the double range.
struct ResourceNumber {
  using Integer = boost::multiprecision::cpp_int;
  bool floating = false;
  Integer integer = 0;
  double real = 0;
  void add(const ResourceNumber &other) {
    if (!floating && !other.floating) {
      integer += other.integer;
      return;
    }
    real = (floating ? real : integer.convert_to<double>()) +
           (other.floating ? other.real : other.integer.convert_to<double>());
    floating = true;
  }
  bool greater(const ResourceNumber &other) const {
    if (!floating && !other.floating)
      return integer > other.integer;
    if (floating && other.floating)
      return real > other.real;
    if (floating) {
      if (!std::isfinite(real))
        return real > 0;
      Integer floor(real);
      return floor > other.integer ||
             (floor == other.integer && real > std::floor(real));
    }
    if (!std::isfinite(other.real))
      return other.real < 0;
    return integer > Integer(other.real);
  }
};

// CPython set layout for append-only model names: the existing hard-budget
// reference sums in set iteration order, which affects mixed float rounding.
// Hashes are imported once; all probing, union and iteration are native.
class ResourceSet {
public:
  struct Entry {
    std::size_t id = 0;
    std::uint64_t hash = 0;
    bool present = false;
  };
  ResourceSet() : table_(8) {}
  void add(std::size_t id, std::uint64_t hash) {
    if (insert(id, hash) && ++used_ * 5 >= (table_.size() - 1) * 3)
      resize(used_ > 50000 ? used_ * 2 : used_ * 4);
  }
  void merge(const ResourceSet &other) {
    if (!other.used_)
      return;
    if ((used_ + other.used_) * 5 >= (table_.size() - 1) * 3)
      resize((used_ + other.used_) * 2);
    if (!used_ && table_.size() == other.table_.size()) {
      *this = other;
      return;
    }
    if (!used_) {
      for (const auto &e : other.table_)
        if (e.present)
          insert_clean(e);
      used_ = other.used_;
    } else
      for (const auto &e : other.table_)
        if (e.present)
          add(e.id, e.hash);
  }
  const std::vector<Entry> &entries() const { return table_; }

private:
  std::vector<Entry> table_;
  std::size_t used_ = 0;
  bool insert(std::size_t id, std::uint64_t hash) {
    const auto mask = table_.size() - 1;
    auto i = hash & mask, perturb = hash;
    while (true) {
      auto probes = i + 9 <= mask ? 9 : 0;
      auto position = i;
      do {
        auto &e = table_[position++];
        if (!e.present) {
          e = {id, hash, true};
          return true;
        }
        if (e.id == id)
          return false;
      } while (probes--);
      perturb >>= 5;
      i = (i * 5 + 1 + perturb) & mask;
    }
  }
  void insert_clean(const Entry &e) { (void)insert(e.id, e.hash); }
  void resize(std::size_t minimum) {
    std::size_t size = 8;
    while (size <= minimum)
      size *= 2;
    if (size == table_.size())
      return;
    auto previous = std::move(table_);
    table_.assign(size, {});
    for (const auto &e : previous)
      if (e.present)
        insert_clean(e);
  }
};
}
