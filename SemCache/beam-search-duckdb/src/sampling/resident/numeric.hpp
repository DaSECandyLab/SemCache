#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace semcache::sampling::resident_detail {
struct Sum {
  double hi = 0, lo = 0;
  void add(double x) {
    double t = hi + x;
    lo += std::abs(hi) >= std::abs(x) ? (hi - t) + x : (x - t) + hi;
    hi = t;
  }
  double get() const { return lo && std::isfinite(lo) ? hi + lo : hi; }
};
struct FSum {
  std::vector<double> p;
  void add(double x) {
    std::size_t i = 0;
    for (double y : p) {
      if (std::abs(x) < std::abs(y))
        std::swap(x, y);
      double h = x + y, l = y - (h - x);
      if (l)
        p[i++] = l;
      x = h;
    }
    p.resize(i);
    if (x)
      p.push_back(x);
  }
  double get() const {
    double h = 0, l = 0;
    std::size_t n = p.size();
    if (n) {
      h = p[--n];
      while (n) {
        double x = h, y = p[--n];
        h = x + y;
        l = y - (h - x);
        if (l)
          break;
      }
      if (n && ((l < 0 && p[n - 1] < 0) || (l > 0 && p[n - 1] > 0))) {
        double y = l * 2, x = h + y;
        if (y == x - h)
          h = x;
      }
    }
    return h;
  }
};
}
