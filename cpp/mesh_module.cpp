// C++ of the mesh operations of fieldtracing.mesh (optional extension fieldtracing._mesh): the
// unique edges of a triangle list, its boundary triangles, the two-ring of mesh neighbours and
// the quadratic least-squares fit over it. Each mirrors the Python function of the same name
// (`_edges`, `_on_boundary`, `_two_ring`, `_quadratic_fit`); the package uses the Python
// versions where the extension is not installed.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;
using Ints = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

// sorted (a < b) edge keys of every triangle side, as a·n + b
std::vector<std::int64_t> side_keys(const Ints& tri, std::int64_t n) {
  if (tri.ndim() != 2 || tri.shape(1) != 3) throw std::invalid_argument("triangles must be (M, 3)");
  const auto m = static_cast<std::size_t>(tri.shape(0));
  const std::int64_t* t = tri.data();
  std::vector<std::int64_t> keys(3 * m);
  for (std::size_t i = 0; i < m; ++i)
    for (int s = 0; s < 3; ++s) {
      std::int64_t a = t[3 * i + static_cast<std::size_t>(s)], b = t[3 * i + static_cast<std::size_t>((s + 1) % 3)];
      if (a > b) std::swap(a, b);
      keys[3 * i + static_cast<std::size_t>(s)] = a * n + b;
    }
  return keys;
}

std::int64_t vertex_count(const Ints& tri) {
  std::int64_t n = 0;
  for (py::ssize_t i = 0; i < tri.size(); ++i) n = std::max(n, tri.data()[i] + 1);
  return n;
}

// symmetric eigenvalues of a 5×5 matrix (cyclic Jacobi), ascending
std::array<double, 5> eigenvalues(std::array<double, 25> a) {
  double norm = 0.0;  // Frobenius norm², invariant under the rotations
  for (const double x : a) norm += x * x;
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < 5; ++p)
      for (int q = p + 1; q < 5; ++q) off += a[5 * p + q] * a[5 * p + q];
    if (off <= 1e-32 * norm || norm == 0.0) break;  // off-diagonal at rounding level
    for (int p = 0; p < 5; ++p)
      for (int q = p + 1; q < 5; ++q) {
        const double apq = a[5 * p + q];
        if (std::abs(apq) < 1e-300) continue;
        const double theta = (a[5 * q + q] - a[5 * p + p]) / (2.0 * apq);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < 5; ++k) {
          const double akp = a[5 * k + p], akq = a[5 * k + q];
          a[5 * k + p] = c * akp - s * akq;
          a[5 * k + q] = s * akp + c * akq;
        }
        for (int k = 0; k < 5; ++k) {
          const double apk = a[5 * p + k], aqk = a[5 * q + k];
          a[5 * p + k] = c * apk - s * aqk;
          a[5 * q + k] = s * apk + c * aqk;
        }
      }
  }
  std::array<double, 5> e{};
  for (int i = 0; i < 5; ++i) e[static_cast<std::size_t>(i)] = a[6 * i];
  std::sort(e.begin(), e.end());
  return e;
}

// solve the 5×5 system A x = b (several right-hand sides, columns of b) by LU with pivoting
void solve(std::array<double, 25> a, std::vector<double>& b, std::size_t cols) {
  std::array<int, 5> piv{};
  for (int k = 0; k < 5; ++k) {
    int p = k;
    for (int i = k + 1; i < 5; ++i)
      if (std::abs(a[5 * i + k]) > std::abs(a[5 * p + k])) p = i;
    piv[static_cast<std::size_t>(k)] = p;
    if (p != k) {
      for (int j = 0; j < 5; ++j) std::swap(a[5 * k + j], a[5 * p + j]);
      for (std::size_t c = 0; c < cols; ++c) std::swap(b[static_cast<std::size_t>(k) * cols + c], b[static_cast<std::size_t>(p) * cols + c]);
    }
    for (int i = k + 1; i < 5; ++i) {
      const double f = a[5 * i + k] / a[5 * k + k];
      a[5 * i + k] = f;
      for (int j = k + 1; j < 5; ++j) a[5 * i + j] -= f * a[5 * k + j];
      for (std::size_t c = 0; c < cols; ++c) b[static_cast<std::size_t>(i) * cols + c] -= f * b[static_cast<std::size_t>(k) * cols + c];
    }
  }
  for (int i = 4; i >= 0; --i)
    for (std::size_t c = 0; c < cols; ++c) {
      double s = b[static_cast<std::size_t>(i) * cols + c];
      for (int j = i + 1; j < 5; ++j) s -= a[5 * i + j] * b[static_cast<std::size_t>(j) * cols + c];
      b[static_cast<std::size_t>(i) * cols + c] = s / a[5 * i + i];
    }
}

// The pupil a ray cloud covers (fieldtracing.chain._support): inside the allowed triangles of its
// Delaunay mesh, or within `reach` of the nearest sample. Point location and nearest sample over
// uniform buckets.
class SupportTest {
 public:
  SupportTest(const Doubles& points, const Ints& simplices, const py::array_t<bool>& allowed,
              const Doubles& reach) {
    if (points.ndim() != 2 || points.shape(1) != 2) throw std::invalid_argument("points must be (N, 2)");
    if (simplices.ndim() != 2 || simplices.shape(1) != 3) throw std::invalid_argument("simplices must be (M, 3)");
    n_ = static_cast<std::size_t>(points.shape(0));
    p_.assign(points.data(), points.data() + 2 * n_);
    reach_.assign(reach.data(), reach.data() + n_);
    for (const double r : reach_) max_reach_ = std::max(max_reach_, r);
    const auto m = static_cast<std::size_t>(simplices.shape(0));
    tri_.assign(simplices.data(), simplices.data() + 3 * m);
    allowed_.assign(allowed.data(), allowed.data() + m);
    x0_ = y0_ = std::numeric_limits<double>::infinity();
    double x1 = -x0_, y1 = -y0_;
    for (std::size_t i = 0; i < n_; ++i) {
      x0_ = std::min(x0_, p_[2 * i]);
      x1 = std::max(x1, p_[2 * i]);
      y0_ = std::min(y0_, p_[2 * i + 1]);
      y1 = std::max(y1, p_[2 * i + 1]);
    }
    const double w = std::max(x1 - x0_, 1e-300), h = std::max(y1 - y0_, 1e-300);
    const double cells = std::max<double>(1.0, static_cast<double>(n_) / 2.0);
    cell_ = std::sqrt(w * h / cells);
    if (!(cell_ > 0)) cell_ = std::max(w, h);
    nx_ = static_cast<long>(std::floor(w / cell_)) + 1;
    ny_ = static_cast<long>(std::floor(h / cell_)) + 1;
    tri_bucket_.assign(static_cast<std::size_t>(nx_ * ny_), {});
    pt_bucket_.assign(static_cast<std::size_t>(nx_ * ny_), {});
    for (std::size_t i = 0; i < n_; ++i) pt_bucket_[bucket(cx(p_[2 * i]), cy(p_[2 * i + 1]))].push_back(static_cast<std::int64_t>(i));
    for (std::size_t t = 0; t < m; ++t) {
      double a0 = HUGE_VAL, a1 = -HUGE_VAL, b0 = HUGE_VAL, b1 = -HUGE_VAL;
      for (int c = 0; c < 3; ++c) {
        const auto v = static_cast<std::size_t>(tri_[3 * t + static_cast<std::size_t>(c)]);
        a0 = std::min(a0, p_[2 * v]);
        a1 = std::max(a1, p_[2 * v]);
        b0 = std::min(b0, p_[2 * v + 1]);
        b1 = std::max(b1, p_[2 * v + 1]);
      }
      for (long j = cy(b0); j <= cy(b1); ++j)
        for (long i = cx(a0); i <= cx(a1); ++i) tri_bucket_[bucket(i, j)].push_back(static_cast<std::int64_t>(t));
    }
  }

  py::array_t<bool> inside(const Doubles& qx, const Doubles& qy) const {
    if (qx.size() != qy.size()) throw std::invalid_argument("x and y differ in size");
    const auto q = static_cast<std::size_t>(qx.size());
    py::array_t<bool> out(qx.request().shape);
    bool* o = out.mutable_data();
    const double* x = qx.data();
    const double* y = qy.data();
    {
      py::gil_scoped_release free;
      for (std::size_t k = 0; k < q; ++k) o[k] = test(x[k], y[k]);
    }
    return out;
  }

 private:
  long cx(double x) const { return std::clamp(static_cast<long>(std::floor((x - x0_) / cell_)), 0L, nx_ - 1); }
  long cy(double y) const { return std::clamp(static_cast<long>(std::floor((y - y0_) / cell_)), 0L, ny_ - 1); }
  std::size_t bucket(long i, long j) const { return static_cast<std::size_t>(j * nx_ + i); }

  // in triangle t (barycentric, with the tolerance of scipy's find_simplex)
  bool in_triangle(std::size_t t, double x, double y) const {
    const auto a = static_cast<std::size_t>(tri_[3 * t]), b = static_cast<std::size_t>(tri_[3 * t + 1]),
               c = static_cast<std::size_t>(tri_[3 * t + 2]);
    const double ax = p_[2 * a], ay = p_[2 * a + 1];
    const double e1x = p_[2 * b] - ax, e1y = p_[2 * b + 1] - ay, e2x = p_[2 * c] - ax, e2y = p_[2 * c + 1] - ay;
    const double det = e1x * e2y - e1y * e2x;
    if (det == 0.0) return false;
    const double dx = x - ax, dy = y - ay;
    const double l1 = (dx * e2y - dy * e2x) / det, l2 = (e1x * dy - e1y * dx) / det;
    const double eps = 100.0 * std::numeric_limits<double>::epsilon();
    return l1 >= -eps && l2 >= -eps && 1.0 - l1 - l2 >= -eps;
  }

  bool test(double x, double y) const {
    const bool on_grid = x >= x0_ - cell_ && y >= y0_ - cell_ && x <= x0_ + (static_cast<double>(nx_) + 1) * cell_ &&
                         y <= y0_ + (static_cast<double>(ny_) + 1) * cell_;
    if (on_grid) {
      for (const auto t : tri_bucket_[bucket(cx(x), cy(y))])
        if (in_triangle(static_cast<std::size_t>(t), x, y)) {
          if (allowed_[static_cast<std::size_t>(t)]) return true;
          break;  // in a triangle across a gap: the reach of the nearest sample decides
        }
    }
    // nearest sample: rings of buckets around the query's (clamped) bucket
    const long i0 = cx(x), j0 = cy(y);
    const double ox = std::max({x0_ - x, x - (x0_ + static_cast<double>(nx_) * cell_), 0.0});
    const double oy = std::max({y0_ - y, y - (y0_ + static_cast<double>(ny_) * cell_), 0.0});
    const double outside = std::hypot(ox, oy);  // distance of the query to the bucket grid
    // only a sample within its reach counts, and every reach is ≤ max_reach_: the nearest sample
    // beyond that radius cannot pass, so the search stops there (same answer as the exact nearest)
    if (outside > max_reach_) return false;
    double best = HUGE_VAL;
    std::int64_t arg = -1;
    const long rings = std::min(std::max(nx_, ny_), static_cast<long>(std::ceil(max_reach_ / cell_)) + 1);
    for (long r = 0; r <= rings; ++r) {
      if (arg >= 0 && std::max(outside, (static_cast<double>(r) - 1.0) * cell_) > best) break;  // lower bound
      for (long j = j0 - r; j <= j0 + r; ++j) {
        if (j < 0 || j >= ny_) continue;
        for (long i = i0 - r; i <= i0 + r; ++i) {
          if (i < 0 || i >= nx_ || (std::abs(i - i0) != r && std::abs(j - j0) != r)) continue;
          for (const auto v : pt_bucket_[bucket(i, j)]) {
            const double d = std::hypot(p_[2 * static_cast<std::size_t>(v)] - x, p_[2 * static_cast<std::size_t>(v) + 1] - y);
            if (d < best) {
              best = d;
              arg = v;
            }
          }
        }
      }
    }
    return arg >= 0 && best <= max_reach_ && best <= reach_[static_cast<std::size_t>(arg)];
  }

  std::size_t n_ = 0;
  std::vector<double> p_, reach_;
  std::vector<std::int64_t> tri_;
  std::vector<bool> allowed_;
  double x0_ = 0, y0_ = 0, cell_ = 1, max_reach_ = 0;
  long nx_ = 1, ny_ = 1;
  std::vector<std::vector<std::int64_t>> tri_bucket_, pt_bucket_;
};

}  // namespace

PYBIND11_MODULE(_mesh, m) {
  py::class_<SupportTest>(m, "SupportTest", "The pupil a ray cloud covers (chain._support), in C++")
      .def(py::init<const Doubles&, const Ints&, const py::array_t<bool>&, const Doubles&>(), py::arg("points"),
           py::arg("simplices"), py::arg("allowed"), py::arg("reach"))
      .def("inside", &SupportTest::inside, py::arg("x"), py::arg("y"));

  m.doc() = "Mesh operations of fieldtracing.mesh in C++ (optional)";

  m.def("edges", [](const Ints& tri) {
    const std::int64_t n = vertex_count(tri);
    auto keys = side_keys(tri, n);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    Ints out({static_cast<py::ssize_t>(keys.size()), py::ssize_t{2}});
    auto* o = out.mutable_data();
    for (std::size_t i = 0; i < keys.size(); ++i) {
      o[2 * i] = keys[i] / n;
      o[2 * i + 1] = keys[i] % n;
    }
    return out;
  }, "Unique undirected edges (E, 2), sorted, as fieldtracing.mesh._edges");

  m.def("on_boundary", [](const Ints& tri) {
    const std::int64_t n = vertex_count(tri);
    const auto keys = side_keys(tri, n);
    std::vector<std::int64_t> sorted = keys;
    std::sort(sorted.begin(), sorted.end());
    const auto m3 = keys.size();
    py::array_t<bool> out(static_cast<py::ssize_t>(m3 / 3));
    auto* o = out.mutable_data();
    for (std::size_t i = 0; i < m3 / 3; ++i) {
      bool single = false;
      for (int s = 0; s < 3; ++s) {
        const auto r = std::equal_range(sorted.begin(), sorted.end(), keys[3 * i + static_cast<std::size_t>(s)]);
        single = single || (r.second - r.first) == 1;
      }
      o[i] = single;
    }
    return out;
  }, "True for triangles with an edge no other triangle shares, as fieldtracing.mesh._on_boundary");

  m.def("two_ring", [](std::int64_t n, const Ints& tri) {
    // neighbours (edges) per node, then the union with their neighbours, without the node
    auto keys = side_keys(tri, std::max<std::int64_t>(n, 1));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<std::vector<std::int64_t>> adj(static_cast<std::size_t>(n));
    const std::int64_t nn = std::max<std::int64_t>(n, 1);
    for (const auto k : keys) {
      adj[static_cast<std::size_t>(k / nn)].push_back(k % nn);
      adj[static_cast<std::size_t>(k % nn)].push_back(k / nn);
    }
    std::vector<std::int64_t> indptr(static_cast<std::size_t>(n) + 1, 0), indices;
    std::vector<std::int64_t> ring;
    for (std::int64_t i = 0; i < n; ++i) {
      ring.clear();
      for (const auto j : adj[static_cast<std::size_t>(i)]) {
        ring.push_back(j);
        for (const auto k : adj[static_cast<std::size_t>(j)]) ring.push_back(k);
      }
      std::sort(ring.begin(), ring.end());
      ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
      for (const auto j : ring)
        if (j != i) indices.push_back(j);
      indptr[static_cast<std::size_t>(i) + 1] = static_cast<std::int64_t>(indices.size());
    }
    Ints p(static_cast<py::ssize_t>(indptr.size())), q(static_cast<py::ssize_t>(indices.size()));
    std::copy(indptr.begin(), indptr.end(), p.mutable_data());
    std::copy(indices.begin(), indices.end(), q.mutable_data());
    return py::make_tuple(p, q);
  }, "CSR (indptr, indices) of the neighbours up to two edges away, sorted, without the node");

  m.def("quadratic_fit", [](const Doubles& points, const Doubles& values, const Ints& indptr,
                            const Ints& indices, int min_neighbours) {
    if (points.ndim() != 2 || points.shape(1) != 2) throw std::invalid_argument("points must be (N, 2)");
    if (values.ndim() != 2 || values.shape(0) != points.shape(0)) throw std::invalid_argument("values must be (N, C)");
    const auto n = static_cast<std::size_t>(points.shape(0));
    const auto cols = static_cast<std::size_t>(values.shape(1));
    const double* p = points.data();
    const double* v = values.data();
    const std::int64_t* ptr = indptr.data();
    const std::int64_t* idx = indices.data();
    Doubles grad({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(cols), py::ssize_t{2}});
    Doubles hess({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(cols), py::ssize_t{2}, py::ssize_t{2}});
    py::array_t<bool> valid(static_cast<py::ssize_t>(n));
    double* g = grad.mutable_data();
    double* h = hess.mutable_data();
    bool* ok = valid.mutable_data();
    {
      py::gil_scoped_release free;
      const double nan = std::numeric_limits<double>::quiet_NaN();
      std::vector<double> rhs(5 * cols);
      for (std::size_t i = 0; i < n; ++i) {
        const auto b0 = static_cast<std::size_t>(ptr[i]), b1 = static_cast<std::size_t>(ptr[i + 1]);
        double scale = 0.0;
        for (std::size_t s = b0; s < b1; ++s) {
          const auto j = static_cast<std::size_t>(idx[s]);
          const double dx = p[2 * j] - p[2 * i], dy = p[2 * j + 1] - p[2 * i + 1];
          scale = std::max(scale, dx * dx + dy * dy);
        }
        scale = std::sqrt(scale);
        if (!(scale > 0.0)) scale = 1.0;
        std::array<double, 25> a{};
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (std::size_t s = b0; s < b1; ++s) {
          const auto j = static_cast<std::size_t>(idx[s]);
          const double ux = (p[2 * j] - p[2 * i]) / scale, uy = (p[2 * j + 1] - p[2 * i + 1]) / scale;
          const double basis[5] = {ux, uy, 0.5 * ux * ux, ux * uy, 0.5 * uy * uy};
          for (int r = 0; r < 5; ++r) {
            for (int c = 0; c < 5; ++c) a[static_cast<std::size_t>(5 * r + c)] += basis[r] * basis[c];
            for (std::size_t c = 0; c < cols; ++c)
              rhs[static_cast<std::size_t>(r) * cols + c] += basis[r] * (v[j * cols + c] - v[i * cols + c]);
          }
        }
        const auto eig = eigenvalues(a);
        const bool good = static_cast<int>(b1 - b0) >= min_neighbours && eig[0] > 1e-10 * std::max(eig[4], 1e-300);
        ok[i] = good;
        for (std::size_t c = 0; c < cols; ++c) {
          double* gi = g + (i * cols + c) * 2;
          double* hi = h + (i * cols + c) * 4;
          if (!good) {
            gi[0] = gi[1] = nan;
            hi[0] = hi[1] = hi[2] = hi[3] = nan;
          }
        }
        if (!good) continue;
        solve(a, rhs, cols);
        for (std::size_t c = 0; c < cols; ++c) {
          double* gi = g + (i * cols + c) * 2;
          double* hi = h + (i * cols + c) * 4;
          gi[0] = rhs[0 * cols + c] / scale;
          gi[1] = rhs[1 * cols + c] / scale;
          hi[0] = rhs[2 * cols + c] / (scale * scale);
          hi[1] = hi[2] = rhs[3 * cols + c] / (scale * scale);
          hi[3] = rhs[4 * cols + c] / (scale * scale);
        }
      }
    }
    return py::make_tuple(grad, hess, valid);
  }, "Quadratic least-squares fit over the two-ring (gradients, Hessians, valid), as _quadratic_fit");
}
