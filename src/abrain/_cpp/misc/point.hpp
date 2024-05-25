#ifndef ES_HYPERNEAT_POINT_HPP
#define ES_HYPERNEAT_POINT_HPP

#include <cmath>
#include <cstdint>
#include <sstream>
#include <array>

namespace kgd::eshn::misc {

template <unsigned int DI>
class Point_t {
  using int_t = std::int16_t;

  std::array<int_t, DI> _data;

public:
  static constexpr auto DECIMALS = std::numeric_limits<int_t>::digits10 - 1;
  static constexpr auto DIMENSIONS = DI;

  static constexpr int RATIO = [] {
    int r = 1;
    for (unsigned int i=0; i<DECIMALS; i++) r *= 10;
    return r;
  }();

  static constexpr float EPSILON = [] {
    float v = 1;
    for (unsigned int i=0; i<DECIMALS; i++)  v /= 10.f;
    return v;
  }();

  Point_t(std::initializer_list<float> &&flist) {
    unsigned int i=0;
    for (const float f: flist) set(i++, f);
    for (; i<DIMENSIONS; i++) set(i, 0);
  }

  Point_t() { set(0); }

  static Point_t null () { return Point_t(); }

  [[nodiscard]] float x () const {  return get(0); }

  [[nodiscard]] float y () const {
    return get(1);
  }

  template <unsigned int DI_ = DIMENSIONS>
  std::enable_if_t<DI_ >= 3, float> z () const {
    return get(2);
  }

  [[nodiscard]] float get (unsigned int i) const {
    return _data[i] / static_cast<float>(RATIO);
  }

  void set (unsigned int i, const float v) {
    _data[i] = static_cast<int>(std::round(RATIO * v));
  }

  void set (const float v) {
    for (unsigned int i=0; i<DIMENSIONS; i++) set(i, v);
  }

  const auto& data () const {
    return _data;
  }

  Point_t& operator+= (const Point_t &that) {
    for (unsigned int i=0; i<DIMENSIONS; i++) set(i, get(i) + that.get(i));
    return *this;
  }

  Point_t& operator-= (const Point_t &that) {
    for (unsigned int i=0; i<DIMENSIONS; i++) set(i, get(i) - that.get(i));
    return *this;
  }

  Point_t& operator/= (float v) {
    for (unsigned int i=0; i<DIMENSIONS; i++) set(i, get(i) / v);
    return *this;
  }

  [[nodiscard]] float length () const {
    float sum = 0;
    for (unsigned int i=0; i<DIMENSIONS; i++) sum += get(i)*get(i);
    return std::sqrt(sum);
  }

  friend Point_t operator- (const Point_t &lhs, const Point_t &rhs) {
    Point_t res;
    for (unsigned int i=0; i<DIMENSIONS; i++) res.set(i, lhs.get(i) - rhs.get(i));
    return res;
  }

  friend Point_t operator* (float v, const Point_t &p) {
    Point_t res;
    for (unsigned int i=0; i<DIMENSIONS; i++) res.set(i, v * p.get(i));
    return res;
  }

  friend bool operator< (const Point_t &lhs, const Point_t &rhs) {
    return lhs._data < rhs._data;
  }

  friend bool operator== (const Point_t &lhs, const Point_t &rhs) {
    return lhs._data == rhs._data;
  }

  friend bool operator!= (const Point_t &lhs, const Point_t &rhs) {
    return lhs._data != rhs._data;
  }

  friend std::ostream& operator<< (std::ostream &os, const Point_t &p) {
    os << p.get(0);
    for (unsigned int i=1; i<DIMENSIONS; i++)  os << "," << p.get(i);
    return os;
  }

  friend std::istream& operator>> (std::istream &is, Point_t &p) {
    char c;
    float f;
    is >> f;
    p.set(0, f);
    for (unsigned int i=1; i<DIMENSIONS; i++) {
      is >> c >> f;
      p.set(i, f);
    }
    return is;
  }
};
using Point2D = Point_t<2>;
using Point3D = Point_t<3>;

} // end of namespace kgd::eshn::misc

#endif // ES_HYPERNEAT_POINT_HPP
