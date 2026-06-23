#ifndef TENPACK_TYPES_H_DEFINED
#define TENPACK_TYPES_H_DEFINED

#include "config.hpp"

#include <complex>
#include <concepts>
#include <cstdint>
#include <vector>

namespace tn {

namespace core {

class row_major {};

class column_major {};

template<class Tp>
concept row_like = std::same_as<row_major, Tp>;

template<class Tp>
concept column_like = std::same_as<column_major, Tp>;

template<class Tp>
concept layout_like = row_like<Tp> || column_like<Tp>;

template<class Tp>
concept element_like = std::same_as<Tp, double> || std::same_as<Tp, float>
                    || std::same_as<Tp, int32_t> || std::same_as<Tp, int64_t>;

template<class Tp>
struct is_complex : std::false_type {};

template<class Tp>
struct is_complex<std::complex<Tp>> : std::true_type {};

template<class Tp>
constexpr auto is_complex_v = is_complex<Tp>::value;

template<class Tp>
concept complex_like = is_complex_v<Tp>;

} // namespace core

template<class Tp, shape_t Align>
    requires(Align > 0 && ((Align & (Align - 1)) == 0)
             && (Align % alignof(Tp) == 0))
class aligned_allocator;

template<core::element_like Tp, core::layout_like Layout, class Alloc>
class tensor;

namespace core {

template<class>
struct is_tensor : std::false_type {};

template<class Tp, class Layout, class Alloc>
struct is_tensor<tensor<Tp, Layout, Alloc>> : std::true_type {};

template<class Tp>
constexpr auto is_tensor_v = is_tensor<Tp>::value;

template<class Tp>
concept tensor_like = is_tensor_v<Tp>;

} // namespace core

} // namespace tn

#endif
