#ifndef TENPACK_RANDOM_H_DEFINED
#define TENPACK_RANDOM_H_DEFINED

#include "config.hpp"
#include "extents.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <initializer_list>
#include <random>
#include <type_traits>

namespace tn::random {

namespace global_generator {

static std::random_device device;
static std::mt19937       gen(device());

} // namespace global_generator

template<core::element_like Tp     = double,
         core::layout_like  Layout = TN_DEFAULT_LAYOUT,
         class Alloc               = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_floating_point_v<Tp>
             && std::is_convertible_v<std::decay_t<U>, std::vector<shape_t>>)
inline auto normal(double mean, double stdev, U&& shape) {
    auto normal = std::normal_distribution<Tp>(mean, stdev);
    auto x      = empty<Tp, Layout, Alloc>(std::forward<U>(shape));
    auto data   = x.data();

    for (shape_t i = 0; i < x.size(); ++i)
        data[i] = normal(global_generator::gen);

    return x;
}

template<core::element_like Tp     = double,
         core::layout_like  Layout = TN_DEFAULT_LAYOUT,
         class Alloc               = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_floating_point_v<Tp> && std::is_integral_v<U>)
inline auto normal(double mean, double stdev, std::initializer_list<U> shape) {
    TN_ASSERT(core::range_positive(shape), "Invalid shape {}, must be positive",
              shape);
    return normal(mean, stdev,
                  std::vector<shape_t>(shape.begin(), shape.end()));
}

template<core::element_like Tp     = double,
         core::layout_like  Layout = TN_DEFAULT_LAYOUT,
         class Alloc               = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_floating_point_v<Tp>
             && std::is_convertible_v<std::decay_t<U>, std::vector<shape_t>>)
inline auto standard_normal(U&& shape) {
    return normal(0, 1, std::forward<U>(shape));
}

template<core::element_like Tp     = double,
         core::layout_like  Layout = TN_DEFAULT_LAYOUT,
         class Alloc               = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_floating_point_v<Tp> && std::is_integral_v<U>)
inline auto standard_normal(std::initializer_list<U> shape) {
    return normal(0, 1, shape);
}

} // namespace tn::random

#endif
