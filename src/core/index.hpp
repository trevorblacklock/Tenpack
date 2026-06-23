#ifndef TENPACK_INDEX_H_DEFINED
#define TENPACK_INDEX_H_DEFINED

#include "extents.hpp"

#include <concepts>

namespace tn {

inline auto wrap_index(index_t index, stride_t dim) {
    return index >= 0 ? static_cast<shape_t>(index)
                      : static_cast<shape_t>(index + dim);
}

inline auto wrap_index(index_t index, shape_t dim) {
    return index >= 0 ? static_cast<shape_t>(index)
                      : static_cast<shape_t>(index + dim);
}

template<class Tp, class Layout>
    requires(std::unsigned_integral<typename std::decay_t<Tp>::value_type>)
constexpr auto flatten_index(Tp&&                         indices,
                             const core::extents<Layout>& extent) {
    const auto& shape   = extent.shape();
    const auto& strides = extent.strides();
    shape_t     offset  = 0;
    for (shape_t i = 0; i < indices.size(); ++i) {
        TN_ASSERT(indices[i] < shape[i],
                  "Invalid index {} for tensor of shape {}", indices, shape);
        offset += strides[i] * indices[i];
    }
    return offset;
}

template<class Tp, class Layout>
    requires(std::signed_integral<typename std::decay_t<Tp>::value_type>)
constexpr auto flatten_index(Tp&&                         indices,
                             const core::extents<Layout>& extent) {
    const auto& shape   = extent.shape();
    const auto& strides = extent.strides();
    shape_t     offset  = 0;
    for (shape_t i = 0; i < indices.size(); ++i) {
        auto idx = indices[i];
        auto dim = shape[i];
        TN_ASSERT(static_cast<shape_t>(std::abs(idx)) < dim,
                  "Invalid index {} for tensor of shape {}", indices, shape);
        offset += strides[i] * wrap_index(idx, dim);
    }
    return offset;
}

template<class Tp, class Layout>
    requires(std::unsigned_integral<typename std::decay_t<Tp>::value_type>)
constexpr auto unflatten_index(Tp&&                         index,
                               const core::extents<Layout>& extent) {
    const auto& strides = extent.strides();

    auto indices = std::vector<shape_t>(extent.rank());
    auto permute = std::vector<shape_t>(extent.rank());
    std::iota(permute.begin(), permute.end(), 0);
    std::sort(permute.begin(), permute.end(),
              [&strides](auto a, auto b) { return strides[a] > strides[b]; });

    for (shape_t i = 0; i < extent.rank(); ++i) {
        auto j      = permute[i];
        auto stride = strides[j];
        indices[j]  = index / stride;
        index %= stride;
    }
    return indices;
}

template<class Tp, class Layout>
    requires(std::signed_integral<typename std::decay_t<Tp>::value_type>)
constexpr auto unflatten_index(Tp&&                         index,
                               const core::extents<Layout>& extent) {
    const auto& shape   = extent.shape();
    const auto& strides = extent.strides();

    auto indices = std::vector<shape_t>(extent.rank());
    auto permute = std::vector<shape_t>(extent.rank());
    std::iota(permute.begin(), permute.end(), 0);
    std::sort(permute.begin(), permute.end(),
              [&strides](auto a, auto b) { return strides[a] > strides[b]; });

    for (shape_t i = 0; i < extent.rank(); ++i) {
        auto j      = permute[i];
        auto stride = strides[j];
        indices[j]  = wrap_index(index, shape[j]) / stride;
        index %= stride;
    }
    return indices;
}

} // namespace tn

#endif
