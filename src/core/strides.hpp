#ifndef TENPACK_STRIDES_H_DEFINED
#define TENPACK_STRIDES_H_DEFINED

#include "config.hpp"
#include "types.hpp"

#include <span>

namespace tn::core {

template<core::row_like Layout>
constexpr auto create_strides(std::span<const shape_t> shape) {
    auto     strides = std::vector<stride_t>(shape.size());
    stride_t stride  = 1;
    for (shape_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

template<core::column_like Layout>
constexpr auto create_strides(std::span<const shape_t> shape) {
    auto     strides = std::vector<stride_t>(shape.size());
    stride_t stride  = 1;
    for (shape_t i = 0; i < shape.size(); ++i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

} // namespace tn::core

#endif
