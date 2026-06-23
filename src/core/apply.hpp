#ifndef TENPACK_APPLY_H_DEFINED
#define TENPACK_APPLY_H_DEFINED

#include "layout_transformer.hpp"
#include "tensor.hpp"
#include "types.hpp"

#include <type_traits>
#include <utility>

namespace tn {

namespace core {

template<class Fn, class... Tps, size_t... I>
inline void apply_impl(Fn&&                                func,
                       std::tuple<Tps...>                  data,
                       std::span<shape_t>                  shape,
                       std::span<std::span<stride_t>>      strides,
                       std::span<shape_t>                  permutation,
                       std::index_sequence<I...>           is,
                       std::array<shape_t, sizeof...(Tps)> idxs = {}) {
    auto idx    = permutation.front();
    auto dim    = shape[idx];
    auto stride = std::array {strides[0][idx], strides[I + 1][idx]...};
    if (permutation.size() == 1) {
        for (shape_t i = 0; i < dim; ++i) {
            std::get<0>(data)[idxs[0]]
                = func((std::get<I + 1>(data)[idxs[I + 1]])...);
            idxs[0] += stride[0];
            ((idxs[I + 1] += stride[I + 1]), ...);
        }
    } else {
        permutation = permutation.subspan(1);
        for (shape_t i = 0; i < dim; ++i) {
            apply_impl(std::forward<Fn>(func), data, shape, strides,
                       permutation, is, idxs);
            idxs[0] += stride[0];
            ((idxs[I + 1] += stride[I + 1]), ...);
        }
    }
}

} // namespace core

template<class Fn, core::tensor_like... Tps>
    requires(std::is_invocable_v<std::decay_t<Fn>, typename Tps::value_type...>)
inline auto apply(Fn&& func, const Tps&... args) {
    using U = std::invoke_result_t<Fn, typename Tps::value_type...>;
    // First broadcast to find resultant shape and create emtpy tensor
    auto x   = broadcast(args...);
    auto out = empty<U>(x.shape());
    auto opt = layout_transformer(out, args...);
    core::apply_impl(std::forward<Fn>(func), opt.data(), opt.shape(),
                     opt.strides(), opt.permute(),
                     std::make_index_sequence<sizeof...(Tps)> {});
    return out;
}

} // namespace tn

#endif
