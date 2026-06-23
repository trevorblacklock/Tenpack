#ifndef TENPACK_TRANSPOSE_H_DEFINED
#define TENPACK_TRANSPOSE_H_DEFINED

#include "extents.hpp"
#include "index.hpp"
#include "types.hpp"

#include <initializer_list>
#include <memory>

namespace tn {

template<core::tensor_like Tn, class U>
    requires(std::is_integral_v<typename U::value_type>)
inline auto transpose(const Tn& x, const U& order) {
    static constexpr auto cond = std::is_signed_v<typename U::value_type>;
    TN_ASSERT(order.size() == x.rank(),
              "Unexpected order: {} for tensor of rank {}", order, x.rank());
    TN_ASSERT(std::all_of(order.begin(), order.end(),
                          [&x](auto y) {
                              if constexpr (cond)
                                  return std::cmp_less(std::abs(y), x.rank());
                              else
                                  return std::cmp_less(y, x.rank());
                          }),
              "Unexpected order: {} for tensor of rank {}", order, x.rank());
    auto shape_ref    = x.shape();
    auto strides_ref  = x.strides();
    auto accessor_ref = x.data_accessor();
    auto size         = x.size();

    auto shape_new   = std::vector<shape_t>(x.rank());
    auto strides_new = std::vector<stride_t>(x.rank());

    for (shape_t i = 0; i < x.rank(); ++i) {
        auto j = order[i];
        if constexpr (cond)
            j = wrap_index(j, x.rank());
        shape_new[i]   = shape_ref[j];
        strides_new[i] = strides_ref[j];
    }

    auto extents_new = core::extents<typename Tn::layout_type>(
        std::move(shape_new), std::move(strides_new), size);
    auto accessor_new
        = std::shared_ptr<typename Tn::value_type[]>(accessor_ref);

    return Tn(std::move(extents_new), std::move(accessor_new));
}

template<core::tensor_like Tn, class U>
    requires(std::is_integral_v<U>)
inline auto transpose(const Tn& x, std::initializer_list<U> order) {
    return transpose(x, std::span(order.begin(), order.end()));
}

template<core::tensor_like Tn>
inline auto transpose(const Tn& x) {
    auto order = std::vector<shape_t>(x.rank());
    std::iota(order.rbegin(), order.rend(), 0);
    return transpose(x, order);
}

template<core::tensor_like Tn>
inline auto matrix_transpose(const Tn& x) {
    TN_ASSERT(x.rank() >= 2,
              "Require at least rank 2 tensor to matrix transpose, was given "
              "rank {} instead",
              x.rank());
    auto order = std::vector<shape_t>(x.rank());
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[x.rank() - 1], order[x.rank() - 2]);
    return transpose(x, order);
}

template<core::tensor_like Tn, class U>
    requires(core::is_complex_v<typename Tn::value_type>
             && std::is_integral_v<typename U::value_type>)
inline auto hermitian_conj(const Tn& x, const U& order) {
    return transpose(x.conj(), order);
}

template<core::tensor_like Tn, class U>
    requires(core::is_complex_v<typename Tn::value_type>
             && std::is_integral_v<U>)
inline auto hermitian_conj(const Tn& x, std::initializer_list<U> order) {
    return transpose(x.conj(), std::span(order.begin(), order.end()));
}

template<core::tensor_like Tn>
    requires(core::is_complex_v<typename Tn::value_type>)
inline auto hermitian_conj(const Tn& x) {
    return transpose(x.conj());
}

template<core::tensor_like Tn>
    requires(core::is_complex_v<typename Tn::value_type>)
inline auto matrix_hermitian_conj(const Tn& x) {
    return matrix_transpose(x.conj());
}

} // namespace tn

#endif
