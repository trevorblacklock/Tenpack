#ifndef TENPACK_TRANSFORM_H_DEFINED
#define TENPACK_TRANSFORM_H_DEFINED

#include "apply.hpp"
#include "layout_transformer.hpp"
#include "types.hpp"

#include <type_traits>
#include <utility>

namespace tn {

template<class Fn, core::tensor_like Tp, core::tensor_like... Tps>
    requires(std::is_invocable_v<Fn, typename Tps::value_type...>)
inline void transform(Fn&& func, Tp& out, const Tps&... args) {
    auto opt = layout_transformer(out, args...);
    core::apply_impl(std::forward<Fn>(func), opt.data(), opt.shape(),
                     opt.strides(), opt.permute(),
                     std::make_index_sequence<sizeof...(Tps)> {});
}

} // namespace tn

#endif
