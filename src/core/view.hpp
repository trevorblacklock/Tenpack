#ifndef TENPACK_VIEW_H_DEFINED
#define TENPACK_VIEW_H_DEFINED

#include "extents.hpp"
#include "types.hpp"

#include <memory>

namespace tn {

template<core::tensor_like Tn, class U>
    requires(std::is_integral_v<typename U::value_type>)
inline auto view(const Tn& x, const U& index) {
    TN_ASSERT(index.size() <= x.rank(),
              "Unexpected number of indices {} for tensor of rank {}", index,
              x.rank());

    auto shape_ref    = x.shape();
    auto strides_ref  = x.strides();
    auto accessor_ref = x.data_accessor();
    auto data_ref     = x.data();

    std::vector<shape_t>  shape_new;
    std::vector<stride_t> strides_new;

    if (index.size() == x.rank()) {
        shape_new   = std::vector<shape_t> {1};
        strides_new = std::vector<stride_t> {1};
    } else {
        shape_new
            = std::vector(shape_ref.begin() + index.size(), shape_ref.end());
        strides_new = std::vector(strides_ref.begin() + index.size(),
                                  strides_ref.end());
    }

    auto offset = flatten_index(index, x.extents());
    auto ptr    = std::shared_ptr<typename Tn::value_type[]>(accessor_ref,
                                                             &data_ref[offset]);
    auto extent = core::extents<typename Tn::layout_type>(
        std::move(shape_new), std::move(strides_new));

    return Tn(std::move(extent), std::move(ptr));
}

template<core::tensor_like Tn, class U>
    requires(std::is_integral_v<U>)
inline auto view(const Tn& x, std::initializer_list<U> index) {
    return view(x, std::span(index.begin(), index.end()));
}

template<core::tensor_like Tn, class U>
    requires(std::is_integral_v<U>)
inline auto view(const Tn& x, U index) {
    return view(x, std::array {index});
}

} // namespace tn

#endif
