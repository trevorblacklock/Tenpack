#ifndef TENPACK_BROADCAST_H_DEFINED
#define TENPACK_BROADCAST_H_DEFINED

#include "types.hpp"
#include "utils.hpp"

#include <span>

namespace tn {

namespace core {

inline void broadcast_shape_impl(std::span<shape_t>            out_shape,
                                 std::span<std::span<shape_t>> shapes,
                                 std::span<shape_t>            ranks,
                                 shape_t                       max_rank) {
    for (shape_t i = 0; i < max_rank; ++i) {
        shape_t sentinel = 1;
        for (shape_t j = 0; j < shapes.size(); ++j) {
            if (i >= ranks[j])
                continue;

            auto tmp = shapes[j][ranks[j] - i - 1];
            sentinel = std::max(sentinel, tmp);

            TN_ASSERT(tmp == sentinel || tmp == 1,
                      "Incompatible shapes for broadcasting!")
        }
        out_shape[max_rank - i - 1] = sentinel;
    }
}

} // namespace core

template<class... Tps>
    requires((std::is_convertible_v<std::decay_t<Tps>, std::vector<shape_t>>
              && ...))
inline auto broadcast_shape(const Tps&... args) {
    // Const cast to make compiler happy, shouldn't alter 'args' memory!
    auto shapes = std::array {
        std::span(const_cast<shape_t*>(args.data()), args.size())...};
    auto ranks     = std::array {args.size()...};
    auto max_rank  = core::range_max(ranks);
    auto out_shape = std::vector<shape_t>(max_rank);
    core::broadcast_shape_impl(out_shape, shapes, ranks, max_rank);
    return out_shape;
}

template<core::tensor_like... Tps>
class broadcast {
 public:
    constexpr auto& shape() const {
        return m_shape;
    }

    constexpr auto ndim() const {
        return m_num;
    }

    constexpr auto rank() const {
        return m_rank;
    }

    explicit broadcast(const Tps&... args)
        : m_rank(core::range_max(std::array {args.rank()...})),
          m_num(sizeof...(Tps)),
          m_shape(broadcast_shape(args.shape()...)) {
    }

 private:
    shape_t m_rank;
    shape_t m_num;

    std::vector<shape_t> m_shape;
};

} // namespace tn

#endif
