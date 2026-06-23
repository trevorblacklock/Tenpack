#ifndef TENPACK_UTILS_H_DEFINED
#define TENPACK_UTILS_H_DEFINED

#include <algorithm>
#include <functional>
#include <numeric>

namespace tn::core {

template<class Range>
constexpr auto range_sum(Range&& range) {
    return std::accumulate(std::ranges::begin(range), std::ranges::end(range),
                           std::ranges::range_value_t<Range> {0}, std::plus());
}

template<class Range>
constexpr auto range_product(Range&& range) {
    return std::accumulate(std::ranges::begin(range), std::ranges::end(range),
                           std::ranges::range_value_t<Range> {1},
                           std::multiplies());
}

template<class Range>
constexpr auto range_max(Range&& range) {
    return *std::max_element(std::ranges::begin(range),
                             std::ranges::end(range));
}

template<class Range>
constexpr auto range_min(Range&& range) {
    return *std::min_element(std::ranges::begin(range),
                             std::ranges::end(range));
}

template<class Range>
constexpr auto range_positive(Range&& range) {
    return std::all_of(std::ranges::begin(range), std::ranges::end(range),
                       [](auto x) { return x > 0; });
}

} // namespace tn::core

#endif
