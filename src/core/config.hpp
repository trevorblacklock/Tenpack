#ifndef TENPACK_CONFIG_H_DEFINED
#define TENPACK_CONFIG_H_DEFINED

#include <cstddef>

namespace tn {

using shape_t  = size_t;
using stride_t = ptrdiff_t;
using index_t  = ptrdiff_t;

} // namespace tn

#ifdef TN_DISABLE_ASSERTIONS
#define TN_ASSERT(condition, message, ...)
#else
#include "fmt/ranges.h"
#define TN_ASSERT(condition, message, ...)                                    \
    if (!(condition)) {                                                       \
        fmt::print(stderr, "{}:{}:\nAssertion failed: ", __FILE__, __LINE__); \
        fmt::println(stderr, message __VA_OPT__(, ) __VA_ARGS__);             \
        std::abort();                                                         \
    }
#endif

#ifndef TN_DISABLE_SIMD
#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;
#endif

#ifndef TN_MAX_ALIGNMNET
#define TN_MAX_ALIGNMENT 128ul
#endif

#ifndef TN_DEFAULT_ALIGNMENT
#define TN_DEFAULT_ALIGNMENT 64ul
#endif

static_assert((TN_DEFAULT_ALIGNMENT & (TN_DEFAULT_ALIGNMENT - 1)) == 0,
              "Default alignment must be a power of 2");
static_assert((TN_MAX_ALIGNMENT & (TN_MAX_ALIGNMENT - 1)) == 0,
              "Max alignment must be a power of 2");
static_assert((TN_DEFAULT_ALIGNMENT <= TN_MAX_ALIGNMENT),
              "Default alignment must be less than max alignment");

#ifdef TN_DISABLE_SIMD
#define TN_DEFAULT_ALLOCATOR(T) ::std::allocator<T>
#else
#define TN_DEFAULT_ALLOCATOR(T) ::tn::aligned_allocator<T, TN_DEFAULT_ALIGNMENT>
#endif

#ifndef TN_DEFAULT_LAYOUT
#define TN_DEFAULT_LAYOUT ::tn::core::row_major
#endif

#endif
