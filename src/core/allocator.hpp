#ifndef TENPACK_ALLOCATOR_H_DEFINED
#define TENPACK_ALLOCATOR_H_DEFINED

#include "config.hpp"

#include <memory>

namespace tn {

template<class Tp, shape_t Align>
    requires(Align > 0 && ((Align & (Align - 1)) == 0)
             && (Align % alignof(Tp) == 0))
class aligned_allocator {
 public:
    using value_type      = Tp;
    using size_type       = shape_t;
    using difference_type = ptrdiff_t;

    static constexpr shape_t alignment = Align;

    template<class U>
    struct rebind {
        using other = aligned_allocator<U, Align>;
    };

    constexpr aligned_allocator() noexcept = default;

    [[nodiscard]] Tp* allocate(size_type n) {
        return static_cast<Tp*>(
            ::operator new(n * sizeof(Tp), std::align_val_t(alignment)));
    }

    void deallocate(Tp* ptr, size_type n) {
        ::operator delete(ptr, n * sizeof(Tp), std::align_val_t(alignment));
    }
};

template<class Tp1, class Tp2, size_t Align1, size_t Align2>
constexpr bool operator==(const aligned_allocator<Tp1, Align1>&,
                          const aligned_allocator<Tp2, Align2>&) {
    return Align1 == Align2;
}

template<class Tp1, class Tp2, size_t Align1, size_t Align2>
constexpr bool operator!=(const aligned_allocator<Tp1, Align1>&,
                          const aligned_allocator<Tp2, Align2>&) {
    return Align1 != Align2;
}

template<class Tp, class Alloc>
    requires(std::is_arithmetic_v<std::remove_extent_t<Tp>>)
constexpr auto allocate_shared_fast(Alloc&& allocator, size_t n) {
    auto ptr = allocator.allocate(n);
    std::uninitialized_default_construct(ptr, ptr + n);
    return std::shared_ptr<Tp>(
        ptr, [allocator, n](std::remove_extent_t<Tp>* x) mutable {
            allocator.deallocate(x, n);
        });
}

} // namespace tn

#endif
