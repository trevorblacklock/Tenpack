#ifndef TENPACK_EXTENTS_H_DEFINED
#define TENPACK_EXTENTS_H_DEFINED

#include "strides.hpp"
#include "utils.hpp"

namespace tn::core {

template<core::layout_like Layout>
class extents {
 public:
    using layout_type = Layout;

    constexpr auto rank() const noexcept {
        return m_rank;
    }

    constexpr auto size() const noexcept {
        return m_size;
    }

    constexpr auto& shape() const noexcept {
        return m_shape;
    }

    constexpr auto& strides() const noexcept {
        return m_strides;
    }

    template<class T>
        requires(std::is_convertible_v<std::decay_t<T>, std::vector<shape_t>>)
    explicit extents(T&& shape)
        : m_rank(shape.size()),
          m_size(core::range_product(shape)),
          m_shape(std::forward<T>(shape)),
          m_strides(core::create_strides<Layout>(m_shape)) {
    }

    template<class T>
        requires(std::is_convertible_v<std::decay_t<T>, std::vector<shape_t>>)
    explicit extents(T&& shape, shape_t size)
        : m_rank(shape.size()),
          m_size(size),
          m_shape(std::forward<T>(shape)),
          m_strides(core::create_strides<Layout>(m_shape)) {
    }

    template<class T, class U>
        requires(std::is_convertible_v<std::decay_t<T>, std::vector<shape_t>>
                 && std::is_convertible_v<std::decay_t<U>,
                                          std::vector<stride_t>>)
    explicit extents(T&& shape, U&& strides)
        : m_rank(shape.size()),
          m_size(core::range_product(shape)),
          m_shape(std::forward<T>(shape)),
          m_strides(std::forward<U>(strides)) {
    }

    template<class T, class U>
        requires(std::is_convertible_v<std::decay_t<T>, std::vector<shape_t>>
                 && std::is_convertible_v<std::decay_t<U>,
                                          std::vector<stride_t>>)
    explicit extents(T&& shape, U&& strides, shape_t size)
        : m_rank(shape.size()),
          m_size(size),
          m_shape(std::forward<T>(shape)),
          m_strides(std::forward<U>(strides)) {
    }

    extents(const extents<Layout>& other)
        : m_rank(other.m_rank),
          m_size(other.m_size),
          m_shape(other.m_shape),
          m_strides(other.m_strides) {
    }

    extents(extents<Layout>&& other) noexcept
        : m_rank(std::exchange(other.m_rank, 0)),
          m_size(std::exchange(other.m_size, 0)),
          m_shape(std::move(other.m_shape)),
          m_strides(std::move(other.m_strides)) {
    }

    auto& operator=(const extents<Layout>& other) {
        m_rank    = other.m_rank;
        m_size    = other.m_size;
        m_shape   = other.m_shape;
        m_strides = other.m_strides;
        return *this;
    }

    auto& operator=(extents<Layout>&& other) noexcept {
        m_rank    = std::exchange(other.m_rank, 0);
        m_size    = std::exchange(other.m_size, 0);
        m_shape   = std::move(other.m_shape);
        m_strides = std::move(other.m_strides);
        return *this;
    }

 private:
    shape_t m_rank;
    shape_t m_size;

    std::vector<shape_t>  m_shape;
    std::vector<stride_t> m_strides;
};

} // namespace tn::core

#endif
