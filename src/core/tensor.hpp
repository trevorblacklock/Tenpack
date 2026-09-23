#ifndef TENPACK_TENSOR_H_DEFINED
#define TENPACK_TENSOR_H_DEFINED

#include "allocator.hpp"
#include "config.hpp"
#include "extents.hpp"
#include "index.hpp"
#include "transpose.hpp"
#include "types.hpp"

#include <initializer_list>
#include <memory>

namespace tn {

template<core::element_like Tp     = double,
         core::layout_like  Layout = TN_DEFAULT_LAYOUT,
         class Alloc               = TN_DEFAULT_ALLOCATOR(Tp)>
class tensor {
 public:
    using value_type     = Tp;
    using layout_type    = Layout;
    using allocator_type = Alloc;
    using extents_type   = core::extents<layout_type>;

    constexpr auto size() const noexcept {
        return m_extents->size();
    }

    constexpr auto rank() const noexcept {
        return m_extents->rank();
    }

    constexpr auto& shape() const noexcept {
        return m_extents->shape();
    }

    constexpr auto& strides() const noexcept {
        return m_extents->strides();
    }

    constexpr auto extent(shape_t idx = 0) const {
        return m_extents->shape()[idx];
    }

    constexpr auto data() const noexcept {
        return m_data.get();
    }

    constexpr auto& extents() const noexcept {
        return *m_extents;
    }

    constexpr auto& data_accessor() const noexcept {
        return m_data;
    }

    constexpr auto& extents_accessor() const noexcept {
        return m_extents;
    }

    // Method that returns the conjugate
    constexpr auto conj() const
        requires(core::is_complex_v<value_type>)
    {
        auto conjugate = *this;
        auto data      = conjugate.data();
        for (shape_t i = 0; i < conjugate.size(); ++i)
            data[i] = std::conj(data[i]);
        return conjugate;
    }

    // Methods to transpose, no args just means a full axis reversal
    template<class U>
        requires(std::is_integral_v<typename U::value_type>)
    auto transpose(const U& order) const {
        return ::tn::transpose(*this, order);
    }

    template<class U>
        requires(std::is_integral_v<U>)
    auto transpose(std::initializer_list<U> order) const {
        return ::tn::transpose(*this, order);
    }

    auto transpose() const {
        return ::tn::transpose(*this);
    }

    // Alias for full transpose and matrix transpose
    auto T() const {
        return ::tn::transpose(*this);
    }

    auto mT() const {
        return ::tn::matrix_transpose(*this);
    }

    // Alias for hermitian conjugate
    template<class U>
        requires(std::is_integral_v<typename U::value_type>)
    auto hermitian_conj(const U& order) const
        requires(core::is_complex_v<value_type>)
    {
        return ::tn::hermitian_conj(*this, order);
    }

    template<class U>
        requires(std::is_integral_v<U>)
    auto hermitian_conj(std::initializer_list<U> order) const
        requires(core::is_complex_v<value_type>)
    {
        return ::tn::hermitian_conj(*this, order);
    }

    auto H() const
        requires(core::is_complex_v<value_type>)
    {
        return ::tn::hermitian_conj(*this);
    }

    auto mH() const
        requires(core::is_complex_v<value_type>)
    {
        return ::tn::matrix_hermitian_conj(*this);
    }

    template<std::integral Arg>
    constexpr auto& item(Arg index) {
        TN_ASSERT(std::cmp_less(index, this->size()),
                  "Index of {} out of bounds for tensor of size {}", index,
                  this->size());
        return m_data[index];
    }

    template<std::integral Arg>
    constexpr auto item(Arg index) const {
        TN_ASSERT(std::cmp_less(index, this->size()),
                  "Index of {} out of bounds for tensor of size {}", index,
                  this->size());
        return m_data[index];
    }

    constexpr auto& item() {
        TN_ASSERT(this->size() == 1, "Calling item() with no argument is only "
                                     "defined for a tensor of size 1")
        return this->item(0);
    }

    constexpr auto item() const {
        TN_ASSERT(this->size() == 1, "Calling item() with no argument is only "
                                     "defined for a tensor of size 1")
        return this->item(0);
    }

    template<class U>
        requires(sizeof(Tp) % sizeof(U) == 0 || sizeof(U) % sizeof(Tp) == 0)
    constexpr auto view() const {
        auto shape   = this->shape();
        auto strides = this->strides();
        auto found_ptr
            = std::find(strides.begin(), strides.end(), stride_t {1});
        TN_ASSERT(found_ptr != strides.end()
                      && sizeof(Tp) * shape[*found_ptr] % sizeof(U) == 0,
                  "Data must have a contiguous axis which is a multiple of {} "
                  "bytes to form a view",
                  sizeof(U));

        auto low_axis = std::distance(strides.begin(), found_ptr);
        auto ptr_view
            = std::reinterpret_pointer_cast<U[]>(this->data_accessor());
        auto shape_view  = std::vector(shape);
        auto stride_view = std::vector(strides);

        if constexpr (sizeof(U) > sizeof(Tp)) {
            constexpr auto ratio = sizeof(U) / sizeof(Tp);
            shape_view[low_axis] /= ratio;
            for (shape_t i = 0; i < this->rank(); ++i) {
                if (std::cmp_equal(i, low_axis))
                    continue;
                stride_view[i] /= ratio;
            }
        } else {
            constexpr auto ratio = sizeof(U) / sizeof(Tp);
            shape_view[low_axis] *= ratio;
            for (shape_t i = 0; i < this->rank(); ++i) {
                if (std::cmp_equal(i, low_axis))
                    continue;
                stride_view[i] *= ratio;
            }
        }

        auto extent = ::tn::core::extents<Layout>(std::move(shape_view),
                                                  std::move(stride_view));

        return ::tn::tensor<U, Layout, Alloc>(std::move(extent),
                                              std::move(ptr_view));
    }

    template<class U>
        requires(std::integral<typename U::value_type>)
    constexpr auto& item(const U& indices) {
        return m_data[::tn::core::flatten_index(indices, *m_extents)];
    }

    template<class U>
        requires(std::integral<typename U::value_type>)
    constexpr auto item(const U& indices) const {
        return m_data[::tn::core::flatten_index(indices, *m_extents)];
    }

    template<std::integral... Args>
    constexpr auto& item(Args... indices) {
        return m_data[::tn::core::flatten_index(std::array {indices...},
                                                *m_extents)];
    }

    template<std::integral... Args>
    constexpr auto item(Args... indices) const {
        return m_data[::tn::core::flatten_index(std::array {indices...},
                                                *m_extents)];
    }

    explicit tensor(const extents_type& extent)
        : m_extents(std::make_unique<extents_type>(extent)),
          m_data(allocate_shared_fast<value_type[]>(allocator_type(),
                                                    m_extents->size())) {
    }

    explicit tensor(extents_type&& extent)
        : m_extents(std::make_unique<extents_type>(std::move(extent))),
          m_data(allocate_shared_fast<value_type[]>(allocator_type(),
                                                    m_extents->size())) {
    }

    template<size_t D1, size_t S = D1>
    explicit tensor(Tp (&&array)[D1]) {
        auto shape   = std::vector {D1};
        auto strides = make_strides<layout_type>(shape);
        m_extents    = std::make_unique<extents_type>(std::move(shape),
                                                      std::move(strides), S);
        m_data       = allocate_shared_fast<value_type[]>(allocator_type(), S);
        std::copy(reinterpret_cast<Tp*>(&array),
                  reinterpret_cast<Tp*>(&array) + S, this->data());
    }

    template<size_t D1, size_t D2, size_t S = D1 * D2>
    explicit tensor(Tp (&&array)[D1][D2]) {
        auto shape   = std::vector {D1, D2};
        auto strides = make_strides<layout_type>(shape);
        m_extents    = std::make_unique<extents_type>(std::move(shape),
                                                      std::move(strides), S);
        m_data       = allocate_shared_fast<value_type[]>(allocator_type(), S);
        std::copy(reinterpret_cast<Tp*>(&array),
                  reinterpret_cast<Tp*>(&array) + S, this->data());
    }

    template<size_t D1, size_t D2, size_t D3, size_t S = D1 * D2 * D3>
    explicit tensor(Tp (&&array)[D1][D2][D3]) {
        auto shape   = std::vector {D1, D2, D3};
        auto strides = make_strides<layout_type>(shape);
        m_extents    = std::make_unique<extents_type>(std::move(shape),
                                                      std::move(strides), S);
        m_data       = allocate_shared_fast<value_type[]>(allocator_type(), S);
        std::copy(reinterpret_cast<Tp*>(&array),
                  reinterpret_cast<Tp*>(&array) + S, this->data());
    }

    template<size_t D1,
             size_t D2,
             size_t D3,
             size_t D4,
             size_t S = D1 * D2 * D3 * D4>
    explicit tensor(Tp (&&array)[D1][D2][D3][D4]) {
        auto shape   = std::vector {D1, D2, D3, D4};
        auto strides = make_strides<layout_type>(shape);
        m_extents    = std::make_unique<extents_type>(std::move(shape),
                                                      std::move(strides), S);
        m_data       = allocate_shared_fast<value_type[]>(allocator_type(), S);
        std::copy(reinterpret_cast<Tp*>(&array),
                  reinterpret_cast<Tp*>(&array) + S, this->data());
    }

    template<size_t D1,
             size_t D2,
             size_t D3,
             size_t D4,
             size_t D5,
             size_t S = D1 * D2 * D3 * D4 * D5>
    explicit tensor(Tp (&&array)[D1][D2][D3][D4][D5]) {
        auto shape   = std::vector {D1, D2, D3, D4, D5};
        auto strides = make_strides<layout_type>(shape);
        m_extents    = std::make_unique<extents_type>(std::move(shape),
                                                      std::move(strides), S);
        m_data       = allocate_shared_fast<value_type[]>(allocator_type(), S);
        std::copy(reinterpret_cast<Tp*>(&array),
                  reinterpret_cast<Tp*>(&array) + S, this->data());
    }

    template<size_t D1,
             size_t D2,
             size_t D3,
             size_t D4,
             size_t D5,
             size_t D6,
             size_t S = D1 * D2 * D3 * D4 * D5 * D6>
    explicit tensor(Tp (&&array)[D1][D2][D3][D4][D5][D6]) {
        auto shape   = std::vector {D1, D2, D3, D4, D5, D6};
        auto strides = make_strides<layout_type>(shape);
        m_extents    = std::make_unique<extents_type>(std::move(shape),
                                                      std::move(strides), S);
        m_data       = allocate_shared_fast<value_type[]>(allocator_type(), S);
        std::copy(reinterpret_cast<Tp*>(&array),
                  reinterpret_cast<Tp*>(&array) + S, this->data());
    }

    explicit tensor(extents_type&&          extent,
                    std::shared_ptr<Tp[]>&& data) noexcept
        : m_extents(std::make_unique<extents_type>(std::move(extent))),
          m_data(std::move(data)) {
    }

    tensor(const tensor<Tp, Layout, Alloc>& other)
        : m_extents(std::make_unique<extents_type>(other.extents())),
          m_data(other.m_data) {
    }

    tensor(tensor<Tp, Layout, Alloc>&& other) noexcept
        : m_extents(std::move(other.m_extents)),
          m_data(std::move(other.m_data)) {
    }

    auto& operator=(const tensor<Tp, Layout, Alloc>& other) {
        m_extents = std::make_unique<extents_type>(other.extents());
        m_data    = other.m_data;
        return *this;
    }

    auto& operator=(tensor<Tp, Layout, Alloc>&& other) noexcept {
        m_extents = std::move(other.m_extents);
        m_data    = std::move(other.m_data);
        return *this;
    }

    template<class U>
        requires(std::integral<typename U::value_type>)
    constexpr auto operator()(const U& indices) const {
        return ::tn::index(*this, indices);
    }

    template<std::integral... Args>
    constexpr auto operator()(Args... indices) const {
        return ::tn::index(*this, std::array {indices...});
    }

 private:
    std::unique_ptr<extents_type> m_extents;
    std::shared_ptr<value_type[]> m_data;
};

template<class Tp     = double,
         class Layout = TN_DEFAULT_LAYOUT,
         class Alloc  = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_convertible_v<std::decay_t<U>, std::vector<shape_t>>)
constexpr auto empty(U&& shape) {
    auto extent = core::extents<Layout>(std::forward<U>(shape));
    return tensor<Tp, Layout, Alloc>(std::move(extent));
}

template<class Tp     = double,
         class Layout = TN_DEFAULT_LAYOUT,
         class Alloc  = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_convertible_v<U, shape_t>)
constexpr auto empty(std::initializer_list<U> shape) {
    TN_ASSERT(core::range_positive(shape), "Invalid shape {}, must be positive",
              shape);
    return empty<Tp, Layout, Alloc>(
        std::vector<shape_t>(shape.begin(), shape.end()));
}

template<class Tp     = double,
         class Layout = TN_DEFAULT_LAYOUT,
         class Alloc  = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_convertible_v<std::decay_t<U>, std::vector<shape_t>>)
constexpr auto full(U&& shape, const Tp value) {
    auto extent = core::extents<Layout>(std::forward<U>(shape));
    auto t      = tensor<Tp, Layout, Alloc>(std::move(extent));
    std::fill(t.data(), t.data() + t.size(), value);
    return t;
}

template<class Tp     = double,
         class Layout = TN_DEFAULT_LAYOUT,
         class Alloc  = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_convertible_v<U, shape_t>)
constexpr auto full(std::initializer_list<U> shape, const Tp value) {
    TN_ASSERT(core::range_positive(shape), "Invalid shape {}, must be positive",
              shape);
    return full<Tp, Layout, Alloc>(
        std::vector<shape_t>(shape.begin(), shape.end()), value);
}

template<class Tp     = double,
         class Layout = TN_DEFAULT_LAYOUT,
         class Alloc  = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_convertible_v<std::decay_t<U>, std::vector<shape_t>>)
constexpr auto zeros(U&& shape) {
    return ::tn::full<Tp, Layout, Alloc>(std::forward<U>(shape), Tp {0});
}

template<class Tp     = double,
         class Layout = TN_DEFAULT_LAYOUT,
         class Alloc  = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_convertible_v<U, shape_t>)
constexpr auto zeros(std::initializer_list<U> shape) {
    return ::tn::full<Tp, Layout, Alloc>(shape, Tp {0});
}

template<class Tp     = double,
         class Layout = TN_DEFAULT_LAYOUT,
         class Alloc  = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_convertible_v<std::decay_t<U>, std::vector<shape_t>>)
constexpr auto ones(U&& shape) {
    return ::tn::full<Tp, Layout, Alloc>(std::forward<U>(shape), Tp {1});
}

template<class Tp     = double,
         class Layout = TN_DEFAULT_LAYOUT,
         class Alloc  = TN_DEFAULT_ALLOCATOR(Tp),
         class U>
    requires(std::is_convertible_v<U, shape_t>)
constexpr auto ones(std::initializer_list<U> shape) {
    return ::tn::full<Tp, Layout, Alloc>(shape, Tp {1});
}

} // namespace tn

#endif
