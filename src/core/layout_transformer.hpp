#ifndef TENPACK_LAYOUT_TRANSFORMER_H_DEFINED
#define TENPACK_LAYOUT_TRANSFORMER_H_DEFINED

#include "broadcast.hpp"

#include <memory>
#include <variant>

namespace tn {

namespace core {

// Used for checking contiguity in row major vs column major tensors
struct coalescing_visitor {
    auto operator()(row_major) {
        return a <= b;
    }

    auto operator()(column_major) {
        return a >= b;
    }

    stride_t a;
    stride_t b;
};

class layout_transformer_base {
 public:
    constexpr auto& shape() const {
        return m_virt_shape;
    }

    constexpr auto& strides() const {
        return m_virt_strides;
    }

    constexpr auto& permute() const {
        return m_permute;
    }

    template<class T, class U, class V, class W>
    explicit layout_transformer_base(T&& ranks,
                                     U&& shapes,
                                     V&& strides,
                                     W&& layouts)
        : m_num(ranks.size()),
          m_rank(range_max(ranks)),
          m_processed_rank(0),
          m_virt_rank(0),
          m_num_rank_spans(2),
          m_num_shape_spans(3 + m_num),
          m_num_stride_spans(4 * m_num),
          m_num_subspans(6 * m_num),
          m_buffer_size(m_num * sizeof(std::variant<row_major, column_major>)
                        + (m_num + 1) * (m_rank / 2)
                              * sizeof(std::pair<shape_t, shape_t>)
                        + m_num_subspans * sizeof(std::span<shape_t>)
                        + m_num_rank_spans * m_num * sizeof(shape_t)
                        + m_num_shape_spans * m_rank * sizeof(shape_t)
                        + m_num_stride_spans * m_rank * sizeof(stride_t)
                        + m_rank * sizeof(size_t) + m_rank * sizeof(double)),
          m_buffer(std::make_unique<std::byte[]>(m_buffer_size)) {
        // Initialize the spans by iterating through the byte buffer
        this->initialize_spans();

        // Fill all the data here
        std::copy(ranks.begin(), ranks.end(), m_base_ranks.begin());
        for (shape_t i = 0; i < m_num; ++i) {
            m_base_layouts[i]
                = std::variant<row_major, column_major>(layouts[i]);
            std::copy(shapes[i].begin(), shapes[i].end(),
                      m_base_shapes[i].begin());
            std::copy(strides[i].begin(), strides[i].end(),
                      m_base_strides[i].begin());
        }
        core::broadcast_shape_impl(m_real_shape, m_base_shapes, m_base_ranks,
                                   m_rank);

        this->process_base();
        this->initialize_coalesce_ranges();
        this->coalesce_shapes_strides();
        this->generate_optimal_permutation();
    }

 protected:
    void initialize_spans() {
        auto active = &m_buffer[0];

        m_base_layouts = std::span(
            reinterpret_cast<std::variant<row_major, column_major>*>(active),
            m_num);
        active += m_num * sizeof(std::variant<row_major, column_major>);

        m_coalesce_ranges = std::span(
            reinterpret_cast<std::span<std::pair<shape_t, shape_t>>*>(active),
            m_num);
        active += m_num * sizeof(std::span<std::pair<shape_t, shape_t>>);

        for (auto& coalesce_range : m_coalesce_ranges) {
            coalesce_range = std::span(
                reinterpret_cast<std::pair<shape_t, shape_t>*>(active),
                m_rank / 2);
            active += (m_rank / 2) * sizeof(std::pair<shape_t, shape_t>);
        }

        m_intersection_coalesce_range = std::span(
            reinterpret_cast<std::pair<shape_t, shape_t>*>(active), m_rank / 2);
        active += (m_rank / 2) * sizeof(std::pair<shape_t, shape_t>);

        m_base_ranks = std::span(reinterpret_cast<shape_t*>(active), m_num);
        active += m_num * sizeof(shape_t);

        m_base_shapes
            = std::span(reinterpret_cast<std::span<shape_t>*>(active), m_num);
        active += m_num * sizeof(std::span<shape_t>);

        for (auto& base_shape : m_base_shapes) {
            base_shape = std::span(reinterpret_cast<shape_t*>(active), m_rank);
            active += m_rank * sizeof(shape_t);
        }

        m_base_strides
            = std::span(reinterpret_cast<std::span<stride_t>*>(active), m_num);
        active += m_num * sizeof(std::span<stride_t>);

        for (auto& base_stride : m_base_strides) {
            base_stride
                = std::span(reinterpret_cast<stride_t*>(active), m_rank);
            active += m_rank * sizeof(stride_t);
        }

        m_processed_ranks
            = std::span(reinterpret_cast<shape_t*>(active), m_num);
        active += m_num * sizeof(shape_t);

        m_processed_shape
            = std::span(reinterpret_cast<shape_t*>(active), m_rank);
        active += m_rank * sizeof(shape_t);

        m_processed_strides
            = std::span(reinterpret_cast<std::span<stride_t>*>(active), m_num);
        active += m_num * sizeof(std::span<stride_t>);

        for (auto& processed_stride : m_processed_strides) {
            processed_stride
                = std::span(reinterpret_cast<stride_t*>(active), m_rank);
            active += m_rank * sizeof(stride_t);
        }

        m_real_shape = std::span(reinterpret_cast<shape_t*>(active), m_rank);
        active += m_rank * sizeof(shape_t);

        m_real_strides
            = std::span(reinterpret_cast<std::span<stride_t>*>(active), m_num);
        active += m_num * sizeof(std::span<stride_t>);

        for (auto& real_stride : m_real_strides) {
            real_stride
                = std::span(reinterpret_cast<stride_t*>(active), m_rank);
            active += m_rank * sizeof(stride_t);
        }

        m_virt_shape = std::span(reinterpret_cast<shape_t*>(active), m_rank);
        active += m_rank * sizeof(shape_t);

        m_virt_strides
            = std::span(reinterpret_cast<std::span<stride_t>*>(active), m_num);
        active += m_num * sizeof(std::span<stride_t>);

        for (auto& virt_stride : m_virt_strides) {
            virt_stride
                = std::span(reinterpret_cast<stride_t*>(active), m_rank);
            active += m_rank * sizeof(stride_t);
        }

        m_permute = std::span(reinterpret_cast<size_t*>(active), m_rank);
        active += m_rank * sizeof(size_t);

        m_cost = std::span(reinterpret_cast<double*>(active), m_rank);
        active += m_rank * sizeof(double);

        TN_ASSERT(
            active == &m_buffer[m_buffer_size],
            "Internal error! Expected size {} buffer but instead got size {}.",
            m_buffer_size, static_cast<uintptr_t>(active - &m_buffer[0]));
    }

    void process_base() {
        bool cont_flag;
        for (shape_t i = 0; i < m_num; ++i) {
            shape_t rank = 0;
            cont_flag    = true;
            for (shape_t j = 0; j < m_base_ranks[i]; ++j) {
                if (m_base_shapes[i][j] == 1) {
                    if (cont_flag)
                        continue;
                    m_processed_strides[i][rank] = 0;
                } else {
                    m_processed_strides[i][rank] = m_base_strides[i][j];
                }

                rank += 1;
                cont_flag = false;
            }
            m_processed_ranks[i] = rank;
        }

        cont_flag = true;
        for (shape_t i = 0; i < m_rank; ++i) {
            if (m_real_shape[i] == 1 && cont_flag)
                continue;

            m_processed_shape[m_processed_rank] = m_real_shape[i];
            m_processed_rank += 1;
            cont_flag = false;
        }
    }

    void initialize_coalesce_ranges() {
        bool condition;
        for (shape_t i = 0; i < m_num; ++i) {
            // Cannot coalesce rank one arrays
            auto rank = m_processed_ranks[i];
            if (rank == 1)
                continue;

            auto rank_diff = m_processed_rank - rank;

            shape_t start = 0;
            shape_t j     = 0;
            for (shape_t k = 1; k < m_base_ranks[i]; ++k) {
                auto visitor = coalescing_visitor(m_processed_strides[i][k - 1],
                                                  m_processed_strides[i][k]);
                condition    = std::visit(visitor, m_base_layouts[i])
                            || m_processed_strides[i][k - 1] == 0
                            || m_processed_strides[i][k] == 0;
                if (condition) {
                    if (k - start >= 2)
                        m_coalesce_ranges[i][j++]
                            = std::pair {start + rank_diff, k - 1 + rank_diff};
                    start = k;
                }
            }

            if (m_base_ranks[i] - start >= 2)
                m_coalesce_ranges[i][j] = std::pair {
                    start + rank_diff, m_base_ranks[i] - 1 + rank_diff};
        }

        for (shape_t i = 0; i < m_rank / 2; ++i) {
            auto& interval = m_intersection_coalesce_range[i];
            interval       = std::pair {std::numeric_limits<shape_t>::min(),
                                        std::numeric_limits<shape_t>::max()};
            for (shape_t j = 0; j < m_num; ++j) {
                const auto& comp = m_coalesce_ranges[j][i];
                interval.first   = std::max(interval.first, comp.first);
                interval.second  = std::min(interval.second, comp.second);
            }
        }
    }

    void coalesce_shapes_strides() {
        auto interval = &m_intersection_coalesce_range[0];
        for (shape_t i = 0; i < m_processed_rank; ++i) {
            if (i == interval->first && interval->second > interval->first) {
                auto n = 1 + interval->second - interval->first;
                m_virt_shape[m_virt_rank]
                    = range_product(std::span(&m_processed_shape[i], n));
                for (shape_t j = 0; j < m_num; ++j) {
                    auto k = i - m_processed_rank + m_processed_ranks[j];
                    m_virt_strides[j][m_virt_rank]
                        = range_min(std::span(&m_processed_strides[j][k], n));
                }
                i = interval->second;
                interval += 1;
            } else {
                m_virt_shape[m_virt_rank] = m_processed_shape[i];
                for (shape_t j = 0; j < m_num; ++j) {
                    auto k = i - m_processed_rank + m_processed_ranks[j];
                    m_virt_strides[j][m_virt_rank]
                        = (k < m_processed_ranks[j]) ? m_processed_strides[j][k]
                                                     : 0;
                }
            }
            m_virt_rank += 1;
        }

        m_permute    = m_permute.subspan(0, m_virt_rank);
        m_cost       = m_cost.subspan(0, m_virt_rank);
        m_virt_shape = m_virt_shape.subspan(0, m_virt_rank);
        for (shape_t i = 0; i < m_num; ++i)
            m_virt_strides[i] = m_virt_strides[i].subspan(0, m_virt_rank);
    }

    void generate_optimal_permutation() {
        std::iota(m_permute.begin(), m_permute.end(), 0);
        for (shape_t i = 0; i < m_virt_rank; ++i) {
            for (shape_t j = 0; j < m_num; ++j) {
                auto stride = m_virt_strides[j][i];
                m_cost[i] += stride == 0 ? 1.0 : 1.0 / stride;
            }
        }
        std::sort(m_permute.begin(), m_permute.end(),
                  [this](auto a, auto b) { return m_cost[a] < m_cost[b]; });
    }

    shape_t m_num;
    shape_t m_rank;
    shape_t m_processed_rank;
    shape_t m_virt_rank;

    std::size_t m_num_rank_spans;
    std::size_t m_num_shape_spans;
    std::size_t m_num_stride_spans;
    std::size_t m_num_subspans;

    std::size_t                  m_buffer_size;
    std::unique_ptr<std::byte[]> m_buffer;

    std::span<std::variant<row_major, column_major>> m_base_layouts;

    std::span<std::span<std::pair<shape_t, shape_t>>> m_coalesce_ranges;
    std::span<std::pair<shape_t, shape_t>> m_intersection_coalesce_range;

    std::span<shape_t>             m_base_ranks;
    std::span<std::span<shape_t>>  m_base_shapes;
    std::span<std::span<stride_t>> m_base_strides;

    std::span<shape_t>             m_processed_ranks;
    std::span<shape_t>             m_processed_shape;
    std::span<std::span<stride_t>> m_processed_strides;

    std::span<shape_t>             m_real_shape;
    std::span<std::span<stride_t>> m_real_strides;

    std::span<shape_t>             m_virt_shape;
    std::span<std::span<stride_t>> m_virt_strides;

    std::span<size_t> m_permute;
    std::span<double> m_cost;
};

} // namespace core

template<core::tensor_like... Tps>
class layout_transformer : public core::layout_transformer_base {
 public:
    using value_type    = std::tuple<Tps...>;
    using subvalue_type = std::tuple<typename Tps::value_type...>;
    using layout_type   = std::tuple<typename Tps::layout_type...>;
    using data_type     = std::tuple<std::span<typename Tps::value_type>...>;

    constexpr auto& data() const {
        return m_data;
    }

    explicit layout_transformer(const Tps&... args)
        : core::layout_transformer_base(
              std::array {args.rank()...},
              std::array {std::span(args.shape())...},
              std::array {std::span(args.strides())...},
              std::array {std::variant<core::row_major, core::column_major>(
                  typename Tps::layout_type {})...}),
          m_data(std::tuple {std::span(args.data(), args.size())...}) {
    }

 private:
    data_type m_data;
};

} // namespace tn

#endif
