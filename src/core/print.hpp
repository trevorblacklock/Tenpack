#ifndef TENPACK_PRINT_H_DEFINED
#define TENPACK_PRINT_H_DEFINED

#include "tensor.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace tn {

struct print_options {
    size_t precision  = 8;
    size_t edge_items = 3;
    size_t threshold  = 1000;
    size_t width      = 75;
    double min_small  = 1e-9;
    double min_sci    = 1e-4;
    bool   suppress   = false;

    static print_options& instance() {
        static print_options instance;
        return instance;
    }

    print_options(const print_options&) = delete;

 private:
    print_options() = default;
};

namespace core {

template<class Tp>
inline void pretty_print_vector(std::ostringstream& ss,
                                const Tp&           tensor,
                                size_t              offset,
                                bool                edges) {
    static const auto& opts = print_options::instance();

    auto acc = offset;

    std::ostringstream tmp;
    tmp.copyfmt(ss);

    ss << '[';
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (edges && i == opts.edge_items) {
            tmp << "..., ";
            i = tensor.size() - opts.edge_items - 1;
        } else {
            auto x = tensor(i);

            if (x >= 0)
                tmp << " ";
            if ((opts.suppress && std::abs(x) < std::pow(10, -opts.precision))
                || std::abs(x) < opts.min_small)
                tmp << 0.0;
            else
                tmp << x;
            if (i != tensor.size() - 1)
                tmp << ", ";
        }

        acc += tmp.tellp();

        if (static_cast<size_t>(acc) > opts.width) {
            ss << '\n' << std::string(offset, ' ');
            acc = offset + tmp.tellp();
        }

        ss << tmp.str();

        tmp.str(std::string());
    }
    ss << ']';
}

template<class Tp>
inline void pretty_print_matrix(std::ostringstream& ss,
                                const Tp&           tensor,
                                size_t              offset,
                                bool                edges) {
    static const auto& opts = print_options::instance();

    ss << '[';
    for (size_t i = 0; i < tensor.extent(); ++i) {
        if (edges && i == opts.edge_items) {
            ss << "...,\n" << std::string(offset, ' ');
            i = tensor.extent() - opts.edge_items - 1;
        } else {
            pretty_print_select(ss, tensor.view({i}), offset + 1, edges);
            if (i != tensor.extent() - 1)
                ss << ",\n" << std::string(offset, ' ');
        }
    }
    ss << ']';
}

template<class Tp>
inline void pretty_print_tensor(std::ostringstream& ss,
                                const Tp&           tensor,
                                size_t              offset,
                                bool                edges) {
    static const auto& opts = print_options::instance();

    ss << '[';
    for (size_t i = 0; i < tensor.extent(); ++i) {
        if (edges && i == opts.edge_items) {
            ss << "...,\n\n" << std::string(offset, ' ');
            i = tensor.extent() - opts.edge_items - 1;
        } else {
            pretty_print_select(ss, tensor.view({i}), offset + 1, edges);
            if (i != tensor.extent() - 1)
                ss << ",\n\n" << std::string(offset, ' ');
        }
    }
    ss << ']';
}

template<class Tp>
inline void pretty_print_select(std::ostringstream& ss,
                                const Tp&           tensor,
                                size_t              offset,
                                bool                edges) {
    auto rank = tensor.rank();
    if (rank == 1)
        pretty_print_vector(ss, tensor, offset, edges);
    else if (rank == 2)
        pretty_print_matrix(ss, tensor, offset, edges);
    else
        pretty_print_tensor(ss, tensor, offset, edges);
}

inline void pretty_print_shape(std::ostringstream&     ss,
                               std::span<const size_t> shape) {
    ss << "(" << shape.front();
    for (size_t i = 1; i < shape.size(); ++i)
        ss << ", " << shape[i];
    ss << ")";
}

template<class Tp>
inline void pretty_print(std::ostringstream& ss, const Tp& tensor) {
    const auto& opts = print_options::instance();
    ss << std::fixed << std::setprecision(opts.precision);

    // TODO: Change to broadcasting and math funcs
    auto data = tensor.data();
    auto min  = std::abs(data[0]);
    for (size_t i = 1; i < tensor.size(); ++i)
        min = std::min(min, std::abs(data[i]));

    if (min < opts.min_sci)
        ss << std::scientific;

    ss << "tensor(";
    auto edges  = tensor.size() > opts.threshold;
    auto offset = 1 + ss.tellp();

    pretty_print_select(ss, tensor, offset, edges);
    if (edges) {
        ss << ",\n" << std::string(offset, ' ') << "shape=";
        pretty_print_shape(ss, tensor.shape());
    }
    ss << ")";
}

} // namespace core

template<core::tensor_like Tp>
inline auto& operator<<(std::ostream& os, const Tp& tensor) {
    std::ostringstream ss;
    core::pretty_print(ss, tensor);
    os << ss.str();
    return os;
}

} // namespace tn

template<tn::core::tensor_like Tp>
struct fmt::formatter<Tp> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const Tp& t, FormatContext& ctx) const {
        std::ostringstream ss;
        tn::core::pretty_print(ss, t);
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

#endif
