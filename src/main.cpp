#include "core/apply.hpp"
#include "core/broadcast.hpp"
#include "core/extents.hpp"
#include "core/layout_transformer.hpp"
#include "core/print.hpp"
#include "core/random.hpp"
#include "core/tensor.hpp"
#include "core/transform.hpp"
#include "core/transpose.hpp"
#include "core/types.hpp"

#include <benchmark/benchmark.h>

// inline void index_test(benchmark::State& state) {
//     auto a = tn::random::standard_normal({100, 100, 100});
//     auto b = tn::random::standard_normal({100, 100, 100}).T();

//     for (auto _ : state) {
//         auto c = tn::apply(std::plus(), a, b);
//         benchmark::DoNotOptimize(c);
//         benchmark::ClobberMemory();
//     }
// }

// BENCHMARK(index_test)
//     ->Repetitions(5)
//     ->DisplayAggregatesOnly(true)
//     ->Unit(benchmark::kMicrosecond);

// BENCHMARK_MAIN();

int main() {

    auto a = tn::random::standard_normal({2, 2});
    auto b = tn::random::standard_normal({2, 2}).T();

    fmt::println("{}", a);
    fmt::println("{}", b);

    auto c = tn::apply(std::plus(), a, b);

    tn::transform(std::minus(), c, c, b);

    fmt::println("{}", c(1, 0));
}
