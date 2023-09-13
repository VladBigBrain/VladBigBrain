#ifndef MLP_LIBRARIES_ALGORITHM_H_
#define MLP_LIBRARIES_ALGORITHM_H_

#include <algorithm>
#include <functional>
#include <numeric>

template <typename ContainerOut, typename ContainerIn, typename Function>
auto Transform(ContainerIn const& in, Function const& function)
    -> ContainerOut {
  ContainerOut out;
  out.reserve(in.size());
  std::transform(in.begin(), in.end(), std::back_inserter(out), function);
  return out;
}

template <typename ContainerOut, typename ContainerIn, typename Function>
auto Transform(ContainerIn& in, Function const& function) -> ContainerOut {
  ContainerOut out;
  out.reserve(in.size());
  std::transform(in.begin(), in.end(), std::back_inserter(out), function);
  return out;
}

template <typename Container, typename Function>
auto ForEach(Container const& container, Function const& function) -> void {
  std::for_each(container.begin(), container.end(), function);
}

template <typename Container, typename Function>
auto ForEach(Container& container, Function const& function) -> void {
  std::for_each(container.begin(), container.end(), function);
}

template <typename Container, typename Function>
auto ReverseForEach(Container& container, Function const& function) -> void {
  std::for_each(container.rbegin(), container.rend(), function);
}

template <typename Container, typename Function>
auto ReverseForEach(const Container& container, Function const& function)
    -> void {
  std::for_each(container.rbegin(), container.rend(), function);
}

template <typename ContainerOut, typename ContainerIn, typename Function>
auto Accumulate(ContainerIn const& in, ContainerOut&& _out,
                Function const& function) -> ContainerOut {
  return std::accumulate(in.begin(), in.end(), _out, function);
}

#endif  // MLP_LIBRARIES_ALGORITHM_H_
