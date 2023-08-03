#ifndef STPP_VECTOR_UTILITIES_H
#define STPP_VECTOR_UTILITIES_H

#include <stddef.h>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T>& v) {
    // For reference, this was copy pasta'd from:
    // https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

template <class T>
void reorder(std::vector<T>& arr, std::vector<size_t> index) {
    // Fix all elements one by one

    size_t n = arr.size();
    for (size_t i = 0; i < n - 1; ++i) {
        // While index[i] and arr[i] are not fixed
        while (index[i] != index[index[i]]) {
            std::swap(arr[index[i]], arr[index[index[i]]]);
            std::swap(index[i], index[index[i]]);
        }
    }
}

template <typename... Args>
inline void Swap(size_t i, size_t alt, std::vector<double>& x, Args&... args) {
    Swap(i, alt, x);
    Swap(i, alt, args...);
}

template <>
inline void Swap(size_t i, size_t alt, std::vector<double>& x) {
    std::swap(x[i], x[alt]);
}

template <typename... Args>
void reorder(std::vector<size_t> index, Args&... args) {
    size_t n = index.size();
    for (size_t i = 0; i < n - 1; ++i) {
        while (index[i] != index[index[i]]) {
            // Sort each of the vectors

            // Walk the vectors and sort them
            Swap(index[i], index[index[i]], args...);

            // Last step, swap index
            std::swap(index[i], index[index[i]]);
        }
    }
}

inline std::vector<size_t> insertSimulatedTimesAndIndex(const std::vector<double>& z_curr,
                                                        std::vector<double>& inserted_times) {
    inserted_times.insert(inserted_times.end(), z_curr.begin(), z_curr.end());
    std::vector<size_t> idx1 = sort_indexes(inserted_times);
    reorder(inserted_times, idx1);

    return idx1;
}

/// Delete this one in favor of sweet variadic one
template <typename T>
inline std::vector<T> insertSimulatedSpatialPoints(const std::vector<T>& x, const std::vector<T>& simulated_x,
                                                   const std::vector<size_t>& idx) {
    std::vector<double> x_tmp = x;
    x_tmp.insert(x_tmp.end(), simulated_x.begin(), simulated_x.end());
    reorder(x_tmp, idx);
    return x_tmp;
}

template <typename T>
inline void insertSimulatedSpatialPoints(const std::vector<T>& z_curr_x, const std::vector<size_t>& idx,
                                         std::vector<T>& inserted_x) {
    inserted_x.insert(inserted_x.end(), z_curr_x.begin(), z_curr_x.end());
    reorder(inserted_x,
            idx);  ///@todo srowe: Is that valid? Is the index produced okay in a for loop with expanding vector?
}

#endif  // STPP_VECTOR_UTILITIES_H
