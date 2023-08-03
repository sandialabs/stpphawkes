#include <stddef.h>
#include <armadillo>
#include <chrono>
#include <iostream>
#include <random>
#include "../vector_utilities.h"

template <typename T>
T Difference(const std::vector<T>& x1, const std::vector<T>& x2) {
    T sum = 0;
    for (size_t i = 0; i < x1.size(); ++i) {
        sum += std::fabs(x1[i] - x2[i]);
    }

    return sum;
}

template <typename T>
void Print(const std::vector<T>& x) {
    for (auto val : x) {
        std::cout << "The val is " << val << std::endl;
    }
}

void Print(std::string x) { std::cout << x.c_str() << std::endl; }
template <class T>
void reorderold(std::vector<T>& v, std::vector<size_t> const& order) {
    for (size_t s = 1, d; s < order.size(); ++s) {
        for (d = order[s]; d < s; d = order[d])
            ;
        if (d == s)
            while (d = order[d], d != s)
                std::swap(v[s], v[d]);
    }
}

bool TestSortIndexes() {
    std::vector<double> x{5, 10, 6, 9, 7, 8};

    auto idx = sort_indexes(x);

    std::vector<size_t> expected_idx{0, 2, 4, 5, 3, 1};

    auto diff = Difference(idx, expected_idx);
    if (diff > 0) {
        Print("Test Sort Indexes: FAILED");
        Print("The output idx is");
        Print(idx);

        return false;
    }
    Print("Test Sort Indexes: PASSED");
    return true;
}

bool TestVariadicReorder() {
    std::vector<double> expected_x{5, 6, 7, 8, 9, 10};
    std::vector<double> expected_y{11, 12, 13, 14, 15, 16};

    //     std::vector<size_t> index{0, 5, 1, 4, 2, 3};

    std::vector<double> x{5, 10, 6, 9, 7, 8};
    std::vector<double> y{11, 16, 12, 15, 13, 14};
    auto index = sort_indexes(x);

    reorder(index, x, y);

    double error_x = 0.0;
    double error_y = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        error_x += std::fabs(x[i] - expected_x[i]);
        error_y += std::fabs(y[i] - expected_y[i]);
    }

    if ((error_x == 0.0) && (error_y == 0)) {
        std::cout << "Variadic Reorder: PASSED" << std::endl;

        return true;
    }
    std::cout << "Variadic Reorder: FAILED" << std::endl;
    std::cout << "x after reorder" << std::endl;
    Print(x);
    std::cout << "y after reorder" << std::endl;
    Print(y);
    std::cout << "The error x is " << error_x << " and the error_y is " << error_y << std::endl;
    return false;
}

bool TestVariadicReorderVsStandardReorder() {
    std::vector<double> expected_x{5, 6, 7, 8, 9, 10};
    std::vector<double> x{5, 10, 6, 9, 7, 8};
    auto x2(x);
    auto x3(x);

    auto idx = sort_indexes(x);

    reorder(x, idx);   // Non-variadic
    reorder(idx, x2);  // Variadic

    reorderold(x3, idx);

    auto first_diff = Difference(expected_x, x);
    auto second_diff = Difference(expected_x, x2);
    auto third_diff = Difference(expected_x, x3);
    bool success = true;
    if ((first_diff > 0.0)) {
        success = false;
        std::cout << "The first diff is " << first_diff << std::endl;
        Print("Error on first diff, here is x");
        Print(x);
    }
    if (second_diff > 0.0) {
        success = false;
        std::cout << "The second diff is " << second_diff << std::endl;
    }
    //     if (third_diff > 0.0) {
    //         success = false;
    //         std::cout << "The third diff is " << third_diff << std::endl;
    //     }

    if (success) {
        Print("Variadic Reorder Test: PASSED");
    }
    return success;
}
/*
void TestVariadicInsertSpatialPoints() {
  std::vector<double> t{0,3,6,7,8,11,12};
  std::vector<double> t_simulated{1,2,5,13};
  std::vector<size_t> idx = insertSimulex is
The val is 0
The val is 11
The val is 12
The val is 1
The val is 2
The val is 13
The val is 3
The val is 14
The val is 4
The val is 5
The val is 15
The val is 6
The val is 7
The val is 8
The val is 9
atedTimesAndIndex(t_simulated, t);

  std::vector<double> x{1,2,3,4,5,6,7};
  std::vector<double> x_simulated{8,9,10,11};

  std::vector<double> y{-1,-2,-3,-4,-5,-6,-7};
  std::vector<double> y_simulated{-8,-9,-10,-11};

  std::vector<double> variadic_x, variadic_y;
  //std::tie(variadic_x, variadic_y) = insertSimulatedSpatialPoints(x, y, idx, x_simulated, y_simulated);
  insertSimulatedSpatialPoints(x, y, idx, x_simulated, y_simulated);

}
*/

std::vector<size_t> insertTimesTest(const std::vector<double>& z_curr, std::vector<double>& inserted_times) {
    inserted_times.insert(inserted_times.end(), z_curr.begin(), z_curr.end());
    std::vector<size_t> idx1 = sort_indexes(inserted_times);
    std::sort(inserted_times.begin(), inserted_times.end());

    return idx1;
}

bool insertSimulatedTimesAndIndexComparison() {
    std::vector<double> t{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<double> simulated_t{1.1, 1.2, 3.1, 4.5, 6.7};
    auto t1(t);
    auto t2(t);

    auto idx = insertSimulatedTimesAndIndex(simulated_t, t1);
    insertTimesTest(simulated_t, t2);

    auto diff = Difference(t1, t2);
    bool success = true;
    if (diff > 0) {
        std::cout << "The difference is " << diff << std::endl;
        success = false;
        Print(t1);
        std::cout << "And t2" << std::endl;
        Print(t2);

        std::cout << "The index is " << std::endl;
        Print(idx);

        Print("hi");

        // Concatenate vectors
        t.insert(t.end(), simulated_t.begin(), simulated_t.end());
        // sort_indexes;
        Print("Start with new t");
        Print(t);
        auto idx1 = sort_indexes(t);
        Print("After sort indexes");
        Print(t);
        Print("The sort indexes");
        Print(idx1);
        reorder(t, idx1);
        Print("Yet again");
        Print(t);
    }

    return success;
}

double TestVariadicReorderSpeed() {
    int n = 70000;
    std::vector<double> t(n);
    std::iota(t.begin(), t.end(), 0);

    std::shuffle(t.begin(), t.end(), std::mt19937{std::random_device{}()});

    // Times are shuffled, lets sort indexes
    auto indices = sort_indexes(t);

    // Great, now let's get our spatial points
    std::vector<double> x(n);
    std::vector<double> y(n);
    std::iota(x.begin(), x.end(), -4);
    std::iota(y.begin(), y.end(), 1000);

    // Begin the sorting!
    auto start = std::chrono::system_clock::now();
    reorder(indices, x, y);
    auto end = std::chrono::system_clock::now();
    std::cout << "The time to reorder variadically is "
              << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;

    // Begin the sorting!
    start = std::chrono::system_clock::now();
    reorder(x, indices);
    reorder(y, indices);
    end = std::chrono::system_clock::now();
    std::cout << "The time to reorder non-variadically is "
              << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;
}

int main() {
    /*
        int n = 70000;
        std::vector<double> x(n);
        std::iota(x.begin(), x.end(), -4);
        std::vector<double> newx(x);
        std::vector<size_t> order(n, 0);

        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), std::mt19937{std::random_device{}()});
        std::vector<size_t> new_order(order);

        std::vector<double> sortx(x);

        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += x[i] - newx[i];
        }
        sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += order[i] - new_order[i];
        }
        std::cout << "The difference is " << sum << std::endl;

        auto start = std::chrono::system_clock::now();
        reorderold(x, order);
        auto stop = std::chrono::system_clock::now();

        std::cout << "The time for reorder old is " << std::chrono::duration<double, std::milli>(stop - start).count()
                  << std::endl;

        start = std::chrono::system_clock::now();
        reorder(newx, new_order);
        stop = std::chrono::system_clock::now();

        std::cout << "The time for new reorder is " << std::chrono::duration<double, std::milli>(stop - start).count()
                  << std::endl;

        sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += x[i] - newx[i];
        }

        std::cout << "The difference is" << sum << std::endl;

        // Next we want to understand: is it faster to call sort_indexes, then std::sort() or sort_indexes, then reorder
        std::shuffle(sortx.begin(), sortx.end(), std::mt19937{std::random_device{}()});
        auto reorderx(sortx);
        auto indices = sort_indexes(sortx);
        start = std::chrono::system_clock::now();
        std::sort(sortx.begin(), sortx.end());
        stop = std::chrono::system_clock::now();

        std::cout << "The time to std::sort is " << std::chrono::duration<double, std::milli>(stop - start).count()
                  << std::endl;
        start = std::chrono::system_clock::now();
        reorder(reorderx, indices);
        stop = std::chrono::system_clock::now();
        std::cout << "The time to reorder is " << std::chrono::duration<double, std::milli>(stop - start).count()
                  << std::endl;
        sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += sortx[i] - reorderx[i];
        }

        std::cout << "The difference is" << sum << std::endl;
    */
    TestVariadicReorderSpeed();

    bool success = true;
    Print("Start Variadic Reorder");
    success = success && TestVariadicReorder();

    Print("Start Sort Indexes");
    success = success && TestSortIndexes();

    Print("Variadic Reorder vs. Standard Reorder");
    success = success && TestVariadicReorderVsStandardReorder();

    Print("Insert Simulated Times Index Test");
    success = success && insertSimulatedTimesAndIndexComparison();
    if (!success) {
        return -1;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
