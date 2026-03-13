#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <bitset>
#include <cassert>
#include <stdint.h>

#ifdef _MSC_VER
#define forceinline __forceinline
#elif defined(__GNUC__)
#define forceinline inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
#if __has_attribute(__always_inline__)
#define forceinline inline __attribute__((__always_inline__))
#else
#define forceinline inline
#endif
#else
#define forceinline inline
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE	(64)
#endif

#define L3CACHE_SIZE (22 * (1 << 20))
static double init_predicted_rate = 1.05;
static int seg_error_bound = 512;
static uint8_t expect_m = 10;
static int max_m = 2000;
static int nano_num_limit = 55000;
static uint8_t two_choice_threashold = 8;
static uint8_t group_overflow_nano_num = 1;
    
const uint32_t lockSet = ((uint32_t)1u << 31);
const uint32_t lockMask = ((uint32_t)1u << 31) - 1;

#define CAS(_p, _u, _v)                                             \
  (__atomic_compare_exchange_n(_p, _u, _v, false, __ATOMIC_ACQUIRE, \
                               __ATOMIC_ACQUIRE))

namespace line {

/*** Linear model and model builder ***/
// Forward declaration
template <class T>
class LinearModelBuilder;

// Linear regression model
template <class T>
class LinearModel {
 public:
  double a_ = 0;  // slope
  double b_ = 0;  // intercept

  LinearModel() = default;
  LinearModel(double a, double b) : a_(a), b_(b) {}
  explicit LinearModel(const LinearModel& other) : a_(other.a_), b_(other.b_) {}

  void expand(double expansion_factor) {
    a_ *= expansion_factor;
    b_ *= expansion_factor;
  }

  inline int predict(T key) const {
    return std::floor(a_ * static_cast<double>(key) + b_);
  }

  inline double predict_double(T key) const {
    return a_ * static_cast<double>(key) + b_;
  }

  T predict_reverse(int idx) const {
    long double bd_key = (idx - b_) / a_;
    return static_cast<T>(bd_key);
  }
};

template <class T>
class LinearModelBuilder {
 public:
  LinearModel<T>* model_;

  explicit LinearModelBuilder<T>(LinearModel<T>* model) : model_(model) {}

  inline void add(T x, int y) {
    count_++;
    x_sum_ += static_cast<long double>(x);
    y_sum_ += static_cast<long double>(y);
    xx_sum_ += static_cast<long double>(x) * x;
    xy_sum_ += static_cast<long double>(x) * y;
    x_min_ = std::min<T>(x, x_min_);
    x_max_ = std::max<T>(x, x_max_);
    y_min_ = std::min<double>(y, y_min_);
    y_max_ = std::max<double>(y, y_max_);
  }

  void build() {
    if (count_ <= 1) {
      model_->a_ = 0;
      model_->b_ = static_cast<double>(y_sum_);
      return;
    }

    if (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_ == 0) {
      // all values in a bucket have the same key.
      model_->a_ = 0;
      model_->b_ = static_cast<double>(y_sum_) / count_;
      return;
    }

    auto slope = static_cast<double>(
        (static_cast<long double>(count_) * xy_sum_ - x_sum_ * y_sum_) /
        (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_));
    auto intercept = static_cast<double>(
        (y_sum_ - static_cast<long double>(slope) * x_sum_) / count_);
    model_->a_ = slope;
    model_->b_ = intercept;

    // If floating point precision errors, fit spline
    if (model_->a_ <= 0) {
      model_->a_ = (y_max_ - y_min_) / (x_max_ - x_min_);
      model_->b_ = -static_cast<double>(x_min_) * model_->a_;
    }
  }

 private:
  int count_ = 0;
  long double x_sum_ = 0;
  long double y_sum_ = 0;
  long double xx_sum_ = 0;
  long double xy_sum_ = 0;
  T x_min_ = std::numeric_limits<T>::max();
  T x_max_ = std::numeric_limits<T>::lowest();
  double y_min_ = std::numeric_limits<double>::max();
  double y_max_ = std::numeric_limits<double>::lowest();
};
}