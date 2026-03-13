#pragma once

#include "line_base.h"
#include "nano_node_15.h"
#include "piecewise_linear_model.h"
#include <malloc.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <math.h>
#include <ostream>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

// #define USING_LOCK

thread_local unsigned long long smo_cnt = 0;
thread_local unsigned long long smo_finish = 0;
thread_local unsigned long long migrate_num = 0;

namespace line {
/**
 * @brief A class for Line nodes
 * 
 * @tparam T types of key
 * @tparam P types of payload
 */
template <class T, class P, bool allow_duplicates = false>
class LineDataNode {
 public:
  typedef nano<T, P> nano_type;
  typedef std::pair<T, P> V;
  typedef LineDataNode<T, P, allow_duplicates> self_type;

#ifdef USING_LOCK
  uint32_t leaf_lock_ = 0;                ///< leaf lock for migration
#endif
  uint32_t data_capacity_ = 0;            ///< number of nanos in this data node
  uint16_t group_num_ = 0;                ///< number of groups in this data node
  uint16_t split_point_ = 0;              ///< split point for node split

  LinearModel<T> model_;                  ///< linear model of the data node
  nano_type* nano_slots_ = nullptr;       ///< holds all nanos, [0, data_capacity_) for nano[] and [data_capacity_, data_capacity_ + group_num_) for overflow_nano[]

  // size variables for nanos 
  uint16_t* group_offsets_ = nullptr;     ///< beginning nano idx for each group

  // locks 
#ifdef USING_LOCK
  uint32_t* group_lock_ = nullptr;
#endif

  // structures using during migration
  uint32_t* migration_bitmap_ = nullptr;  ///< migrate status of each group during migration
  self_type* target_ = nullptr;           ///< first target of SMO
  self_type* target2_ = nullptr;          ///< second target of SMO (split only)

  T min_key_ = std::numeric_limits<T>::max();     ///< min key in node

  // pointers to previous and next leaves
  self_type* next_leaf_ = nullptr;
  self_type* prev_leaf_ = nullptr;

  LineDataNode() {
  };

  ~LineDataNode(){
    free(nano_slots_);
    free(group_offsets_);
#ifdef USING_LOCK
    if (group_lock_) {
      delete [] group_lock_;
    }
#endif
    if (migration_bitmap_) {
      delete [] migration_bitmap_;
    }
  }

  LineDataNode<T, P, allow_duplicates>* &get_next_leaf(){
    return next_leaf_;
  }
  LineDataNode<T, P, allow_duplicates>* &get_prev_leaf(){
    return prev_leaf_;
  }

  /*** General helper functions ***/
  inline nano_type * get_nano(int pos) const {
    return &(nano_slots_[pos]);
  }

  inline nano_type * get_of_nano(int pos) const {
    return &(nano_slots_[data_capacity_ + pos]);
  }

#ifdef USING_LOCK
  /*** concurrency functions ***/
  // leaf lock
  inline void get_leaf_lock() {
    uint32_t new_value = 0;
    uint32_t old_value = 0;
    do {
        while (true) {
            old_value = __atomic_load_n(&leaf_lock_, __ATOMIC_ACQUIRE);	
            if (!(old_value & lockSet)) {
                old_value &= lockMask;
                break;
            }
        }
        new_value = old_value | lockSet;
    } while (!CAS(&leaf_lock_, &old_value, new_value));
  }

  inline bool try_get_leaf_lock() {
    uint32_t v = __atomic_load_n(&leaf_lock_, __ATOMIC_ACQUIRE);
    if (v & lockSet) {
        return false;
    }
    auto old_value = v & lockMask;
    auto new_value = v | lockSet;
    return CAS(&leaf_lock_, &old_value, new_value);
  }

  inline void release_leaf_lock() {
    uint32_t v = leaf_lock_;
    __atomic_store_n(&leaf_lock_, (v + 1) & lockMask, __ATOMIC_RELEASE);
  }

  /*if the lock is set, return true*/
  inline bool test_leaf_lock_set(uint32_t &version) const {
    version = __atomic_load_n(&leaf_lock_, __ATOMIC_ACQUIRE);
    return (version & lockSet) != 0;
  }

  // test whether the version has change, if change, return true
  inline bool test_leaf_lock_version_change(uint32_t old_version) const {
    auto value = __atomic_load_n(&leaf_lock_, __ATOMIC_ACQUIRE);
    return (old_version != value);
  }

  // group lock
  inline void get_group_lock(int i) {
    uint32_t new_value = 0;
    uint32_t old_value = 0;
    do {
        while (true) {
            old_value = __atomic_load_n(&group_lock_[i], __ATOMIC_ACQUIRE);	
            if (!(old_value & lockSet)) {
                old_value &= lockMask;
                break;
            }
        }
        new_value = old_value | lockSet;
    } while (!CAS(&group_lock_[i], &old_value, new_value));
  }

  inline bool try_get_group_lock(int i) {
    uint32_t v = __atomic_load_n(&group_lock_[i], __ATOMIC_ACQUIRE);
    if (v & lockSet) {
        return false;
    }
    auto old_value = v & lockMask;
    auto new_value = v | lockSet;
    return CAS(&group_lock_[i], &old_value, new_value);
  }

  inline void release_group_lock(int i) {
    uint32_t v = group_lock_[i];
    __atomic_store_n(&group_lock_[i], (v + 1) & lockMask, __ATOMIC_RELEASE); 
  }

  /*if the lock is set, return true*/
  inline bool test_group_lock_set(int i, uint32_t &version) const {
    version = __atomic_load_n(&group_lock_[i], __ATOMIC_ACQUIRE);
    return (version & lockSet) != 0;
  }

  // test whether the version has change, if change, return true
  inline bool test_group_lock_version_change(int i, uint32_t old_version) const {
    auto value = __atomic_load_n(&group_lock_[i], __ATOMIC_ACQUIRE);
    return (old_version != value);
  }
#endif

  /*** Bulk loading and model building ***/
  // Allocate space for each array
  void initialize() {
    // 1. allocate space for nano_slots_[]
    int total_nano_num = this->data_capacity_ + group_num_ * group_overflow_nano_num;
    nano_slots_ = static_cast<nano_type*>(malloc(sizeof(nano_type) * total_nano_num));
    memset(nano_slots_, 0, sizeof(nano_type) * total_nano_num);

#ifdef USING_LOCK
    // 2. allocate space for group locks
    group_lock_ = new uint32_t[group_num_]();
#endif
  }

  void initialize_for_smo() {
    // 1. allocate space for nano_slots_[]
    int total_nano_num = this->data_capacity_ + group_num_ * group_overflow_nano_num;
    nano_slots_ = static_cast<nano_type*>(malloc(sizeof(nano_type) * total_nano_num));

#ifdef USING_LOCK
    // 2. allocate space for group locks
    group_lock_ = new uint32_t[group_num_]();
#endif
  }

  void calculate_group_offsets(const V values[], int num_keys) {
    int * group_key_num = new int[group_num_]();
    group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (group_num_ + 1)));

    for (int i = 0; i < num_keys; i++) {
      int group_idx = get_key_group(values[i].first);
      group_key_num[group_idx]++;
    }

    int acc_group_nano_num = 0;
    for (int i = 0; i < group_num_; i++) {
      group_offsets_[i] = acc_group_nano_num;
      acc_group_nano_num += std::max(1., std::ceil((double)(group_key_num[i] * init_predicted_rate) / NANO_INIT_KEY_NUM));
    }
    group_offsets_[group_num_] = acc_group_nano_num;
    data_capacity_ = acc_group_nano_num;

    delete[] group_key_num;
  }

  static inline unsigned short hashcode2B1(T x) {
    uint64_t llx = (unsigned long long)x;
    llx = (~llx) + (llx << 21); // llx = (llx << 21) - llx - 1;
    llx = llx ^ (llx >> 24);

    return (unsigned short)(llx&0x0ffffULL);
  }

  static inline unsigned short hashcode2B2(T x) {
    uint64_t llx = (unsigned long long)x;
    llx ^= llx >> 33;
    llx *= 0xff51afd7ed558ccdULL;

    return (unsigned short)(llx&0x0ffffULL);
  }

  void bulk_load(const V values[], int num_keys) {
    // 1. set group_num_ and min_key_
    this->group_num_ = static_cast<int>(std::ceil((double)num_keys / (NANO_INIT_KEY_NUM * expect_m)));
    this->min_key_ = values[0].first;

    // 2. Build model and adjust model intercept and data_capacity
    build_model(values, num_keys, &(this->model_));
    this->model_.expand(static_cast<double>(this->group_num_) / num_keys);

    // 3. calculate group offset for each group and allocate space 
    calculate_group_offsets(values, num_keys);
    initialize();

    // 4. Model-based inserts each key
    for (int i = 0; i < num_keys; i++) {
      int group_idx = get_key_group(values[i].first);
      int group_size = get_group_size(group_idx);
      int bulk_load_idx = 0;
      if (group_size >= two_choice_threashold) {
        int bulk_load_idx1 = get_idx_hash1(group_idx, group_size, values[i].first);
        int bulk_load_idx2 = get_idx_hash2(group_idx, group_size, values[i].first);

        if (!nano_slots_[bulk_load_idx1].isFull()) {
          bulk_load_idx = bulk_load_idx1;
        } else {
          bulk_load_idx = bulk_load_idx2;
        }
      } else { // m < 8
        bulk_load_idx = get_idx_hash1(group_idx, group_size, values[i].first);
      }

      int insert_ret = nano_slots_[bulk_load_idx].insertInNanoWithoutMoving(values[i].first, values[i].second);

      if (insert_ret == -1) {
        nano_slots_[bulk_load_idx].setOverflow();
        nano_type * cur_nano = get_of_nano(group_idx);
        insert_ret = cur_nano->insertInNanoWithoutMoving(values[i].first, values[i].second);
      }
    }
  }

  static void build_model(const V* values, int num_keys, LinearModel<T>* model) {
    LinearModelBuilder<T> builder(model);
    for (int i = 0; i < num_keys; i++) {
      builder.add(values[i].first, i);
    }
    builder.build();
  }

  /*** Lookup ***/
  inline int get_key_group(const T& key) {
    int position = this->model_.predict(key);
    int group_idx = std::max<int>(std::min<int>(position, this->group_num_ - 1), 0);
    return group_idx;
  }

  inline int get_group_size(int group_idx) {
    __builtin_prefetch(&(group_offsets_[group_idx]), 0, 1);
    return group_offsets_[group_idx + 1] - group_offsets_[group_idx];
  }

  inline int get_idx_hash1(int group_idx, int group_size, T key) {
    unsigned short hash_in_group1 = hashcode2B1(key) % group_size;
    return group_offsets_[group_idx] + hash_in_group1;
  }

  inline int get_idx_hash2(int group_idx, int group_size, T key) {
    unsigned short hash_in_group2 = hashcode2B2(key) % group_size;
    return group_offsets_[group_idx] + hash_in_group2;
  }

  int search_while_migrating(const T& key, P*& result) {
    int group_idx = get_key_group(key);
    int group_size = get_group_size(group_idx);
    nano_type * cur_nano = nullptr;
#ifdef USING_LOCK
    int migration_status = __atomic_load_n(&migration_bitmap_[group_idx], __ATOMIC_ACQUIRE);
#else
    int migration_status = migration_bitmap_[group_idx];
#endif
    // group not migrate yet
    if (migration_status == 0) {
      if (group_size >= two_choice_threashold) {
        int search_idx1 = get_idx_hash1(group_idx, group_size, key);
        int search_idx2 = get_idx_hash2(group_idx, group_size, key);

        cur_nano = get_nano(search_idx1);
        cur_nano->prefetchNano();
        int pos_in_nano1 = cur_nano->searchInNano(key);
        if (pos_in_nano1 >= 0) {
#ifdef USING_LOCK
          int new_migration_status = __atomic_load_n(&migration_bitmap_[group_idx], __ATOMIC_ACQUIRE);
          if (migration_status == new_migration_status) { // all zero
            result = &(cur_nano->ch(pos_in_nano1));
            return 0;
          }
          return 1;
#else 
          result = &(cur_nano->ch(pos_in_nano1));
          return 0;
#endif
          
        } else {
          cur_nano = get_nano(search_idx2);
          cur_nano->prefetchNano();
          int pos_in_nano2 = cur_nano->searchInNano(key);
          if (pos_in_nano2 >= 0) {
#ifdef USING_LOCK
            int new_migration_status = __atomic_load_n(&migration_bitmap_[group_idx], __ATOMIC_ACQUIRE);
            if (migration_status == new_migration_status) { // all zero
              result = &(cur_nano->ch(pos_in_nano2));
              return 0;
            }
            return 1;
#else 
            result = &(cur_nano->ch(pos_in_nano2));
            return 0;
#endif
          }

        }
      } else {
        int search_idx1 = get_idx_hash1(group_idx, group_size, key);
        cur_nano = get_nano(search_idx1);
        cur_nano->prefetchNano();
        int pos_in_nano1 = cur_nano->searchInNano(key);
        if (pos_in_nano1 >= 0) {
#ifdef USING_LOCK
          int new_migration_status = __atomic_load_n(&migration_bitmap_[group_idx], __ATOMIC_ACQUIRE);
          if (migration_status == new_migration_status) { // all zero
            result = &(cur_nano->ch(pos_in_nano1));
            return 0;
          }
          return 1;
#else
          result = &(cur_nano->ch(pos_in_nano1));
          return 0;
#endif
        }
      }

      // search in overflow nano
      cur_nano = get_of_nano(group_idx);
      cur_nano->prefetchNano();
      int pos_in_ofnano = cur_nano->searchInNano(key);

      if (pos_in_ofnano >= 0) {
#ifdef USING_LOCK
        int new_migration_status = __atomic_load_n(&migration_bitmap_[group_idx], __ATOMIC_ACQUIRE);
        if (migration_status == new_migration_status) { // all zero
          result = &(cur_nano->ch(pos_in_ofnano));
          return 0;
        }
        return 1;
#else
        result = &(cur_nano->ch(pos_in_ofnano));
        return 0;
#endif
      }
    } 
#ifdef USING_LOCK
    else if (migration_status == 1) {
      return 1;
    }
#endif
    else { // if migration_status == 2
      if (!target2_ || group_idx < split_point_) {
        int ret = target_->search(key, result);
        return ret;
      } else {
        int ret = target2_->search(key, result);
        return ret;
      }
    }
    return 0;
  }

  int search(const T& key, P*& result) {
#ifdef USING_LOCK
    uint32_t leaf_version;
    bool leaf_is_locked = test_leaf_lock_set(leaf_version);
    if (leaf_is_locked) {
      return 1;
    }
#endif
    // 1. get the nano in nano_slots_[] and search the nano
    int group_idx = get_key_group(key);
    int group_size = get_group_size(group_idx);
    nano_type * cur_nano = nullptr;

#ifdef USING_LOCK
    uint32_t group_version;
    bool group_is_locked = test_group_lock_set(group_idx, group_version);
    if (group_is_locked) {
      return 1;
    }
#endif

    if (group_size >= two_choice_threashold) {
      int search_idx1 = get_idx_hash1(group_idx, group_size, key);
      int search_idx2 = get_idx_hash2(group_idx, group_size, key);

      cur_nano = get_nano(search_idx1);
      cur_nano->prefetchNano();
      int pos_in_nano1 = cur_nano->searchInNano(key);
      if (pos_in_nano1 >= 0) {
#ifdef USING_LOCK
        if (test_group_lock_version_change(group_idx, group_version)) {
          return 1;
        } 
#endif
        result = &(cur_nano->ch(pos_in_nano1));
        return 0;
      } 
      if (search_idx2 != search_idx1) {
        cur_nano = get_nano(search_idx2);
        cur_nano->prefetchNano();
        int pos_in_nano2 = cur_nano->searchInNano(key);
        if (pos_in_nano2 >= 0) {
#ifdef USING_LOCK
        if (test_group_lock_version_change(group_idx, group_version)) {
          return 1;
        } 
#endif
          result = &(cur_nano->ch(pos_in_nano2));
          return 0;
        }
      }
    } else {
      int search_idx1 = get_idx_hash1(group_idx, group_size, key);
      cur_nano = get_nano(search_idx1);
      cur_nano->prefetchNano();
      int pos_in_nano1 = cur_nano->searchInNano(key);
      if (pos_in_nano1 >= 0) {
#ifdef USING_LOCK
        if (test_group_lock_version_change(group_idx, group_version)) {
          return 1;
        } 
#endif
        result = &(cur_nano->ch(pos_in_nano1));
        return 0;
      }
    }

    // overflow nano
    cur_nano = get_of_nano(group_idx);
    cur_nano->prefetchNano();
    int pos_in_ofnano = cur_nano->searchInNano(key);
    
    if (pos_in_ofnano >= 0) {
#ifdef USING_LOCK
      if (test_group_lock_version_change(group_idx, group_version)) {
        return 1;
      } 
#endif
      result = &(cur_nano->ch(pos_in_ofnano));
    }
    return 0;
  }

  int search_debug(const T& key, P*& result) {
    // 1. get the nano in nano_slots_[] and search the nano
    int group_idx = get_key_group(key);
    int group_size = get_group_size(group_idx);

    nano_type * cur_nano = nullptr;
    if (group_size >= two_choice_threashold) {
      int search_idx1 = get_idx_hash1(group_idx, group_size, key);
      int search_idx2 = get_idx_hash2(group_idx, group_size, key);
      
      cur_nano = get_nano(search_idx1);
      int pos_in_nano1 = cur_nano->searchInNano(key);
      // cur_nano->printNano();
      if (pos_in_nano1 >= 0) {
        result = &(cur_nano->ch(pos_in_nano1));
        return 0;
      } else {
        cur_nano = get_nano(search_idx2);

        int pos_in_nano2 = cur_nano->searchInNano(key);
        // cur_nano->printNano();
        if (pos_in_nano2 >= 0) {
          result = &(cur_nano->ch(pos_in_nano2));
          return 0;
        }
      }
    } else {
      // std::cout << "group size: " << group_size << std::endl;
      int search_idx1 = get_idx_hash1(group_idx, group_size, key);
      cur_nano = get_nano(search_idx1);
      int pos_in_nano1 = cur_nano->searchInNano(key);
      // cur_nano->printNano();
      if (pos_in_nano1 >= 0) {
        result = &(cur_nano->ch(pos_in_nano1));
        return 0;
      }
    }

    if (cur_nano->getOverflow()) {
      int pos_in_ofnano = -1;
      cur_nano = get_of_nano(group_idx);
      cur_nano->prefetchNano();
      pos_in_ofnano = cur_nano->searchInNano(key);

      // cur_nano->printNano();
      if (pos_in_ofnano >= 0) {
        result = &(cur_nano->ch(pos_in_ofnano));
        return 0;
      }
    }
    return 0;
  }

  uint32_t get_max_group_size(int begin_idx, int end_idx) {
    uint32_t max_m = 0;
    for (int i = begin_idx; i < end_idx; i++) {
      max_m = std::max(max_m, uint32_t(group_offsets_[i+1] - group_offsets_[i]));
    }
    return max_m;
  }

  int segments_transform (int * & seg_key_nums, T * & start_keys, T * & end_keys, long double * & slopes, long double * & intercepts, int seg_num, int orig_error_bound, int target_error_bound, std::vector<int> & global_seg_num) {
    int target_seg_num = 0;
    uint64_t orig_x = start_keys[0];
    int orig_y = 0;
    int cumulated_key_num = 0;
    int seg_cnt = 0;

    uint64_t cur_first_key = start_keys[0];
    long double first_slope = slopes[0];
    long double first_intercept = intercepts[0];
    int seg_key_num = seg_key_nums[0];
    uint64_t cur_last_key = end_keys[0];

    int delta_error_bound = target_error_bound - orig_error_bound;

    long double new_slope_lower = (long double)(first_slope * cur_last_key + first_intercept - delta_error_bound - orig_y) /(cur_last_key - orig_x);
    long double new_slope_upper = (long double)(first_slope * cur_last_key + first_intercept + delta_error_bound - orig_y) /(cur_last_key - orig_x);

    cumulated_key_num += seg_key_num;
    seg_cnt++;
    for (int i = 1; i < seg_num; i++) {
      cur_first_key = start_keys[i];;
      seg_key_num = seg_key_nums[i];
      cur_last_key = end_keys[i];

      long double cur_slope = slopes[i];
      long double cur_intercept = intercepts[i];

      long double cur_slope_lower_numerator = cur_slope * orig_x + cur_intercept - orig_y - delta_error_bound;
      long double cur_slope_upper_numerator = cur_slope * orig_x + cur_intercept - orig_y + delta_error_bound;

      long double cur_new_slope_lower;
      long double cur_new_slope_upper;

      if (cur_slope_lower_numerator < 0) {
        cur_new_slope_lower = cur_slope + (cur_slope_lower_numerator / (cur_last_key - orig_x));
      } else {
        cur_new_slope_lower = cur_slope + (cur_slope_lower_numerator / (cur_first_key - orig_x));
      }

      if (cur_slope_upper_numerator < 0) {
        cur_new_slope_upper = cur_slope + (cur_slope_upper_numerator / (cur_first_key - orig_x));
      } else {
        cur_new_slope_upper = cur_slope + (cur_slope_upper_numerator / (cur_last_key - orig_x));
      }

      new_slope_lower = std::max(new_slope_lower, cur_new_slope_lower);
      new_slope_upper = std::min(new_slope_upper, cur_new_slope_upper);

      if (new_slope_upper < new_slope_lower) {
        global_seg_num.emplace_back(seg_cnt);
        seg_cnt = 0;
        // violate the error bound, count and reset
        target_seg_num++;

        orig_x = cur_first_key;
        orig_y = cumulated_key_num;

        new_slope_lower = (long double)(cur_slope * cur_last_key + cur_intercept - delta_error_bound - orig_y) / (cur_last_key - orig_x);
        new_slope_upper = (long double)(cur_slope * cur_last_key + cur_intercept + delta_error_bound - orig_y) / (cur_last_key - orig_x);
      }

      cumulated_key_num += seg_key_num;
      seg_cnt++;
    }
    target_seg_num++;
    global_seg_num.emplace_back(seg_cnt);
    return target_seg_num;
  }


  // -1: smo already begun

  // 0: expand & group expand
  // 1: expand & group split

  // 2: split & two nodes group expand
  // 3: split & first node expand & second group split

  // 4: split & first node split & second group expand
  // 5: split & two nodes group split

  int decide_smo_method() {
    // already begun
    if (this->migration_bitmap_ != nullptr) {
      return -1;
    }

    // only one group, expand
    if (group_num_ == 1) {
      auto new_max_m = 2 * get_max_group_size(0, group_num_);
      if (new_max_m <= max_m) {
        return 0;
      }
      return 1;
    }

    // too large after expand, split
    if (this->data_capacity_ > nano_num_limit / 2) {
      split_point_ = group_num_ / 2;

      auto new_left_max_m = 2 * get_max_group_size(0, split_point_);
      auto new_right_max_m = 2 * get_max_group_size(split_point_, group_num_);

      if (new_left_max_m <= max_m) {
        if (new_right_max_m <= max_m) {
          return 2;
        } else {
          return 3;
        }
      } else {
        if (new_right_max_m <= max_m) {
          return 4;
        } else {
          return 5;
        }
      }
      return 0;
    } 

    // segmentation transform algorithm
    double load_factor_when_smo = 0.8;
    int * seg_key_nums = new int[group_num_];
    T * start_keys = new T[group_num_];
    T * end_keys = new T[group_num_];
    long double * slopes = new long double[group_num_];
    long double * intercepts = new long double[group_num_];
    T * nano_keys = new T[NANO_KEY_NUM]();
    int start_idx = 0;
    int sum_group_error_bound = 0;
    for (int i = 0; i < group_num_; i++) {
      int group_size = get_group_size(i);
      seg_key_nums[i] = group_size * NANO_INIT_KEY_NUM * load_factor_when_smo;
      start_keys[i] = model_.predict_reverse(i);
      end_keys[i] = model_.predict_reverse(i+1) - 1;
      slopes[i] = (long double)seg_key_nums[i] / (end_keys[i] - start_keys[i]);
      intercepts[i] = start_idx - slopes[i] * start_keys[i];

      nano_type * sample_nano = get_nano(group_offsets_[i]);
      int k = 0;
      if (!(sample_nano->isEmpty())) {
        uint16_t bm = sample_nano->bitmap;
        while (bm) {
          int jj = bitScan(bm)-1;  // next candidate
          T cur_key = sample_nano->ck(jj);
          nano_keys[k] = cur_key;
          k++;
          bm &= ~(0x1<<jj);  // remove this bit
        }
      }
      std::sort(nano_keys, nano_keys + k);
      int cur_eb_max = 0;
      for (int j = 0; j < k; j++) {
        int predicted_pos = slopes[i] * nano_keys[j] + intercepts[i];
        int estimated_pos = start_idx + (double)j/k * seg_key_nums[i];
        cur_eb_max = std::max(std::abs(predicted_pos - estimated_pos), cur_eb_max);
      }
      sum_group_error_bound += cur_eb_max;
      start_idx += seg_key_nums[i];
    }
    int group_error_bound = sum_group_error_bound / group_num_;

    std::vector<int> leaf_seg_group_num;
    auto new_leaf_num = segments_transform(seg_key_nums, start_keys, end_keys, slopes, intercepts, 
                                           group_num_, group_error_bound, seg_error_bound, leaf_seg_group_num);

    delete [] seg_key_nums;
    delete [] start_keys;
    delete [] end_keys;
    delete [] slopes;
    delete [] intercepts;
    delete [] nano_keys;

    // time_sec4 = tn_index.rdtsc();
    bool empty_half = false;
    if (new_leaf_num > 1) {
      int mid_seg_idx = std::round(new_leaf_num / 2.);
      for (int i = 0; i < mid_seg_idx; i++) {
        split_point_ += leaf_seg_group_num[i];
      }
    }

    if (new_leaf_num == 1 || split_point_ == 0 || split_point_ == group_num_) {
      split_point_ = 0;
      auto new_max_m = 2 * get_max_group_size(0, group_num_);
      if (new_max_m <= max_m) {
        return 0;
      }
      return 1;
    }

    auto new_left_max_m = 2 * get_max_group_size(0, split_point_);
    auto new_right_max_m = 2 * get_max_group_size(split_point_, group_num_);

    if (new_left_max_m <= max_m) {
      if (new_right_max_m <= max_m) {
        return 2;
      } else {
        return 3;
      }
    } else {
      if (new_right_max_m <= max_m) {
        return 4;
      } else {
        return 5;
      }
    }

    return 2;
  }

  void clear_group(int group_idx) {
    int nano_start = group_offsets_[group_idx];
    void * start_addr = static_cast<void *>(&(nano_slots_[nano_start]));
    memset(start_addr, 0, sizeof(nano_type) * get_group_size(group_idx));
    int of_nano_start = group_idx * group_overflow_nano_num;
    start_addr = static_cast<void *>(get_of_nano(of_nano_start));
    memset(start_addr, 0, sizeof(nano_type) * group_overflow_nano_num);
  }

  int clear_2_groups(int group_idx) {
    int nano_start = group_offsets_[group_idx];
    void * start_addr = static_cast<void *>(&(nano_slots_[nano_start]));
    int nano_num = group_offsets_[group_idx + 2] - nano_start;
    memset(start_addr, 0, sizeof(nano_type) * nano_num);
    int of_nano_start = group_idx * group_overflow_nano_num;
    start_addr = static_cast<void *>(get_of_nano(of_nano_start));
    memset(start_addr, 0, sizeof(nano_type) * 2 * group_overflow_nano_num);
    return nano_num;
  }

  void migrate_group_expand(int group_idx) {
    migrate_num++;
    target_->clear_group(group_idx);

    int nano_start = group_offsets_[group_idx];
    int nano_end = group_offsets_[group_idx + 1];
    // nano[]
    for (int i = nano_start; i < nano_end; i++) {
      nano_type * cur_nano = get_nano(i);
      if (!(cur_nano->isEmpty())) {
        uint16_t bm = cur_nano->bitmap;
        while (bm) {
          int jj = bitScan(bm)-1;  // next candidate
          T cur_key = cur_nano->ck(jj);
          P cur_payload = cur_nano->cch(jj);
          auto insert_ret = target_->insert_without_lock(cur_key, cur_payload);
          bm &= ~(0x1<<jj);  // remove this bit
        }
      }
    }

    int of_nano_start = group_idx * group_overflow_nano_num;
    int of_nano_end = of_nano_start + group_overflow_nano_num;
    for (int i = of_nano_start; i < of_nano_end; i++) {
      nano_type *of_nano = get_of_nano(i);
      if (!(of_nano->isEmpty())) {
        uint16_t bm = of_nano->bitmap;
        while (bm) {
          int jj = bitScan(bm)-1;  // next candidate
          T cur_key = of_nano->ck(jj);
          P cur_payload = of_nano->cch(jj);
          auto insert_ret = target_->insert_without_lock(cur_key, cur_payload);
          bm &= ~(0x1<<jj);  // remove this bit
        }
      }
    }
  }

  void migrate_group_split(int group_idx) {
    migrate_num++;
    int target_total_nano_num = target_->clear_2_groups(2 * group_idx);

    int nano_start = group_offsets_[group_idx];
    int nano_end = group_offsets_[group_idx + 1];

    V * group_keys = new V[target_total_nano_num * NANO_KEY_NUM];
    int left_num = 0;
    int total_num = 0;

    // nano[]
    for (int i = nano_start; i < nano_end; i++) {
      nano_type * cur_nano = get_nano(i);
      if (!(cur_nano->isEmpty())) {
        uint16_t bm = cur_nano->bitmap;
        while (bm) {
          int jj = bitScan(bm)-1;  // next candidate
          T cur_key = cur_nano->ck(jj);
          P cur_payload = cur_nano->cch(jj);
          if (target_->get_key_group(cur_key) <= 2 * group_idx) {
            left_num++;
          }
          group_keys[total_num] = std::make_pair(cur_key, cur_payload);
          total_num++;
          bm &= ~(0x1<<jj);  // remove this bit
        }
      }
    }

    int of_nano_start = group_idx * group_overflow_nano_num;
    int of_nano_end = of_nano_start + group_overflow_nano_num;
    for (int i = of_nano_start; i < of_nano_end; i++) {
      nano_type *of_nano = get_of_nano(i);
      if (!(of_nano->isEmpty())) {
        uint16_t bm = of_nano->bitmap;
        while (bm) {
          int jj = bitScan(bm)-1;  // next candidate
          T cur_key = of_nano->ck(jj);
          P cur_payload = of_nano->cch(jj);
          if (target_->get_key_group(cur_key) <= 2 * group_idx) {
            left_num++;
          } 
          group_keys[total_num] = std::make_pair(cur_key, cur_payload);
          total_num++;
          bm &= ~(0x1<<jj);  // remove this bit
        }
      }
    }

    double first_group_size = std::min(std::max(1., std::round(((double)left_num / total_num) * target_total_nano_num)), (double)(target_total_nano_num - 1));
    target_->group_offsets_[2 * group_idx + 1] = target_->group_offsets_[2 * group_idx] + first_group_size;

    for (int i = 0; i < total_num; i ++){
      auto insert_ret = target_->insert_without_lock(group_keys[i].first, group_keys[i].second);
    }
    delete [] group_keys;
  }

  void migrate_group_right_expand(int group_idx) {

    migrate_num++;
    target2_->clear_group(group_idx - split_point_);

    int nano_start = group_offsets_[group_idx];
    int nano_end = group_offsets_[group_idx + 1];
    // nano[]
    for (int i = nano_start; i < nano_end; i++) {
      nano_type * cur_nano = get_nano(i);
      if (!(cur_nano->isEmpty())) {
        uint16_t bm = cur_nano->bitmap;
        while (bm) {
          int jj = bitScan(bm)-1;  // next candidate
          T cur_key = cur_nano->ck(jj);
          P cur_payload = cur_nano->cch(jj);
          auto insert_ret = target2_->insert_without_lock(cur_key, cur_payload);
          bm &= ~(0x1<<jj);  // remove this bit
        }
      }
    }

    int of_nano_start = group_idx * group_overflow_nano_num;
    int of_nano_end = of_nano_start + group_overflow_nano_num;
    for (int i = of_nano_start; i < of_nano_end; i++) {
      nano_type *of_nano = get_of_nano(i);
      if (!(of_nano->isEmpty())) {
        uint16_t bm = of_nano->bitmap;
        while (bm) {
          int jj = bitScan(bm)-1;  // next candidate
          T cur_key = of_nano->ck(jj);
          P cur_payload = of_nano->cch(jj);
          auto insert_ret = target2_->insert_without_lock(cur_key, cur_payload);
          bm &= ~(0x1<<jj);  // remove this bit
        }
      }
    }
  }

  void migrate_group_right_split(int group_idx) {
    migrate_num++;
    int left_idx_in_target2 = 2 * (group_idx - split_point_);
    int target_total_nano_num = target2_->clear_2_groups(left_idx_in_target2);

    V * group_keys = new V[target_total_nano_num * NANO_KEY_NUM];
    int left_num = 0;
    int total_num = 0;

    int nano_start = group_offsets_[group_idx];
    int nano_end = group_offsets_[group_idx + 1];
    // nano[]
    for (int i = nano_start; i < nano_end; i++) {
      nano_type * cur_nano = get_nano(i);
      if (!(cur_nano->isEmpty())) {
        uint16_t bm = cur_nano->bitmap;
        while (bm) {
          int jj = bitScan(bm)-1;  // next candidate
          T cur_key = cur_nano->ck(jj);
          P cur_payload = cur_nano->cch(jj);
          if (target2_->get_key_group(cur_key) <= left_idx_in_target2) {
            left_num++;
          } 
          group_keys[total_num] = std::make_pair(cur_key, cur_payload);
          total_num++;
          bm &= ~(0x1<<jj);  // remove this bit
        }
      }
    }

    int of_nano_start = group_idx * group_overflow_nano_num;
    int of_nano_end = of_nano_start + group_overflow_nano_num;
    for (int i = of_nano_start; i < of_nano_end; i++) {
      nano_type *of_nano = get_of_nano(i);
      if (!(of_nano->isEmpty())) {
        uint16_t bm = of_nano->bitmap;
        while (bm) {
          int jj = bitScan(bm)-1;  // next candidate
          T cur_key = of_nano->ck(jj);
          P cur_payload = of_nano->cch(jj);
          if (target2_->get_key_group(cur_key) <= left_idx_in_target2) {
            left_num++;
          } 
          group_keys[total_num] = std::make_pair(cur_key, cur_payload);
          total_num++;
          bm &= ~(0x1<<jj);  // remove this bit
        }
      }
    }

    double first_group_size = std::min(std::max(1., std::round(((double)left_num / total_num) * target_total_nano_num)), (double)(target_total_nano_num - 1));
    target2_->group_offsets_[left_idx_in_target2 + 1] = target2_->group_offsets_[left_idx_in_target2] + first_group_size;
    for (int i = 0; i < total_num; i ++){
      auto insert_ret = target2_->insert_without_lock(group_keys[i].first, group_keys[i].second);
    }
    delete [] group_keys;
  } 

  // insert in migrating node
  int insert_while_migrating(const T& key, const P& payload) {
#ifdef USING_LOCK
    uint32_t leaf_version;
    bool leaf_is_locked = test_leaf_lock_set(leaf_version);
    if (leaf_is_locked) {
      return 1;
    }
#endif

    int group_idx = get_key_group(key);
    int group_size = get_group_size(group_idx);

    // expanding node
    if (target2_ == nullptr) {
      uint32_t migration_status = __atomic_load_n(&migration_bitmap_[group_idx], __ATOMIC_ACQUIRE);
      if (migration_status == 0) {
        uint32_t not_migrate = 0;
        auto ret = CAS(&migration_bitmap_[group_idx], &not_migrate, 1);
        if (ret) { // CAS success
          // migrate the group
          if (target_->group_num_ == group_num_) {
            migrate_group_expand(group_idx);
          } else {
            migrate_group_split(group_idx);
          }

          // insert key & value for this very operation
          auto insert_ret = target_->insert_without_lock(key, payload);
          migration_bitmap_[group_idx] = 2;

          // check if all groups are done
          for (int i = 0; i < group_num_; i++) {
            if (migration_bitmap_[i] != 2) {
              return 0;
            }
          }
          return 3;
        } 
        // CAS fail, retry
        return 1;
      } else if (migration_status == 1) { // during migration, retry
        return 1;
      } else {// migration_status == 2
        auto target_insert_ret = target_->insert(key, payload);
        if (target_insert_ret == 1) {
          return 1;
        }

        // help to migrate a group 
        // find a group
        int help_group_idx = 0;
        while (migration_bitmap_[help_group_idx]!= 0 && help_group_idx < group_num_) {
          help_group_idx++;
        }

        // migrate the group
        if (help_group_idx < group_num_) {
          uint32_t not_migrate = 0;
          auto ret = CAS(&migration_bitmap_[help_group_idx], &not_migrate, 1);
          if (ret) { // CAS success
            if (target_->group_num_ == group_num_) {
              migrate_group_expand(help_group_idx);
            } else {
              migrate_group_split(help_group_idx);
            }
            migration_bitmap_[help_group_idx] = 2;
            // if this is the last group
            for (int i = 0; i < group_num_; i++) {
              if (migration_bitmap_[i] != 2) {
                if (target_insert_ret == 0) {
                  return target_insert_ret;
                } 
                return 1;
              }
            }
            return 3;
          }
        }
        return target_insert_ret;
      }

    // spliting node
    } else {
      uint32_t migration_status = __atomic_load_n(&migration_bitmap_[group_idx], __ATOMIC_ACQUIRE);
      if (migration_status == 0) {
        uint32_t not_migrate = 0;
        auto ret = CAS(&migration_bitmap_[group_idx], &not_migrate, 1);
        if (ret) { // CAS success
          // migrate the group
          if (group_idx < split_point_) {
            if (target_->group_num_ == split_point_) {
              migrate_group_expand(group_idx);
            } else {
              migrate_group_split(group_idx);
            }

            // insert key & value for this very operation
            auto insert_ret = target_->insert_without_lock(key, payload);
          } else {
            if (target2_->group_num_ == (group_num_ - split_point_)) {
              migrate_group_right_expand(group_idx);
            } else {
              migrate_group_right_split(group_idx);
            }

            // insert key & value for this very operation
            auto insert_ret = target2_->insert_without_lock(key, payload);
            while (key < target2_->min_key_) {
              T old_min_key = target2_->min_key_;
              CAS(&(target2_->min_key_), &old_min_key, key);
            }
          }
          migration_bitmap_[group_idx] = 2;

          // check if all groups are done
          for (int i = 0; i < group_num_; i++) {
            if (migration_bitmap_[i] != 2) {
              return 0;
            }
          }
          return 3;
        } else { // CAS fail, retry
          return 1;
        }
      } else if (migration_status == 1) { // during migration, retry
        return 1;
      } else {// migration_status == 2
        // try to insert the key first
        int target_insert_ret;
        if (group_idx < split_point_) {
          target_insert_ret = target_->insert(key, payload);
        } else {
          target_insert_ret = target2_->insert(key, payload);
          if (target_insert_ret == 0) {
            // update the new key if neccessary
            while (key < target2_->min_key_) {
              T old_min_key = target2_->min_key_;
              CAS(&(target2_->min_key_), &old_min_key, key);
            }
          }
        }
        if (target_insert_ret == 1) {
          return 1;
        }
        // insert success, help to migrate a group
        // find a group
        int help_group_idx = 0;
        while (migration_bitmap_[help_group_idx]!= 0 && help_group_idx < group_num_) {
          help_group_idx++;
        }

        // migrate the group
        if (help_group_idx < group_num_) {
          uint32_t not_migrate = 0;
          auto ret = CAS(&migration_bitmap_[help_group_idx], &not_migrate, 1);
          if (ret) { // CAS success
            if (help_group_idx < split_point_) {
              if (target_->group_num_ == split_point_) {
                migrate_group_expand(help_group_idx);
              } else {
                migrate_group_split(help_group_idx);
              }
            } else {
              if (target2_->group_num_ == (group_num_ - split_point_)) {
                migrate_group_right_expand(help_group_idx);
              } else {
                migrate_group_right_split(help_group_idx);
              }
            }
            migration_bitmap_[help_group_idx] = 2;
            // if this is the last group
            for (int i = 0; i < group_num_; i++) {
              if (migration_bitmap_[i] != 2) {
                if (target_insert_ret == 0) {
                  return target_insert_ret;
                } 
                return 1;
              }
            }
            return 3;
          }
        }
        return target_insert_ret;
      }
    }
    return 0;
  }

  // case 2 & 3: normal & preparing node 
  int insert(const T& key, const P& payload) {
    // 0. Check if the leaf is locked
#ifdef USING_LOCK
    uint32_t leaf_version;
    bool leaf_is_locked = test_leaf_lock_set(leaf_version);
    if (leaf_is_locked) {
      return 1;
    }
#endif

    int group_idx = get_key_group(key);
    int group_size = get_group_size(group_idx);
    int insert_idx = 0;

    if (group_size >= two_choice_threashold) {
      int insert_idx1 = get_idx_hash1(group_idx, group_size, key);
      int insert_idx2 = get_idx_hash2(group_idx, group_size, key);

      if (!nano_slots_[insert_idx1].isFull()) {
        insert_idx = insert_idx1;
        nano_type * check_nano = get_nano(insert_idx2);
        check_nano->prefetchNano();
        auto pos = check_nano->searchInNano(key);
        if (pos >= 0) { // duplicate found
          return -1;
        }
      } else {
        insert_idx = insert_idx2;
        nano_type * check_nano = get_nano(insert_idx1);
        check_nano->prefetchNano();
        auto pos = check_nano->searchInNano(key);
        if (pos >= 0) { // duplicate found
          return -1;
        }
      }
    } else {
      insert_idx = get_idx_hash1(group_idx, group_size, key);
    }

#ifdef USING_LOCK
    bool get_lock_success = try_get_group_lock(group_idx);
    if (!get_lock_success) {
      return 1;
    }
#endif

    nano_type * cur_nano = get_nano(insert_idx);
    cur_nano->prefetchNano();
    if (cur_nano->getOverflow()) {
      nano_type * check_nano = get_of_nano(group_idx);
      check_nano->prefetchNano();
      auto pos = check_nano->searchInNano(key);
      if (pos >= 0) { // duplicate found
        return -1;
      }
    }
    int insert_res = cur_nano->insertInNanoWithoutMoving(key, payload);

    // 3. if duplicate found, return
    if (insert_res == -2) { // duplicate found
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
      return -1;
    }

    // 4. if nano insertion success, unlock and return
    if (insert_res >= 0) {
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
      return 0;
    }

    // 4. if nano is full, try to insert in overflow nano
    if (insert_res == -1) { 
      // 4.1 find related overflow nano and set overflow flag
      cur_nano->setOverflow();
      cur_nano = get_of_nano(group_idx);
      // cur_nano->prefetchNano();
      int of_insert_res = cur_nano->insertInNanoWithoutMoving(key, payload);

      if (of_insert_res == -2) {
#ifdef USING_LOCK
        release_group_lock(group_idx);
#endif
        return -1;
      } 
      
      if (of_insert_res >= 0) {
#ifdef USING_LOCK
        release_group_lock(group_idx);
#endif
        return 0;
      }

      if (of_insert_res == -1) { 
#ifdef USING_LOCK
        release_group_lock(group_idx);
#endif

        return 2;
      }
    } 
#ifdef USING_LOCK
    release_group_lock(group_idx);
#endif
    return 0;
  }

  // 0: insertion success
  // 1: insertion failed
  int insert_without_lock(const T& key, const P& payload) {
    int group_idx = get_key_group(key);
    int group_size = get_group_size(group_idx);

    int insert_idx = 0;
    if (group_size >= two_choice_threashold) {
      int insert_idx1 = get_idx_hash1(group_idx, group_size, key);
      int insert_idx2 = get_idx_hash2(group_idx, group_size, key);

      if (!nano_slots_[insert_idx1].isFull()) {
        insert_idx = insert_idx1;
      } else {
        insert_idx = insert_idx2;
      }
    } else {
      insert_idx = get_idx_hash1(group_idx, group_size, key);
    }

    nano_type * cur_nano = get_nano(insert_idx);
    cur_nano->prefetchNano();
    int insert_res = cur_nano->insertInNanoWithoutMoving(key, payload);

    // 4. if nano insertion success, unlock and return
    if (insert_res >= 0) {
      return 0;
    }

    if (insert_res == -2) {
      return 0;
    }
    // 4. if nano is full, try to insert in overflow nano
    if (insert_res == -1) { // 
      // 4.1 find related overflow nano and set overflow flag
      cur_nano->setOverflow();
      cur_nano = get_of_nano(group_idx);
      cur_nano->prefetchNano();
      int of_insert_res = cur_nano->insertInNanoWithoutMoving(key, payload);

      if (of_insert_res >= 0) {
        return 0;
      }

      if (of_insert_res == -2) {
        return 0;
      }
    } 
    return 1;
  }

  T find_min_key_of_target2() {
    // make sure target2 is not null
    int group_idx = split_point_;
    
    T cur_min_key = std::numeric_limits<T>::max();
    while (group_idx < group_num_) {
      int nano_start = group_offsets_[group_idx];
      int nano_end = group_offsets_[group_idx + 1];
      
      // nano[]
      for (int i = nano_start; i < nano_end; i++) {
        nano_type * cur_nano = get_nano(i);
        if (!(cur_nano->isEmpty())) {
          cur_min_key = std::min(cur_min_key, cur_nano->getMinKey());
        }
      }

      int of_nano_start = group_idx * group_overflow_nano_num;
      int of_nano_end = of_nano_start + group_overflow_nano_num;
      for (int i = of_nano_start; i < of_nano_end; i++) {
        nano_type *of_nano = get_of_nano(i);
        if (!(of_nano->isEmpty())) {
          cur_min_key = std::min(cur_min_key, of_nano->getMinKey());
        }
      }

      if (cur_min_key != std::numeric_limits<T>::max()) {
        break;
      }

      group_idx++;
    }
    return cur_min_key;
  }

  T find_max_key_of_target() {
    int group_idx = split_point_-1;
    T cur_max_key = std::numeric_limits<T>::min();
    while (group_idx >= 0) {
      int nano_start = group_offsets_[group_idx];
      int nano_end = group_offsets_[group_idx + 1];

      // nano[]
      for (int i = nano_start; i < nano_end; i++) {
        nano_type * cur_nano = get_nano(i);
        if (!(cur_nano->isEmpty())) {
          cur_max_key = std::max(cur_max_key, cur_nano->getLargestKey());
        }
      }

      int of_nano_start = group_idx * group_overflow_nano_num;
      int of_nano_end = of_nano_start + group_overflow_nano_num;
      for (int i = of_nano_start; i < of_nano_end; i++) {
        nano_type *of_nano = get_of_nano(i);
        if (!(of_nano->isEmpty())) {
          cur_max_key = std::max(cur_max_key, of_nano->getLargestKey());
        }
      }
      if (cur_max_key != std::numeric_limits<T>::min()) {
        break;
      }
      group_idx--;
    }
    return cur_max_key;
  }


  void prepare_for_migration(int smo_method) {
    // 1. allocate & initialize migration bitmap
    migration_bitmap_ = new uint32_t[group_num_]();
    double expand_times = 2.;
    switch (smo_method)
    {
    case 0: 
    {
      // 2. allocatre new node
      self_type* new_data_node = new self_type();
      new_data_node->min_key_ = this->min_key_;
      new_data_node->group_num_ = this->group_num_;

      new_data_node->model_.a_ = this->model_.a_;
      new_data_node->model_.b_ = this->model_.b_;

      // expand each group and update group_offsets_[], data_capacity_
      new_data_node->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node->group_num_ + 1)));
      // new_data_node->group_offsets_ = new uint16_t[new_data_node->group_num_ + 1];
      int cur_group_start = 0;
      for (int i = 0; i < group_num_; i++) {
        new_data_node->group_offsets_[i] = cur_group_start;
        int cur_group_size = get_group_size(i);
        int new_group_size = cur_group_size * expand_times;
        cur_group_start += new_group_size;

      }
      new_data_node->group_offsets_[new_data_node->group_num_] = cur_group_start;
      new_data_node->data_capacity_ = cur_group_start;

      new_data_node->initialize_for_smo();

      // 4. set this node's target
      this->target_ = new_data_node;
      this->target2_ = nullptr;
      
      break;
    }

    case 1:
    {
      // 2. allocatre new node
      self_type* new_data_node = new self_type();
      new_data_node->min_key_ = this->min_key_;
      new_data_node->group_num_ = expand_times * (this->group_num_);

      new_data_node->model_.a_ = expand_times * this->model_.a_;
      new_data_node->model_.b_ = expand_times * this->model_.b_;

      // expand each group and update group_offsets_[], data_capacity_
      new_data_node->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node->group_num_ + 1)));
      int cur_group_start = 0;
      for (int i = 0; i < group_num_; i++) {
        new_data_node->group_offsets_[2 * i] = cur_group_start;
        int cur_group_size = get_group_size(i);
        int new_group_size = cur_group_size * expand_times;

        int new_group_first = new_group_size / 2;
        cur_group_start += new_group_first;
        new_data_node->group_offsets_[2 * i + 1] = cur_group_start;

        int new_group_second = new_group_size - new_group_first;
        cur_group_start += new_group_second;

      }
      new_data_node->group_offsets_[new_data_node->group_num_] = cur_group_start;
      new_data_node->data_capacity_ = cur_group_start;

      new_data_node->initialize_for_smo();

      // 4. set this node's target
      this->target_ = new_data_node;
      this->target2_ = nullptr;

      break;
    }
    case 2:
    {
      // malloc and set some parameters
      self_type* new_data_node1 = new self_type();
      new_data_node1->min_key_ = this->min_key_;
      new_data_node1->group_num_ = this->split_point_;

      self_type* new_data_node2 = new self_type();
      new_data_node2->min_key_ = this->find_min_key_of_target2();
      if (__glibc_unlikely(new_data_node2->min_key_ == std::numeric_limits<T>::max())) {
        new_data_node2->min_key_ = this->find_max_key_of_target() + 1;
      }
      new_data_node2->group_num_ = (this->group_num_ - this->split_point_);

      // set models
      new_data_node1->model_.a_ = (this->model_.a_);
      new_data_node1->model_.b_ = (this->model_.b_);

      new_data_node2->model_.a_ = (this->model_.a_);
      new_data_node2->model_.b_ = (this->model_.b_ - this->split_point_);

      // expand each group and update group_offsets_[], data_capacity_
      new_data_node1->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node1->group_num_ + 1)));
      new_data_node2->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node2->group_num_ + 1)));

      int cur_group_start = 0;
      for (int i = 0; i < split_point_; i++) {
        new_data_node1->group_offsets_[i] = cur_group_start;
        int cur_group_size = get_group_size(i);
        int new_group_size = cur_group_size * expand_times;
        cur_group_start += new_group_size;
      }
      new_data_node1->group_offsets_[new_data_node1->group_num_] = cur_group_start;
      new_data_node1->data_capacity_ = cur_group_start;

      cur_group_start = 0;
      for (int i = split_point_; i < group_num_; i++) {
        int idx_in_target2 = i - split_point_;
        new_data_node2->group_offsets_[idx_in_target2] = cur_group_start;
        int cur_group_size = this->get_group_size(i);
        int new_group_size = cur_group_size * expand_times;
        cur_group_start += new_group_size;
      }
      new_data_node2->group_offsets_[new_data_node2->group_num_] = cur_group_start;
      new_data_node2->data_capacity_ = cur_group_start;

      new_data_node1->initialize_for_smo();
      new_data_node2->initialize_for_smo();

      // 4. set this node's target
      this->target_ = new_data_node1;
      this->target2_ = new_data_node2;

      new_data_node1->next_leaf_ = new_data_node2;
      new_data_node2->prev_leaf_ = new_data_node1;
      break;
    }
    case 3:
    {
      // malloc and set some parameters
      self_type* new_data_node1 = new self_type();
      new_data_node1->min_key_ = this->min_key_;
      new_data_node1->group_num_ = this->split_point_;

      self_type* new_data_node2 = new self_type();
      new_data_node2->min_key_ = this->find_min_key_of_target2();
      if (__glibc_unlikely(new_data_node2->min_key_ == std::numeric_limits<T>::max())) {
        new_data_node2->min_key_ = this->find_max_key_of_target() + 1;
      }
      new_data_node2->group_num_ = expand_times * (this->group_num_ - this->split_point_);

      // set models
      new_data_node1->model_.a_ = (this->model_.a_);
      new_data_node1->model_.b_ = (this->model_.b_);

      new_data_node2->model_.a_ = expand_times * (this->model_.a_);
      new_data_node2->model_.b_ = expand_times * (this->model_.b_ - this->split_point_);

      // expand each group and update group_offsets_[], data_capacity_
      new_data_node1->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node1->group_num_ + 1)));
      new_data_node2->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node2->group_num_ + 1)));

      int cur_group_start = 0;
      for (int i = 0; i < split_point_; i++) {
        new_data_node1->group_offsets_[i] = cur_group_start;
        int cur_group_size = get_group_size(i);
        int new_group_size = cur_group_size * expand_times;
        cur_group_start += new_group_size;
      }
      new_data_node1->group_offsets_[new_data_node1->group_num_] = cur_group_start;
      new_data_node1->data_capacity_ = cur_group_start;

      cur_group_start = 0;
      for (int i = split_point_; i < group_num_; i++) {
        int idx_in_target2 = i - split_point_;
        new_data_node2->group_offsets_[2 * idx_in_target2] = cur_group_start;
        int cur_group_size = this->get_group_size(i);
        int new_group_size = cur_group_size * expand_times;
        int new_group_first = new_group_size / 2;
        cur_group_start += new_group_first;
        new_data_node2->group_offsets_[2 * idx_in_target2 + 1] = cur_group_start;

        int new_group_second = new_group_size - new_group_first;
        cur_group_start += new_group_second;
      }
      new_data_node2->group_offsets_[new_data_node2->group_num_] = cur_group_start;
      new_data_node2->data_capacity_ = cur_group_start;

      new_data_node1->initialize_for_smo();
      new_data_node2->initialize_for_smo();

      // 4. set this node's target
      this->target_ = new_data_node1;
      this->target2_ = new_data_node2;

      new_data_node1->next_leaf_ = new_data_node2;
      new_data_node2->prev_leaf_ = new_data_node1;
      break;
    }
    case 4:
    { 
      self_type* new_data_node1 = new self_type();
      new_data_node1->min_key_ = this->min_key_;
      new_data_node1->group_num_ = expand_times * this->split_point_;

      self_type* new_data_node2 = new self_type();
      new_data_node2->min_key_ = this->find_min_key_of_target2();
      if (__glibc_unlikely(new_data_node2->min_key_ == std::numeric_limits<T>::max())) {
        new_data_node2->min_key_ = this->find_max_key_of_target() + 1;
      }
      new_data_node2->group_num_ = (this->group_num_ - this->split_point_);

      // set models
      new_data_node1->model_.a_ = expand_times * (this->model_.a_);
      new_data_node1->model_.b_ = expand_times * (this->model_.b_);

      new_data_node2->model_.a_ = (this->model_.a_);
      new_data_node2->model_.b_ = (this->model_.b_ - this->split_point_);

      // expand each group and update group_offsets_[], data_capacity_
      new_data_node1->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node1->group_num_ + 1)));
      new_data_node2->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node2->group_num_ + 1)));

      int cur_group_start = 0;
      for (int i = 0; i < split_point_; i++) {
        new_data_node1->group_offsets_[2 * i] = cur_group_start;
        int cur_group_size = get_group_size(i);
        int new_group_size = cur_group_size * expand_times;
        int new_group_first = new_group_size / 2;
        cur_group_start += new_group_first;
        new_data_node1->group_offsets_[2 * i + 1] = cur_group_start;
        int new_group_second = new_group_size - new_group_first;
        cur_group_start += new_group_second;
      }
      new_data_node1->group_offsets_[new_data_node1->group_num_] = cur_group_start;
      new_data_node1->data_capacity_ = cur_group_start;

      cur_group_start = 0;
      for (int i = split_point_; i < group_num_; i++) {
        int idx_in_target2 = i - split_point_;
        new_data_node2->group_offsets_[idx_in_target2] = cur_group_start;
        int cur_group_size = this->get_group_size(i);
        int new_group_size = cur_group_size * expand_times;
        cur_group_start += new_group_size;
      }
      new_data_node2->group_offsets_[new_data_node2->group_num_] = cur_group_start;
      new_data_node2->data_capacity_ = cur_group_start;

      new_data_node1->initialize_for_smo();
      new_data_node2->initialize_for_smo();

      // 4. set this node's target
      this->target_ = new_data_node1;
      this->target2_ = new_data_node2;

      new_data_node1->next_leaf_ = new_data_node2;
      new_data_node2->prev_leaf_ = new_data_node1;
      break;
    }
    case 5:
    {
      self_type* new_data_node1 = new self_type();
      new_data_node1->min_key_ = this->min_key_;
      new_data_node1->group_num_ = expand_times * this->split_point_;

      self_type* new_data_node2 = new self_type();
      new_data_node2->min_key_ = this->find_min_key_of_target2();
      if (__glibc_unlikely(new_data_node2->min_key_ == std::numeric_limits<T>::max())) {
        new_data_node2->min_key_ = this->find_max_key_of_target() + 1;
      }
      new_data_node2->group_num_ = expand_times * (this->group_num_ - this->split_point_);

      // set models
      new_data_node1->model_.a_ = expand_times * (this->model_.a_);
      new_data_node1->model_.b_ = expand_times * (this->model_.b_);

      new_data_node2->model_.a_ = expand_times * (this->model_.a_);
      new_data_node2->model_.b_ = expand_times * (this->model_.b_ - this->split_point_);

      // expand each group and update group_offsets_[], data_capacity_
      new_data_node1->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node1->group_num_ + 1)));
      new_data_node2->group_offsets_ = static_cast<uint16_t*>(malloc(sizeof(uint16_t) * (new_data_node2->group_num_ + 1)));

      int cur_group_start = 0;
      for (int i = 0; i < split_point_; i++) {
        new_data_node1->group_offsets_[2 * i] = cur_group_start;
        int cur_group_size = get_group_size(i);

        int new_group_size = cur_group_size * expand_times;
        int new_group_first = new_group_size / 2;
        cur_group_start += new_group_first;
        new_data_node1->group_offsets_[2 * i + 1] = cur_group_start;

        int new_group_second = new_group_size - new_group_first;
        cur_group_start += new_group_second;
      }
      new_data_node1->group_offsets_[new_data_node1->group_num_] = cur_group_start;
      new_data_node1->data_capacity_ = cur_group_start;

      cur_group_start = 0;
      for (int i = split_point_; i < group_num_; i++) {
        int idx_in_target2 = i - split_point_;
        new_data_node2->group_offsets_[2 * idx_in_target2] = cur_group_start;
        int cur_group_size = this->get_group_size(i);
        int new_group_size = cur_group_size * expand_times;
        int new_group_first = new_group_size / 2;
        cur_group_start += new_group_first;
        new_data_node2->group_offsets_[2 * idx_in_target2 + 1] = cur_group_start;

        int new_group_second = new_group_size - new_group_first;
        cur_group_start += new_group_second;
      }
      new_data_node2->group_offsets_[new_data_node2->group_num_] = cur_group_start;
      new_data_node2->data_capacity_ = cur_group_start;

      new_data_node1->initialize_for_smo();
      new_data_node2->initialize_for_smo();

      // 4. set this node's target
      this->target_ = new_data_node1;
      this->target2_ = new_data_node2;

      new_data_node1->next_leaf_ = new_data_node2;
      new_data_node2->prev_leaf_ = new_data_node1;
      break;
    }
    default:
      break;
    }

    return;
  }

  /*** Deletes ***/
  
  // Erase all keys with the input value
  // Returns the number of keys erased (there may be multiple keys with the same
  // value)
  int erase(const T& key) {
#ifdef USING_LOCK
    uint32_t leaf_version;
    bool leaf_is_locked = test_leaf_lock_set(leaf_version);
    if (leaf_is_locked) {
      return 1;
    }
#endif

    int group_idx = get_key_group(key);
    int group_size = get_group_size(group_idx);
    nano_type * cur_nano = nullptr;

#ifdef USING_LOCK
    bool get_lock_success = try_get_group_lock(group_idx);
    if (!get_lock_success) {
      return 1;
    }
#endif

    if (group_size >= two_choice_threashold) {
      int search_idx1 = get_idx_hash1(group_idx, group_size, key);
      int search_idx2 = get_idx_hash2(group_idx, group_size, key);

      cur_nano = get_nano(search_idx1);
      cur_nano->prefetchNano();
      int pos_in_nano1 = cur_nano->searchInNano(key);
      if (pos_in_nano1 >= 0) {
        cur_nano->deleteInNano(key, pos_in_nano1);
        return 1;
      } 

      if (search_idx2 != search_idx1) {
        cur_nano = get_nano(search_idx2);
        cur_nano->prefetchNano();
        int pos_in_nano2 = cur_nano->searchInNano(key);
        if (pos_in_nano2 >= 0) {
          cur_nano->deleteInNano(key, pos_in_nano2);
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
          return 1;
        }
      }
    } else {
      int search_idx1 = get_idx_hash1(group_idx, group_size, key);
      cur_nano = get_nano(search_idx1);
      cur_nano->prefetchNano();
      int pos_in_nano1 = cur_nano->searchInNano(key);
      if (pos_in_nano1 >= 0) {
        cur_nano->deleteInNano(key, pos_in_nano1);
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
        return 1;
      }
    }

    // overflow nano
    cur_nano = get_of_nano(group_idx);
    cur_nano->prefetchNano();
    int pos_in_ofnano = cur_nano->searchInNano(key);

    
    if (pos_in_ofnano >= 0) {
      cur_nano->deleteInNano(key, pos_in_ofnano);
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
      return 1;
    }
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
    return 0;
  }

  // 0: update success
  // 1: update fail
  /*** Updates ***/
  int update(const T& key, const P& payload) {
#ifdef USING_LOCK
    uint32_t leaf_version;
    bool leaf_is_locked = test_leaf_lock_set(leaf_version);
    if (leaf_is_locked) {
      return 1;
    }
#endif
    int group_idx = get_key_group(key);
    int group_size = get_group_size(group_idx);
    nano_type * cur_nano = nullptr;

#ifdef USING_LOCK
    bool get_lock_success = try_get_group_lock(group_idx);
    if (!get_lock_success) {
      return 1;
    }
#endif

    if (group_size >= two_choice_threashold) {
      int search_idx1 = get_idx_hash1(group_idx, group_size, key);
      int search_idx2 = get_idx_hash2(group_idx, group_size, key);

      cur_nano = get_nano(search_idx1);
      cur_nano->prefetchNano();
      int pos_in_nano1 = cur_nano->searchInNano(key);
      if (pos_in_nano1 >= 0) {
        cur_nano->updateInNano(key, payload, pos_in_nano1);
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
        return 0;
      } 

      if (search_idx2 != search_idx1) {
        cur_nano = get_nano(search_idx2);
        cur_nano->prefetchNano();
        int pos_in_nano2 = cur_nano->searchInNano(key);
        if (pos_in_nano2 >= 0) {
          cur_nano->updateInNano(key, payload, pos_in_nano2);
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
          return 0;
        }
      }
    } else {
      int search_idx1 = get_idx_hash1(group_idx, group_size, key);
      cur_nano = get_nano(search_idx1);
      cur_nano->prefetchNano();
      int pos_in_nano1 = cur_nano->searchInNano(key);
      if (pos_in_nano1 >= 0) {
        cur_nano->updateInNano(key, payload, pos_in_nano1);
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
        return 0;
      }
    }

    // overflow nano
    cur_nano = get_of_nano(group_idx);
    cur_nano->prefetchNano();
    int pos_in_ofnano = cur_nano->searchInNano(key);

    if (pos_in_ofnano >= 0) {
      cur_nano->updateInNano(key, payload, pos_in_ofnano);
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
      return 0;
    }
#ifdef USING_LOCK
      release_group_lock(group_idx);
#endif
    return 1;
  }

  /*** Range Scan ***/
  // returns the number of keys scanned in this leaf  
  int range_scan_by_size(const T& key, int to_scan, V* result) {
    return 0;
  }


  int range_scan_whole_group(int group_idx, V* result) {
    int j = 0;
    int nano_start = group_offsets_[group_idx];
    int nano_end = group_offsets_[group_idx+1];
    // nano[]
    for (int i = nano_start; i < nano_end; i++) {
      nano_type * cur_nano = get_nano(i);
      if (!(cur_nano->isEmpty())) {
        for (int k = 0; k < 15; k++) {
          if (cur_nano->exist(k)) {
            T cur_key = cur_nano->ck(k);
            P cur_payload = cur_nano->cch(k);

            result[j].first = cur_key;
            result[j].second = cur_payload;
            j++;
          }
        }
      }
    }

    // of_nano
    nano_type *of_nano = get_of_nano(group_idx);
    if (!(of_nano->isEmpty())) {
      for (int k = 0; k < 15; k++) {
        if (of_nano->exist(k)) {
          T cur_key = of_nano->ck(k);
          P cur_payload = of_nano->cch(k);

          result[j].first = cur_key;
          result[j].second = cur_payload;
          j++;
        }
      }
    }

    return j;
  }

  int range_scan_group_by_key(const T& low_key, const T& up_key, int group_idx, V* result) {
    int j = 0;
    int nano_start = group_offsets_[group_idx];
    int nano_end = group_offsets_[group_idx+1];

    // nano[]
    for (int i = nano_start; i < nano_end; i++) {
      nano_type * cur_nano = get_nano(i);
      if (!(cur_nano->isEmpty())) {
          for (int k = 0; k < 15; k++) {
          if (cur_nano->exist(k)) {
            T cur_key = cur_nano->ck(k);
            P cur_payload = cur_nano->cch(k);

            if (cur_key >= low_key && cur_key < up_key){
              result[j].first = cur_key;
              result[j].second = cur_payload;
              j++;
            }
          }
        }
      }
    }

    // of_nano
    nano_type *of_nano = get_of_nano(group_idx);
    if (!(of_nano->isEmpty())) {
      for (int k = 0; k < 15; k++) {
        if (of_nano->exist(k)) {
          T cur_key = of_nano->ck(k);
          P cur_payload = of_nano->cch(k);
          if (cur_key >= low_key && cur_key < up_key){
            result[j].first = cur_key;
            result[j].second = cur_payload;
            j++;
          }
        }
      }
    }

    return j;
  }

  // naive scanning
  int range_scan_by_key(const T& low_key, const T& up_key, V* result) {
    int begin_group_idx = get_key_group(low_key); 
    //model_.predict(low_key);
    int end_group_idx = get_key_group(up_key); 
    // model_.predict(up_key);
    int j = 0;
    j += range_scan_group_by_key(low_key, up_key, begin_group_idx, result + j);
    for (int i = begin_group_idx + 1; i <= end_group_idx - 1; i++) {
        j += range_scan_whole_group(i, result + j);
    }
    if (end_group_idx != begin_group_idx) {
      j += range_scan_group_by_key(low_key, up_key, end_group_idx, result + j);
    }
    return j;
  }


  int range_scan_by_key_while_migrating(const T& low_key, const T& up_key, V* result) {
    int group_idx = get_key_group(low_key);
    int end_group_idx = get_key_group(up_key);
    int j = 0;

    while (group_idx <= end_group_idx) {
      if (migration_bitmap_[group_idx] == 0) {
        j += range_scan_group_by_key(low_key, up_key, group_idx, result + j);
      } else { 
        // should be 2, scan in target(s)
        if (!target2_ || group_idx < split_point_) {
          j += target_->range_scan_group_by_key(low_key, up_key, group_idx, result+j);
        } else {
          j += target2_->range_scan_group_by_key(low_key, up_key, group_idx - split_point_, result+j);
        }
      }
      group_idx++;
    }
    
    return j;
  }

  int range_scan_whole_leaf(V* result) {
    int j = 0;
    for (int i = 0; i < group_num_; i++) {
      j+= range_scan_whole_group(i, result+j);
    }
    return j;
  }

  /*** Stats ***/
  // Total size of node metadata (including group_offsets_[])
  long long node_size() const { return sizeof(self_type) + (group_num_ + 1) * sizeof(uint16_t); }

  // Total size in bytes of all nanos
  long long data_size() const {
    long long data_size = this->data_capacity_ * sizeof(nano_type);
    data_size += group_num_ * group_overflow_nano_num * sizeof(nano_type);
    return data_size;
  }

  long long leaf_total_size() const {
    long long leaf_size = node_size() + data_size();
    return leaf_size;
  }

  long long group_key_num(int group_idx) const {
    long long group_key_num = 0;
    int nano_start = group_offsets_[group_idx];
    int nano_end = group_offsets_[group_idx+1];
    for (int i = nano_start; i < nano_end; i++) {
      group_key_num += get_nano(i)->num();
    }
    int of_nano_start = group_idx * group_overflow_nano_num;
    int of_nano_end = of_nano_start + group_overflow_nano_num;
    for (int i = of_nano_start; i < of_nano_end; i++) {
      group_key_num += get_of_nano(i)->num();
    }
    return group_key_num;
  }

  long long total_key_num() const {
    long long total_key_num = 0;
    for (int i = 0; i < data_capacity_; i ++) {
      total_key_num += get_nano(i)->num();
    }
    for (int i = 0; i < group_num_ * group_overflow_nano_num; i ++) {
      total_key_num += get_of_nano(i)->num();
    }
    return total_key_num;
  }

  long long total_nano_num() const {
    long long nano_num = this->data_capacity_ + group_num_ * group_overflow_nano_num;
    return nano_num;
  }

  /*** Debugging ***/

  bool validate_structure(bool verbose = false) const {
    if (this->data_capacity_ == 0){
      std::cout << "[Data node no room]"
                << " node addr: " << this << std::endl;
      return false;
    }

    if (this->group_num_ == 0){
      std::cout << "[Data node no group]"
                << " node addr: " << this << std::endl;
      return false;
    }

    if (this->migration_bitmap_ != nullptr && this->target_ == nullptr) {
      std::cout << "[Data node migrating but no target]"
                << " node addr: " << this 
                << ", migration_bitmap_: " << this->migration_bitmap_ 
                << std::endl;
      return false;
    }

    if (this->target_ != nullptr && this->target_->target_ != nullptr) {
      std::cout << "[Data node double target]"
                << " node addr: " << this 
                << ", node target: " << this->target_
                << ", target's target: " << this->target_->target_
                << std::endl;
      return false;
    }

    if (this->target2_ != nullptr && this->target_->target_ != nullptr) {
      std::cout << "[Data node double target2]"
                << " node addr: " << this << std::endl;
      return false;
    }

    return true;
  }
 };
}
