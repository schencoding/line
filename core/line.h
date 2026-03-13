#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stack>
#include <type_traits>
#include <numeric>
#include <algorithm>
#include <vector>
#include <malloc.h>

#include "piecewise_linear_model.h"
#include "line_base.h"
#include "line_nodes.h"
#include "nano_node_15.h"

#define INNER 0
// 0: ALEX
// 1: LIPP
// 2: B+Tree
// 3: ALEXOL
// #define FOREST

#if (INNER == 0)
#include "alex_src/alex.h"
#elif (INNER == 1)
#include "lipp_src/lipp.h"
#elif (INNER == 2)
#include "btree_src/src/include/stx/btree.h"
#elif (INNER == 3)
#include "alexol_src/alex.h"
#endif

namespace line {

// In inner index, P = LineInDataNode<T, P, Compare, allow_duplicates> *
template <class T, class P, bool allow_duplicates = true>
class InnerIndex {
  typedef std::pair<T, P> V;
 public:

#if (INNER == 0)
  inneralex::Alex <T, P, inneralex::AlexCompare, std::allocator<std::pair <T, P>>, false> original_index;
#elif (INNER == 1)
  innerlipp::LIPP <T, P> original_index;
#elif (INNER == 2)
  stx::btree<T, P> original_index;
#elif (INNER == 3)
  inneralexol::Alex <T, P, inneralexol::AlexCompare, std::allocator<std::pair <T, P>>, false> original_index;
#endif

  // build node with a list of keys and pointer of leaves
  void build_with_leaves(const V key_leaves[], int leaf_num) {
#if (INNER == 0 || INNER == 1)
    // alex & lipp
    original_index.bulk_load(key_leaves, leaf_num);
#elif (INNER == 2)
    // btree
    original_index.bulk_load(key_leaves, key_leaves + leaf_num);
#elif (INNER == 3)
    // alexol
    original_index.set_max_model_node_size(1 << 24); 
    original_index.set_max_data_node_size(1 << 19);
    original_index.bulk_load(key_leaves, leaf_num);
#endif
  }

  // find the leaf containing a certain key
  inline bool find_leaf(const T key, P & leaf_ptr) {
#if (INNER == 0 || INNER == 3)
    // alex & alexol
    auto ret = original_index.get_leaf_node(key, &leaf_ptr);
    return ret;
#elif (INNER == 1)
    // lipp
    leaf_ptr = original_index.at(key);
    return (leaf_ptr != nullptr);
#elif (INNER == 2)
    // btree
    auto iter = original_index.upper_bound(key);
    iter--;
    leaf_ptr = iter.data();
    return true;
#endif
  }

  // insert a leaf with a certain minimal key
  bool insert_leaf(T key, P leaf_ptr) {
#if (INNER == 0)
    // alex
    return original_index.insert(key, leaf_ptr).second;
#elif (INNER == 1)
    // lipp
    return original_index.insert(key, leaf_ptr);
#elif (INNER == 2)
    // btree
    auto res_pair = original_index.insert(key, leaf_ptr);
    return res_pair.second;
#elif (INNER == 3)
    // alexol
    return original_index.insert(key, leaf_ptr);
#endif
  }

  bool update_leaf(T key, P old_leaf_ptr, P leaf_ptr) {
#if (INNER == 0 || INNER == 3)
    // alex & alexol
    auto inner_leaf = original_index.get_leaf(key);
    int pos = inner_leaf->find_key(key);
    auto cas_ret = CAS(&(inner_leaf->get_payload(pos)), &old_leaf_ptr, leaf_ptr);
    return cas_ret;
#elif (INNER == 1)
    // lipp
    // return original_index.update(key, leaf_ptr);
#elif (INNER == 2)
    // btree
    auto iter = original_index.find(key);
    iter.data() = leaf_ptr;
    return true;
#endif
  }
};

template <class T, class P, bool allow_duplicates = true>
class Line {
  static_assert(std::is_arithmetic<T>::value, "key type must be numeric.");

 public:
  typedef nano<T, P> nano_type;
  typedef std::pair<T, P> V;

  typedef Line<T, P, allow_duplicates> self_type;
  typedef LineDataNode<T, P, allow_duplicates> data_node_type;
#ifdef FOREST
  class InnerIndex<T, data_node_type *> *inner_index;
  std::vector<T> cmp_keys;
#else
  class InnerIndex<T, data_node_type *> inner_index;
#endif
 public:
  Line() {
  }

  ~Line() {
  }

 public:
// Return the data node that contains the key (if it exists).
#ifdef FOREST
  forceinline data_node_type* get_leaf(const T key) {
    data_node_type * leaf_ptr = nullptr;
    int i = 0;
    for (; i<cmp_keys.size(); i++) {
      if (key < cmp_keys[i])
        break;
    }
    // std::cout << key << " compared to " << cmp_keys[i-1] << std::endl;
    auto ret = inner_index[i-1].find_leaf(key, leaf_ptr);
    return leaf_ptr;
  }
#else
  forceinline data_node_type* get_leaf(const T key) {
    data_node_type * leaf_ptr = nullptr;
    auto ret = inner_index.find_leaf(key, leaf_ptr);
    return leaf_ptr;
  }
#endif

#ifdef FOREST
  forceinline void insert_leaf(T key, data_node_type* leaf_ptr) {
    int i = 0;
    for (; i<cmp_keys.size(); i++) {
      if (key < cmp_keys[i])
        break;
    }
    inner_index[i-1].insert_leaf(key, leaf_ptr);
  }
#else
  forceinline void insert_leaf(T key, data_node_type* leaf_ptr) {
    inner_index.insert_leaf(key, leaf_ptr);
  }
#endif

#ifdef FOREST
  forceinline void update_leaf(T key, data_node_type* old_leaf_ptr, data_node_type* leaf_ptr) {
    int i = 0;
    for (; i<cmp_keys.size(); i++) {
      if (key < cmp_keys[i])
        break;
    }
    inner_index[i-1].update_leaf(key, old_leaf_ptr, leaf_ptr);
  }
#else
  forceinline void update_leaf(T key, data_node_type* old_leaf_ptr, data_node_type* leaf_ptr) {
    inner_index.update_leaf(key, old_leaf_ptr, leaf_ptr);
  }
#endif


  // Return left-most data node, only available for inner alex
#if (INNER == 0)
  data_node_type* first_data_node() const {
    auto inner_first_data_node = inner_index.original_index.first_data_node();
    return inner_first_data_node->get_payload(inner_first_data_node->first_pos());
  }

  // Return right-most data node
  data_node_type* last_data_node() const {
    auto inner_last_data_node = inner_index.original_index.last_data_node();
    return inner_last_data_node->get_payload(inner_last_data_node->last_pos());
  }
  // Returns minimum key in the index
  T get_min_key() const { return first_data_node()->first_key(); }

  // Returns maximum key in the index
  T get_max_key() const { return last_data_node()->last_key(); }
#endif

  /*** Bulk loading ***/
 public:

  int segments_transform(std::vector<OptimalPiecewiseLinearModel<uint64_t, size_t>::CanonicalSegment> & segs, T keys[], int orig_error_bound, int target_error_bound) {
    int target_seg_num = 0;
    uint64_t orig_x = keys[0];
    int orig_y = 0;
    int cumulated_key_num = 0;
    int seg_cnt = 0;

    auto cur_seg = segs.begin();
    uint64_t cur_first_key = (*cur_seg).get_first_x();
    auto [first_slope, first_intercept] = (*cur_seg).get_floating_point_segment(cur_first_key);
    int seg_key_num = (*cur_seg).get_number();
    uint64_t cur_last_key = keys[cumulated_key_num + seg_key_num - 1];

    int delta_error_bound = target_error_bound - orig_error_bound;

    long double new_slope_lower = (long double)(first_slope * (cur_last_key - cur_first_key) + first_intercept - delta_error_bound - orig_y) /(cur_last_key - orig_x);
    long double new_slope_upper = (long double)(first_slope * (cur_last_key - cur_first_key) + first_intercept + delta_error_bound - orig_y) /(cur_last_key - orig_x);

    cumulated_key_num += seg_key_num;
    seg_cnt++;
    cur_seg++;
    for (; cur_seg!=segs.end(); cur_seg++) {
      cur_first_key = (*cur_seg).get_first_x();
      seg_key_num = (*cur_seg).get_number();
      cur_last_key = keys[cumulated_key_num + seg_key_num - 1];

      auto [cur_slope, cur_intercept] = (*cur_seg).get_floating_point_segment(cur_first_key);

      long double cur_slope_lower_numerator = -cur_slope * (cur_first_key - orig_x) + cur_intercept - orig_y - delta_error_bound;
      long double cur_slope_upper_numerator = -cur_slope * (cur_first_key - orig_x) + cur_intercept - orig_y + delta_error_bound;

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
        // violate the error bound, count and reset
        target_seg_num++;

        orig_x = keys[cumulated_key_num];
        orig_y = cumulated_key_num;

        new_slope_lower = (long double)(cur_slope * (cur_last_key - cur_first_key) + cur_intercept - delta_error_bound - orig_y) /(cur_last_key - orig_x);
        new_slope_upper = (long double)(cur_slope * (cur_last_key - cur_first_key) + cur_intercept + delta_error_bound - orig_y) /(cur_last_key - orig_x);
      }

      cumulated_key_num += seg_key_num;
      seg_cnt++;
    }
    return target_seg_num;
  }

  int error_bound_select(typename std::vector<T> key_temp, const V values[], int leaf_num_limit, int init_eb) {
    if (seg_error_bound == 0) {
      int cur_seg_error_bound = init_eb;
      int last_seg_error_bound = cur_seg_error_bound;
      auto segments = make_segmentation(key_temp.begin(), key_temp.end(), cur_seg_error_bound);
      int seg_num = segments.size();
      int new_seg_error_bound = last_seg_error_bound;
      while (seg_num > leaf_num_limit) {
        new_seg_error_bound = last_seg_error_bound * 2;
        seg_num = segments_transform(segments, values, cur_seg_error_bound, new_seg_error_bound);
        last_seg_error_bound = new_seg_error_bound;
      }
      return new_seg_error_bound;
    } else {
      return seg_error_bound;
    }
  }

#ifdef FOREST
  int build_leaves_for_one_tree(typename std::vector<T>::iterator key_begin, typename std::vector<T>::iterator key_end, std::pair<T, data_node_type *> * & key_leaves_arr, const V values[], int & real_leaf_num) {
    auto segments = make_segmentation(key_begin, key_end, seg_error_bound);
    int leaf_num = segments.size();
    std::cout << "Bulkload all leaves, init leaf num " << leaf_num << std::endl;
    std::vector<std::pair<T, data_node_type *>> key_leaves;
    key_leaves.reserve(leaf_num);
    int seg_begin_idx = 0;
    for (auto cur_seg = segments.begin(); cur_seg!=segments.end(); cur_seg++) {
      T first_key = (*cur_seg).get_first_x();
      int seg_key_num = (*cur_seg).get_number();
      int estimate_nano_num = std::ceil((double)seg_key_num * init_predicted_rate / NANO_INIT_KEY_NUM);
      if (estimate_nano_num <= nano_num_limit) {
        auto predict_data_node = new data_node_type();
        predict_data_node->bulk_load(values+seg_begin_idx, seg_key_num);

        key_leaves.emplace_back(std::make_pair(first_key, predict_data_node));

        seg_begin_idx+=seg_key_num;
      } 
    }

    real_leaf_num = key_leaves.size();
    key_leaves_arr = new std::pair<T, data_node_type *>[real_leaf_num]();
    std::cout << "Build inner tree with " << real_leaf_num << " leaves" << std::endl;
    std::cout << "inner key range: " << key_leaves[0].first << " ~ " << key_leaves[real_leaf_num-1].first << std::endl;
    for (int i = 0; i < real_leaf_num; i++) {
      key_leaves_arr[i] = key_leaves[i];
    }

    return seg_begin_idx;
  }

  void bulk_load(const V values[], int num_keys) {
    std::vector<T> key_temp;
    key_temp.reserve(num_keys);
    for (size_t i = 0; i < num_keys; i++) {
      key_temp.push_back(values[i].first);
    }

    int key_part_num = 100;
    int * key_part = new int[key_part_num]();
    long double slope = (double)key_part_num / (values[num_keys-1].first - values[0].first);
    long double intercept = - slope * values[0].first;

    for (size_t i = 0; i < num_keys; i++) {
        int pos = std::min(static_cast<int>(std::floor(key_temp[i] * slope + intercept)), key_part_num-1);
        key_part[pos]++;
    }

    std::vector<std::pair<int, int>> empty_start_and_len;
    int empty_part = 0;
    int max_empty_len = 0;
    int cur_empty_len = 0;
    std::vector<uint64_t> empty_segs;

    for (int i = 0; i < key_part_num; i++) {
      if (key_part[i] <= 10) {
        empty_part++;
        cur_empty_len++;
      } else {
        if (cur_empty_len > max_empty_len) {
          max_empty_len = cur_empty_len;
        }
        if (cur_empty_len != 0) {
          empty_segs.push_back(cur_empty_len);
          empty_start_and_len.push_back(std::make_pair(cur_empty_len, i-cur_empty_len));
          cur_empty_len = 0;
        }
      }
    }
    // last emtpy seg
    if (cur_empty_len > max_empty_len) {
      max_empty_len = cur_empty_len;
    }
    if (cur_empty_len != 0) {
      empty_segs.push_back(cur_empty_len);
      empty_start_and_len.push_back(std::make_pair(cur_empty_len, key_part_num-cur_empty_len));
    }

    std::sort(empty_start_and_len.begin(),empty_start_and_len.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b){ return a.first > b.first; }); 

    for (int i = 0; i < empty_start_and_len.size(); i++) {
      std::cout << empty_start_and_len[i].first << ", " << empty_start_and_len[i].second << std::endl;
    }
    // choose first 2 longest empty segs
    int num_valid_empty_seg = 0;
    auto end_iter = empty_start_and_len.begin();
    for (int i = 0; i < empty_start_and_len.size(); i++) {
      if (empty_start_and_len[i].first > 20) {
        end_iter++;
      } else {
        break;
      }
    }
    int tree_num = 2 * (end_iter - empty_start_and_len.begin()) + 1;
    inner_index = new InnerIndex<T, data_node_type *>[tree_num];
    int tree_idx = 0;
    // sort according to idx
    std::sort(empty_start_and_len.begin(), end_iter, [](const std::pair<int, int>& a, const std::pair<int, int>& b){ return a.second < b.second; }); 

    for (int i = 0; i < empty_start_and_len.size(); i++) {
      std::cout << empty_start_and_len[i].first << ", " << empty_start_and_len[i].second << std::endl;
    }
    uint64_t last_key = values[0].first;
    int empty_idx = 0;
    int last_end = 0;
    int acc_key_num = 0;
    auto key_begin_itr = key_temp.begin();
    auto key_end_itr = key_temp.begin();
    int seg_begin_idx = 0;

    for (auto itr = empty_start_and_len.begin(); itr < end_iter; itr++) {
      int len = (*itr).first;
      int start = (*itr).second;
      std::cout << "cur itr: " << len << ", " << start << std::endl;
      int cur_num_keys = 0;
      for (int j = last_end; j < start; j++) {
        cur_num_keys += key_part[j];
      }
      if (cur_num_keys > 0) {
        std::cout << "leaf " << tree_idx << " begin: " <<  last_key << ", " << cur_num_keys << std::endl;
        key_end_itr += cur_num_keys;
        cmp_keys.emplace_back(last_key);
        
        std::pair<T, data_node_type *> * key_leaves_arr = nullptr;
        int real_leaf_num = 0;
        int bl_key_num = build_leaves_for_one_tree(key_begin_itr, key_end_itr, key_leaves_arr, values+seg_begin_idx, real_leaf_num);
        seg_begin_idx += bl_key_num;
        
        // 3. Build inner index with leaves
        inner_index[tree_idx].build_with_leaves(key_leaves_arr, real_leaf_num);
        tree_idx++;
        delete [] key_leaves_arr;
      }

      // ----------the "empty" part-------------
      key_begin_itr = key_end_itr;
      acc_key_num += cur_num_keys;

      last_key = values[acc_key_num].first;
      cur_num_keys = 0;
      for (int j = start; j < start + len; j++) {
        cur_num_keys += key_part[j];
      }
      std::cout << "acc_key_num: " << acc_key_num << std::endl;
      std::cout << "cur_num_keys: " << cur_num_keys << std::endl;

      if (cur_num_keys > 0) {
        std::cout << "leaf " << tree_idx << " begin: " <<  last_key << ", " << cur_num_keys << std::endl;
        key_end_itr += cur_num_keys;
        acc_key_num += cur_num_keys;
        cmp_keys.emplace_back(last_key);

        std::pair<T, data_node_type *> * key_leaves_arr = nullptr;
        int real_leaf_num = 0;
        int bl_key_num = build_leaves_for_one_tree(key_begin_itr, key_end_itr, key_leaves_arr, values+seg_begin_idx, real_leaf_num);
        seg_begin_idx += bl_key_num;

        // 3. Build inner index with leaves
        inner_index[tree_idx].build_with_leaves(key_leaves_arr, real_leaf_num);
        delete [] key_leaves_arr;
        tree_idx++;
        key_begin_itr = key_end_itr;
      }

      last_end = start + len;
      std::cout << last_end << std::endl;
      std::cout << acc_key_num << std::endl;
      if (acc_key_num < cur_num_keys) {
        last_key = values[acc_key_num].first;
      }
    }

    // last tree
    if (last_end != key_part_num) {
      int cur_num_keys = 0;
      for (int j = last_end; j < key_part_num; j++) {
        cur_num_keys += key_part[j];
      }
      std::cout << "leaf " << tree_idx << " begin: " << last_key << ", " << cur_num_keys << std::endl;
      acc_key_num += cur_num_keys;
      cmp_keys.emplace_back(last_key);
      key_end_itr += cur_num_keys;

      std::pair<T, data_node_type *> * key_leaves_arr = nullptr;
      int real_leaf_num = 0;
      int bl_key_num = build_leaves_for_one_tree(key_begin_itr, key_end_itr, key_leaves_arr, values+seg_begin_idx, real_leaf_num);
      seg_begin_idx += bl_key_num;

      // 3. Build inner index with leaves
      inner_index[tree_idx].build_with_leaves(key_leaves_arr, real_leaf_num);
      delete [] key_leaves_arr;
    }

    std::cout << tree_idx << std::endl;
    for (int i = 0; i < cmp_keys.size(); i++) {
      std::cout << cmp_keys[i] << std::endl;
      #if (INNER==0)
      inner_index[i].original_index.print_depth();
      #endif
    }
  }
#else
  // values should be the sorted array of key-payload pairs.
  // The number of elements should be num_keys.
  // The index must be empty when calling this method.
  void bulk_load(const V values[], int num_keys) {
    // 1. Generate segments using PLA
    std::vector<T> key_temp;
    key_temp.reserve(num_keys);
    for (size_t i = 0; i < num_keys; i++) {
      key_temp.push_back(values[i].first);
    }

    auto segments = make_segmentation(key_temp.begin(), key_temp.end(), seg_error_bound);
    int leaf_num = segments.size();

    // 2. Bulkload all leaves and collect them
    std::cout << "Bulkload all leaves, init leaf num " << leaf_num << std::endl;
    int k = 0;
    int seg_begin_idx = 0;
    std::vector<std::pair<T, data_node_type *>> key_leaves;
    key_leaves.reserve(leaf_num);

    for (auto cur_seg = segments.begin(); cur_seg!=segments.end(); cur_seg++) {
      T first_key = (*cur_seg).get_first_x();
      int seg_key_num = (*cur_seg).get_number();
      int estimate_nano_num = std::ceil((double)seg_key_num * init_predicted_rate / NANO_INIT_KEY_NUM);
      if (estimate_nano_num <= nano_num_limit) {
        auto predict_data_node = new data_node_type();
        predict_data_node->bulk_load(values+seg_begin_idx, seg_key_num);

        key_leaves.emplace_back(std::make_pair(first_key, predict_data_node));

        seg_begin_idx+=seg_key_num;
      } else { // for one leaf is too large
        int sub_leaf_num = std::ceil((double)estimate_nano_num / nano_num_limit);
        int acc_key_num = 0;
        for (int j = 0; j < sub_leaf_num; j++) {
          int sub_leaf_key_num = 0;
          if (j == sub_leaf_num - 1) {
            sub_leaf_key_num = seg_key_num - acc_key_num;
          } else {
            sub_leaf_key_num = std::ceil((double)seg_key_num / sub_leaf_num);
          }
          acc_key_num += sub_leaf_key_num;
          auto predict_data_node = new data_node_type();
          predict_data_node->bulk_load(values+seg_begin_idx, sub_leaf_key_num);

          first_key = values[seg_begin_idx].first;
          key_leaves.emplace_back(std::make_pair(first_key, predict_data_node));

          seg_begin_idx+=sub_leaf_key_num;
        }
      }
    }

    // 3. Build inner index with leaves
    int real_leaf_num = key_leaves.size();
    std::cout << "Build inner index with " << real_leaf_num << " leaves" << std::endl;
    auto key_leaves_arr = new std::pair<T, data_node_type *>[real_leaf_num]();
    std::cout << "inner key range: " << key_leaves[0].first << " ~ " << key_leaves[real_leaf_num-1].first << std::endl;
    for (int i = 0; i < real_leaf_num; i++) {
      key_leaves_arr[i] = key_leaves[i];
    }
    inner_index.build_with_leaves(key_leaves_arr, real_leaf_num);
    delete [] key_leaves_arr;

    // 4. Link data nodes 
    std::cout << "Link data nodes" << std::endl;
    if (real_leaf_num > 1) {
      key_leaves[0].second->next_leaf_ = key_leaves[1].second;
    }
    for (int i = 1; i <real_leaf_num - 1; i++) {
      key_leaves[i].second->prev_leaf_ = key_leaves[i-1].second;
      key_leaves[i].second->next_leaf_ = key_leaves[i+1].second;
    }
    if (real_leaf_num > 1) {
      key_leaves[real_leaf_num -1].second->prev_leaf_ = key_leaves[real_leaf_num -2].second;
    }
  }
#endif

  /*** Lookup ***/
 public:
  // Directly returns a pointer to the payload found through find(key)
  // Returns null pointer if there is no exact match of the key
  P* get_payload(const T& key) {
SEARCH_RETRY:
    data_node_type* leaf = get_leaf(key);
    if (leaf) {
      P* result = nullptr;
      int ret = 0;
      bool leaf_not_migrating = (leaf->migration_bitmap_ == nullptr);
      if (__glibc_likely(leaf_not_migrating)) {
        ret = leaf->search(key, result);
      } else {
        ret = leaf->search_while_migrating(key, result);
      }
      if (ret == 1) {
        goto SEARCH_RETRY;
      } else {
        if(__glibc_unlikely(result == nullptr)) {
          leaf->search_debug(key, result);
        }
        return result;
      }
    } 
    return nullptr;
  }

  /*** Insert ***/

  // This will NOT do an update of an existing key.
  // Insert does not happen if duplicates are not allowed and duplicate is
  // found.
  bool insert(const T& key, const P& payload) {
INSERT_RETRY:
    // 1. get the leaf to insert the key
    data_node_type* leaf = get_leaf(key);

    // 2. insert into the leaf
    if (leaf) {
      int ret = 0;
      bool is_leaf_migrating = (leaf->migration_bitmap_ != nullptr);
      if (__glibc_unlikely(is_leaf_migrating)) {
        ret = leaf->insert_while_migrating(key, payload);
      } else {
        ret = leaf->insert(key, payload);
      }

      if (ret == 0) {
        return true;
      }

      if (ret == -1) { // duplicate found
        return true;
      }

      if (ret == 1) {
        goto INSERT_RETRY;
      }

      // SMO trigger
      if (ret == 2) {
        // lock the old leaf and prepare the new leaf
#ifdef USING_LOCK
        bool get_leaf_lock_success = leaf->try_get_leaf_lock();
        if (!get_leaf_lock_success) {
          goto INSERT_RETRY;
        }
#endif
        // calculate SMO method, check if smo has already begun
        auto smo_method = leaf->decide_smo_method();
#ifdef USING_LOCK
        if (smo_method == -1) {
          leaf->release_leaf_lock();
          goto INSERT_RETRY;
        }
        for (int i = 0; i < leaf->group_num_; i++){
          leaf->get_group_lock(i);
        }
#endif
        smo_cnt++;
        leaf->prepare_for_migration(smo_method);
#ifdef USING_LOCK
        leaf->release_leaf_lock();
#endif
        goto INSERT_RETRY;
      }

      // SMO finish
      if (ret == 3) {
        // expand
        if (leaf->target2_ == nullptr) {
          update_leaf(leaf->min_key_, leaf, leaf->target_);
          if (ret) { // CAS success
            smo_finish++;

            if (leaf->prev_leaf_) {
              (leaf->prev_leaf_)->next_leaf_ = leaf->target_;
              (leaf->target_)->prev_leaf_ = leaf->prev_leaf_;
            }
            if (leaf->next_leaf_) {
              (leaf->next_leaf_)->prev_leaf_ = leaf->target_;
              (leaf->target_)->next_leaf_ = leaf->next_leaf_;
            }
          } 
#ifdef USING_LOCK
          leaf->release_leaf_lock();
#endif
          delete leaf;
          return true;
        } 

        // split
#ifdef USING_LOCK
        auto get_leaf_lock_success = leaf->try_get_leaf_lock();
        if (get_leaf_lock_success) {
          if (leaf->min_key_ == std::numeric_limits<T>::max()) {
            leaf->release_leaf_lock();
            return true;
          }
          insert_leaf(leaf->target2_->min_key_, leaf->target2_);
          update_leaf(leaf->min_key_, leaf, leaf->target_);
          if (leaf->prev_leaf_) {
            (leaf->prev_leaf_)->next_leaf_ = leaf->target_;
          }
          if (leaf->next_leaf_) {
            (leaf->next_leaf_)->prev_leaf_ = leaf->target2_;
          }

          // smo_finish++;
          leaf->min_key_ = std::numeric_limits<T>::max();
          leaf->release_leaf_lock();
        } 
#else
        insert_leaf(leaf->target2_->min_key_, leaf->target2_);
        update_leaf(leaf->min_key_, leaf, leaf->target_);
        if (leaf->prev_leaf_) {
          (leaf->prev_leaf_)->next_leaf_ = leaf->target_;
          (leaf->target_)->prev_leaf_ = leaf->prev_leaf_;
        }
        if (leaf->next_leaf_) {
          (leaf->next_leaf_)->prev_leaf_ = leaf->target2_;
          (leaf->target2_)->next_leaf_ = leaf->next_leaf_;
        }
        smo_finish++;
#endif
        delete leaf;
        return true;
      }
      return true;
    }
    return false;
  }

  /*** Delete ***/
 public:
  // Erases all keys with a certain key value
  int erase(const T& key) {
    data_node_type* leaf = get_leaf(key);
    int num_erased = leaf->erase(key);
    return num_erased;
  }

  int update(const T& key, const P& payload) {
    data_node_type* leaf = get_leaf(key);
    int updated = leaf->update(key, payload);
    return 1-updated;
  }

  int range_scan_by_key(const T& low_key, const T& up_key, V* &result = nullptr) {
    int total_scan_num = 0;
    std::vector<data_node_type*> leaves_to_scan;
    data_node_type* low_leaf = get_leaf(low_key);
    while (low_leaf != nullptr && low_leaf->min_key_ < up_key) {
      leaves_to_scan.emplace_back(low_leaf);
      low_leaf = low_leaf->next_leaf_;
    }

    int scan_leaf_num = leaves_to_scan.size();
    int result_idx = 0;
    int temp_scan_num = 0;

    // leaf[0]
    data_node_type * cur_leaf = leaves_to_scan[0];
    temp_scan_num = cur_leaf->range_scan_by_key(low_key, up_key, result + result_idx);
    result_idx += temp_scan_num;
    total_scan_num += temp_scan_num;

    // leaf[1 ~ N-2]
    for (int i = 1; i < scan_leaf_num-1; i++) {
      cur_leaf = leaves_to_scan[i];
      temp_scan_num = cur_leaf->range_scan_whole_leaf(result + result_idx);

      result_idx += temp_scan_num;
      total_scan_num += temp_scan_num;
    }

    // leaf[N-1]
    if (scan_leaf_num > 1) {
      cur_leaf = leaves_to_scan[scan_leaf_num-1];
      temp_scan_num = cur_leaf->range_scan_by_key(low_key, up_key, result + result_idx);
      total_scan_num += temp_scan_num;
    }

    return total_scan_num;
  }

  /*** Stats ***/
 public:
  // Size in bytes of all the leaves
#if (INNER == 0)
  long long model_size() {
    long long size = 0;
    data_node_type * cur_leaf = first_data_node();
    while (cur_leaf) {
      size += cur_leaf->node_size();
      cur_leaf = cur_leaf->next_leaf_;
    }
    return size;
  }

  long long leaf_size() {
    long long size = 0;
    data_node_type * cur_leaf = first_data_node();
    while (cur_leaf) {
      size += cur_leaf->leaf_total_size();
      cur_leaf = cur_leaf->next_leaf_;
    }
    return size;
  }

  // Size in bytes of the inner index
  long long inner_size() {
    return inner_index.original_index.model_size() + inner_index.original_index.data_size();;
  }
#endif
};
}