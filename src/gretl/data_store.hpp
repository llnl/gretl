// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file data_store.hpp
 */

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <functional>
#include <memory>
#include <any>
#include <type_traits>
#include <utility>
#include "checkpoint.hpp"
#include "checkpoint_strategy.hpp"
#include "print_utils.hpp"

#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace gretl {

using Int = unsigned int;  ///< gretl Int type

class DataStore;

struct StateBase;

template <typename T, typename D = T>
struct State;

/// @brief UpstreamState is a wrapper for a states.  Its used in external-facing interfaces to ensure const correctness
/// for users to encourage correct usage.
struct UpstreamState {
  Int step_;              ///< step
  DataStore* dataStore_;  ///< datastore

  /// @brief get underlying value
  template <typename T>
  const T& get() const;

  /// @brief get underlying dual value
  template <typename D, typename T>
  D& get_dual() const;
};

/// @brief UpstreamStates is a wrapper for a vector of states.  Its used in external-facing interfaces to ensure const
/// correctness for users to encourage correct usage.
struct UpstreamStates {
  /// @brief Default constructor to use in std containers
  UpstreamStates() = default;

  /// @brief Constructor for upstream states
  /// @param store datastore
  /// @param steps vector of upstream steps
  UpstreamStates(DataStore& store, std::vector<Int> steps)
  {
    for (Int s : steps) {
      states_.push_back({s, &store});
    }
  }

  /// @brief Accessor for individual upstream states
  /// @param index index
  template <typename IntT>
  const UpstreamState& operator[](IntT index) const
  {
    return states_[static_cast<size_t>(index)];
  }

  /// @brief Accessor for individual upstream states
  /// @param index index
  const UpstreamState& operator[](Int index) const { return states_[index]; }

  /// @brief Number of upstream states
  Int size() const { return static_cast<Int>(states_.size()); }

  /// @brief Vector of upstream step indices
  const std::vector<UpstreamState>& states() const { return states_; }

 private:
  std::vector<UpstreamState> states_;  ///< states
};

/// @brief DownstreamState is a wrapper for a state.  Its used in external-facing interfaces to ensure const correctness
/// for users to encourage correct usage.
struct DownstreamState {
  /// @brief Constructor
  /// @param s datastore
  /// @param step step
  DownstreamState(DataStore* s, Int step) : dataStore_(s), step_(step) {}

  /// @brief set underlying value (copy)
  template <typename T, typename D = T>
  void set(const T& t);

  /// @brief set underlying value (move)
  template <typename T, typename D = std::decay_t<T>>
  void set(T&& t);

  /// @brief get underlying value
  template <typename T, typename D = T>
  const T& get() const;

  /// @brief get underlying dual value
  template <typename D, typename T = D>
  const D& get_dual() const;

  friend class DataStore;

 private:
  DataStore* dataStore_;  ///< datastore
  Int step_;              ///< step
};

/// @brief ZeroDual function type
template <typename T, typename D = T>
using InitializeZeroDual = std::function<D(const T&)>;

/// @brief Default zero initializer,
template <typename T, typename D = T>
struct defaultInitializeZeroDual {
  /// @brief functor operator
  D operator()(const T&) { return D{}; }
};

/// @brief DataStore class hold onto states, duals and additional information to represent a computational graph, its
/// checkpointing state information, and its backpropagated sensitivities
class DataStore {
 public:
  /// @brief Constructor requiring a checkpoint strategy.
  /// @param strategy a checkpoint strategy implementation (e.g., WangCheckpointStrategy,
  /// StrummWaltherCheckpointStrategy)
  explicit DataStore(std::unique_ptr<CheckpointStrategy> strategy);

  /// @brief virtual destructor. Must clear states_ first because StateBase
  /// destructors call try_to_free() which accesses upstreams_ and other members.
  /// Without this, implicit reverse-declaration-order destruction would destroy
  /// upstreams_ before states_, causing use-after-free.
  /// @brief virtual destructor
  virtual ~DataStore()
  {
    // Set flag to prevent try_to_free() from accessing freed memory during destruction
    isDestroying_ = true;
  }

  /// @brief create a new state in the graph, store it, return it
  template <typename T, typename D>
  State<T, D> create_state(const T& t, InitializeZeroDual<T, D> initial_zero_dual = [](const T&) { return D{}; })
  {
    State<T, D> state(this, states_.size(), std::make_shared<std::any>(t), initial_zero_dual);
    add_state(std::make_unique<State<T, D>>(state), {});
    return state;
  }

  /// @brief  unwind one step of the graph
  virtual void reverse_state();

  /// @brief unwind the entire graph
  void back_prop();

  /// @brief clear all but persistent state, keeping the graph. Returns the number of persistent states.
  void reset();

  /// @brief reevaluates the final state and refills checkpoints to get ready for another back propagation
  void reset_for_backprop();

  /// @brief clear all but persistent state, remove the graph
  void reset_graph();

  /// @brief resize data structures
  void resize(Int newSize);

  /// @brief get total number of states in the graph
  Int size() { return static_cast<Int>(states_.size()); }

  /// @brief print all checkpoint data in data store
  void print_graph() const;

  /// @brief do internal checks of consistency with respect to checkpoints and
  bool check_validity() const;

  /// @brief create a new state in the graph, store it, return it
  template <typename T, typename D>
  State<T, D> create_empty_state(InitializeZeroDual<T, D> initial_zero_dual, const std::vector<StateBase>& upstreams)
  {
    gretl_assert(!upstreams.empty());
    auto t = std::make_shared<std::any>(T{});
    State<T, D> state(this, states_.size(), t, initial_zero_dual);
    add_state(std::make_unique<State<T, D>>(state), upstreams);
    return state;
  }

  /// @brief vjp
  void vjp(StateBase& state);

  /// @brief function for safely adding new states to graph and checkpoint
  void add_state(std::unique_ptr<StateBase> newState, const std::vector<StateBase>& upstreams);

  /// @brief method for fetching states at a particular step
  void fetch_state_data(Int);

  /// @brief erase the data for a particular step
  void erase_step_state_data(Int);

  /// @brief clear usage at a particular step
  void clear_usage(Int step);

  /// @brief std::function for evaluating downstream from upstreams
  using EvalT = std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)>;

  /// @brief std::function for computing vector-jacobian product from downstream dual to upstream duals
  using VjpT = std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)>;

  /// @brief Get the primal data as a shared_ptr to std::any (type-erased)
  /// @param step
  std::shared_ptr<std::any>& any_primal(Int step);

  /// @brief Get primal value
  /// @param step
  template <typename T>
  const T& get_primal(Int step)
  {
    T* tptr = std::any_cast<T>(any_primal(step).get());
    if (stillConstructingGraph_) {
      if (!tptr) {
        gretl_assert(check_validity());
        print_graph();
        print("on reverse, at ", currentStep_, "getting", step);
      }
      gretl_assert_msg(tptr, "bad step " + std::to_string(step));
    } else {
      if (!tptr) {
        fetch_state_data(step);
        tptr = std::any_cast<T>(any_primal(step).get());
      }
      gretl_assert_msg(tptr, "bad step " + std::to_string(step));
    }
    return *tptr;
  }

  /// @brief Set primal value (forwarding version: moves rvalues, copies lvalues)
  /// @param step step
  /// @param t value of type T to set primal to
  template <typename T>
  void set_primal(Int step, T&& t)
  {
    using U = std::decay_t<T>;
    U* tptr = std::any_cast<U>(any_primal(step).get());
    if (!tptr) {
      gretl_assert(!stillConstructingGraph_);
      any_primal(step) = std::make_shared<std::any>(std::forward<T>(t));
      return;
    }
    gretl_assert(tptr);
    *tptr = std::forward<T>(t);
  }

  /// @brief Get dual value
  /// @param step
  template <typename D, typename T>
  D& get_dual(Int step)
  {
    if (!duals_[step]) {
      const T& thisPrimal = get_primal<T>(step);
      auto thisState = dynamic_cast<const State<T, D>*>(states_[step].get());
      gretl_assert_msg(thisState, std::string("failed to get primal to this state, step ") + std::to_string(step));
      duals_[step] = std::make_unique<std::any>(thisState->initialize_zero_dual_(thisPrimal));
    }
    auto dualData = std::any_cast<D>(duals_[step].get());
    gretl_assert(dualData);
    return *dualData;
  }

  /// @brief Set dual value
  /// @param step step
  /// @param d value of type D to set dual to
  template <typename D>
  void set_dual(Int step, const D& d)
  {
    if (!duals_[step]) {
      duals_[step] = std::make_unique<std::any>(d);
    }
    auto dualData = std::any_cast<D>(duals_[step].get());
    gretl_assert(dualData);
    *dualData = d;
  }

  /// @brief Deallocate the dual value
  /// @param step
  void clear_dual(Int step)
  {
    if (duals_[step]) {
      duals_[step] = nullptr;
    }
  }

  /// @brief Check if state in use
  /// @param step step
  /// @return bool
  bool state_in_use(Int step) const;

  /// @brief Check if state is persistent
  /// @param step step
  /// @return bool
  bool is_persistent(Int step) const;

  /// @brief Register the graph as being complete.  This is mostly for internal consistency checks.
  void finalize_graph() { stillConstructingGraph_ = false; }

  /// @brief Attempt to free the primal value for this state.  This will happen so long as: 1.) the checkpointer doesn't
  /// have is as an active state; 2.) no downstream state which is active according to checkpointer depends on it as an
  /// upstream; and 3.) an external copy of this state is not being help for potential future use outside of the graph.
  void try_to_free(Int step);

  std::vector<std::unique_ptr<StateBase>> states_;  ///< states for steps
  std::vector<std::unique_ptr<std::any>> duals_;    ///< duals for steps
  std::vector<UpstreamStates> upstreams_;           ///< upstreams dependencies for steps
  std::vector<EvalT> evals_;                        ///< forward evaluation functions for steps
  std::vector<VjpT> vjps_;                          ///< vector-jacobian product functions for steps
  std::vector<bool> active_;                        ///< active status for steps
  std::vector<Int> usageCount_;  ///< count how many times a step is used in some downstream still is the scope of the
                                 ///< checkpoint algorithm

  std::vector<Int>
      lastStepUsed_;  ///< for a given step, records the last known future-step where its used as an upstream
  std::vector<std::vector<Int>> passthroughs_;  ///< at a given step, the list of all the previous steps which are
                                                ///< eventually used in some future step as an upstream

  /// container which track the states in the graph with allocated data
  std::unique_ptr<CheckpointStrategy> checkpointStrategy_;

  /// step counter
  Int currentStep_;

  /// @brief specifies if graph is in construction or back-prop mode.  This is used for internal asserts.
  bool stillConstructingGraph_ = true;

  /// @brief flag to prevent accessing freed memory during destruction
  bool isDestroying_ = false;

  friend struct StateBase;

  template <typename T, typename D>
  friend struct State;

  friend struct UpstreamState;
  friend struct DownstreamState;
};

template <typename T>
const T& UpstreamState::get() const
{
  return dataStore_->get_primal<T>(step_);
}

template <typename D, typename T>
D& UpstreamState::get_dual() const
{
  return dataStore_->get_dual<D, T>(step_);
}

template <typename T, typename D>
void DownstreamState::set(const T& t)
{
  dataStore_->set_primal(step_, t);
}

template <typename T, typename D>
void DownstreamState::set(T&& t)
{
  dataStore_->set_primal(step_, std::forward<T>(t));
}

template <typename T, typename D>
const T& DownstreamState::get() const
{
  return dataStore_->get_primal<T>(step_);
}

template <typename D, typename T>
const D& DownstreamState::get_dual() const
{
  return dataStore_->get_dual<D, T>(step_);
}

}  // namespace gretl
