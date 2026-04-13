// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

//
// Stress tests for state lifecycle, try_to_free, usageCount, checkpoint
// eviction, scope-based external reference tracking, and performance scaling.
//

#include <vector>
#include <cmath>
#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>
#include "gtest/gtest.h"
#include "gretl/data_store.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"
#include "gretl/state.hpp"
#include "gretl/double_state.hpp"
#include "gretl/vector_state.hpp"
#include "gretl/test_utils.hpp"

using gretl::DataStore;
using gretl::State;
using gretl::VectorState;

// ---------------------------------------------------------------------------
// Helpers: build graph steps in sub-functions to stress scope/lifetime
// ---------------------------------------------------------------------------

// Build a chain of N steps in a sub-function and return only the final state.
// All intermediate State objects go out of scope when this function returns,
// which triggers destructors -> try_to_free for each intermediate.
static State<double> build_chain_in_subfunc(const State<double>& x0, int N)
{
  State<double> x = x0;
  for (int i = 0; i < N; ++i) {
    // x = 0.5*x + 1.0  => after N steps: x = x0/2^N + 2*(1 - 1/2^N)
    x = gretl::axpb(0.5, x, 1.0);
  }
  return x;
}

// Build a diamond: two branches from x0 that rejoin.
//    x0 -> a = 2*x0+1
//    x0 -> b = 3*x0-1
//    c = a + b = 5*x0
// All intermediates (a, b) go out of scope.
static State<double> build_diamond_in_subfunc(const State<double>& x0)
{
  auto a = gretl::axpb(2.0, x0, 1.0);
  auto b = gretl::axpb(3.0, x0, -1.0);
  return a + b;  // 5*x0
}

// Build a fan-out: x0 is used as upstream by many independent states.
// Returns the sum of all of them.
static State<double> build_fanout_in_subfunc(const State<double>& x0, int fanWidth)
{
  State<double> accum = gretl::axpb(1.0, x0, 0.0);  // copy of x0
  for (int i = 1; i < fanWidth; ++i) {
    auto branch = gretl::axpb(1.0, x0, 0.0);  // another copy
    accum = accum + branch;
  }
  return accum;  // = fanWidth * x0
}

// Nested sub-function: calls another sub-function, introducing two levels
// of scope nesting.
static State<double> build_nested_subfuncs(const State<double>& x0)
{
  auto mid = build_chain_in_subfunc(x0, 3);   // 3-step chain
  auto out = build_chain_in_subfunc(mid, 3);  // another 3-step chain
  return out;
}

// Build a chain that saves intermediates into a user-held vector,
// simulating the pattern of "holding states in scope externally."
static State<double> build_chain_holding_intermediates(const State<double>& x0, int N, std::vector<State<double>>& held)
{
  State<double> x = x0;
  for (int i = 0; i < N; ++i) {
    x = gretl::axpb(0.5, x, 1.0);
    held.push_back(x);  // external reference keeps use_count > 1
  }
  return x;
}

// ---------------------------------------------------------------------------
// TEST SUITE: ScopeLifetime
// States created in sub-functions going out of scope during graph construction
// ---------------------------------------------------------------------------

TEST(ScopeLifetime, ChainInSubfunc_SmallBudget)
{
  // Very tight checkpoint budget (2), long chain built entirely in a sub-func.
  // Intermediates go out of scope on return.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(2));
  auto x0 = store.create_state<double, double>(3.0);
  auto xN = build_chain_in_subfunc(x0, 20);

  double expected = 3.0;
  for (int i = 0; i < 20; ++i) expected = 0.5 * expected + 1.0;
  EXPECT_NEAR(xN.get(), expected, 1e-12);

  gretl::set_as_objective(xN);
  store.back_prop();

  // df/dx0 = (0.5)^20
  EXPECT_NEAR(x0.get_dual(), std::pow(0.5, 20), 1e-12);
}

TEST(ScopeLifetime, ChainInSubfunc_TinyBudget)
{
  // Budget of 1 (absolute minimum for non-persistent checkpoints)
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(1));
  auto x0 = store.create_state<double, double>(5.0);
  auto xN = build_chain_in_subfunc(x0, 10);

  double expected = 5.0;
  for (int i = 0; i < 10; ++i) expected = 0.5 * expected + 1.0;
  EXPECT_NEAR(xN.get(), expected, 1e-12);

  gretl::set_as_objective(xN);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), std::pow(0.5, 10), 1e-12);
}

TEST(ScopeLifetime, DiamondInSubfunc)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(2.0);
  auto result = build_diamond_in_subfunc(x0);

  EXPECT_NEAR(result.get(), 5.0 * 2.0, 1e-14);

  gretl::set_as_objective(result);
  store.back_prop();

  // d(5*x0)/dx0 = 5
  EXPECT_NEAR(x0.get_dual(), 5.0, 1e-14);
}

TEST(ScopeLifetime, NestedSubfuncs)
{
  // Two levels of sub-function scope nesting
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(4.0);
  auto result = build_nested_subfuncs(x0);

  // 6 steps of x -> 0.5*x + 1.0
  double expected = 4.0;
  for (int i = 0; i < 6; ++i) expected = 0.5 * expected + 1.0;
  EXPECT_NEAR(result.get(), expected, 1e-12);

  gretl::set_as_objective(result);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), std::pow(0.5, 6), 1e-12);
}

TEST(ScopeLifetime, FanoutInSubfunc)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(4));
  auto x0 = store.create_state<double, double>(3.0);
  auto result = build_fanout_in_subfunc(x0, 5);

  EXPECT_NEAR(result.get(), 5.0 * 3.0, 1e-14);

  gretl::set_as_objective(result);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), 5.0, 1e-14);
}

// ---------------------------------------------------------------------------
// TEST SUITE: ExternalReferences
// States held externally while graph operations proceed
// ---------------------------------------------------------------------------

TEST(ExternalReferences, HeldIntermediatesPreventPrematureFreeing)
{
  // Hold all intermediates in a vector -- use_count stays > 1 for each.
  // This should prevent try_to_free from deallocating them prematurely.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(2.0);

  std::vector<State<double>> held;
  auto xN = build_chain_holding_intermediates(x0, 8, held);

  // Verify intermediates are still accessible
  double expected = 2.0;
  for (int i = 0; i < 8; ++i) {
    expected = 0.5 * expected + 1.0;
    EXPECT_NEAR(held[static_cast<size_t>(i)].get(), expected, 1e-12) << "intermediate " << i;
  }

  gretl::set_as_objective(xN);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), std::pow(0.5, 8), 1e-12);
}

TEST(ExternalReferences, HeldIntermediatesThenDropped)
{
  // Hold intermediates, run backprop, then drop them.
  // This tests the destructor path when states go out of scope
  // after the graph has been back-propagated.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(2.0);

  {
    std::vector<State<double>> held;
    auto xN = build_chain_holding_intermediates(x0, 6, held);

    gretl::set_as_objective(xN);
    store.back_prop();

    EXPECT_NEAR(x0.get_dual(), std::pow(0.5, 6), 1e-12);
    // held goes out of scope here, destructors fire for all held states
  }
  // If we get here without crash/assert, the destructors handled
  // the post-backprop state correctly.
  SUCCEED();
}

TEST(ExternalReferences, CopyStateAcrossScopes)
{
  // Create a state in one scope, copy it to another, let original go out of scope.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(4));
  auto x0 = store.create_state<double, double>(7.0);

  // Build inner in a lambda that returns it, so the original goes out of scope
  auto outer = [&]() {
    auto inner = gretl::axpb(2.0, x0, 3.0);  // 2*7+3 = 17
    return inner;
  }();

  auto result = gretl::axpb(3.0, outer, 0.0);  // 3*17 = 51
  EXPECT_NEAR(result.get(), 51.0, 1e-14);

  gretl::set_as_objective(result);
  store.back_prop();

  // d(3*(2*x0+3))/dx0 = 6
  EXPECT_NEAR(x0.get_dual(), 6.0, 1e-14);
}

// ---------------------------------------------------------------------------
// TEST SUITE: AssignmentOperator
// Tests for the StateBase assignment operator and its try_to_free calls
// ---------------------------------------------------------------------------

TEST(AssignmentOperator, ReassignMidGraph)
{
  // Reassign a local state variable multiple times in graph construction.
  // Each assignment triggers try_to_free on the old step.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(4));
  auto x0 = store.create_state<double, double>(1.0);

  auto a = gretl::axpb(2.0, x0, 0.0);      // step 1: 2.0
  a = gretl::axpb(3.0, x0, 0.0);           // step 2: 3.0, old step 1 freed
  a = gretl::axpb(4.0, a, 0.0);            // step 3: 12.0, old step 2 is upstream so NOT freed
  auto result = gretl::axpb(1.0, a, 0.0);  // step 4: 12.0

  EXPECT_NEAR(result.get(), 12.0, 1e-14);

  gretl::set_as_objective(result);
  store.back_prop();

  // d(4*3*x0)/dx0 = 12
  EXPECT_NEAR(x0.get_dual(), 12.0, 1e-14);
}

TEST(AssignmentOperator, ReassignInLoop)
{
  // Classic pattern: `x = f(x)` in a loop. Each iteration reassigns
  // the local variable, old step must be freed properly.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(1.0);

  auto x = gretl::axpb(1.0, x0, 0.0);  // copy
  int N = 15;
  for (int i = 0; i < N; ++i) {
    x = gretl::axpb(0.9, x, 0.1);  // x = 0.9*x + 0.1
  }

  double expected = 1.0;
  for (int i = 0; i < N; ++i) expected = 0.9 * expected + 0.1;
  EXPECT_NEAR(x.get(), expected, 1e-10);

  gretl::set_as_objective(x);
  store.back_prop();

  // df/dx0 = 0.9^N (chain rule through linear maps)
  // The copy step adds a factor of 1.0
  EXPECT_NEAR(x0.get_dual(), std::pow(0.9, N), 1e-10);
}

// ---------------------------------------------------------------------------
// TEST SUITE: CheckpointEviction
// Stress the checkpoint manager with various budget/graph combinations
// ---------------------------------------------------------------------------

TEST(CheckpointEviction, LongChainMinimalBudget)
{
  // 50 steps with budget of 2: forces many recomputations.
  int N = 50;
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(2));
  auto x0 = store.create_state<double, double>(1.0);

  auto x = x0;
  for (int i = 0; i < N; ++i) {
    x = gretl::axpb(0.95, x, 0.05);
  }

  gretl::set_as_objective(x);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), std::pow(0.95, N), 1e-8);
}

TEST(CheckpointEviction, LongChainLargeBudget)
{
  // Same chain, but with generous budget. Exercises different checkpoint
  // decisions (most states fit in memory).
  int N = 50;
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(60));
  auto x0 = store.create_state<double, double>(1.0);

  auto x = x0;
  for (int i = 0; i < N; ++i) {
    x = gretl::axpb(0.95, x, 0.05);
  }

  gretl::set_as_objective(x);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), std::pow(0.95, N), 1e-10);
}

TEST(CheckpointEviction, MediumChainExactBudget)
{
  // Budget = N: every state fits. Edge case for checkpoint manager.
  int N = 10;
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(N)));
  auto x0 = store.create_state<double, double>(2.0);

  auto x = x0;
  for (int i = 0; i < N; ++i) {
    x = gretl::axpb(0.8, x, 0.3);
  }

  gretl::set_as_objective(x);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), std::pow(0.8, N), 1e-12);
}

// ---------------------------------------------------------------------------
// TEST SUITE: DAGTopology
// Non-linear graph topologies: diamonds, fan-in, fan-out, skip connections
// ---------------------------------------------------------------------------

TEST(DAGTopology, DiamondDependency)
{
  // x0 -> a, x0 -> b, c = a*b (diamond)
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(4));
  auto x0 = store.create_state<double, double>(3.0);

  auto a = gretl::axpb(2.0, x0, 1.0);   // 2*3+1 = 7
  auto b = gretl::axpb(3.0, x0, -1.0);  // 3*3-1 = 8
  auto c = a * b;                       // 7*8 = 56

  EXPECT_NEAR(c.get(), 56.0, 1e-14);

  gretl::set_as_objective(c);
  store.back_prop();

  // dc/dx0 = dc/da * da/dx0 + dc/db * db/dx0
  //        = b * 2 + a * 3
  //        = 8*2 + 7*3 = 16 + 21 = 37
  EXPECT_NEAR(x0.get_dual(), 37.0, 1e-13);
}

TEST(DAGTopology, MultiInputDiamond)
{
  // x0, y0 -> a = x0+y0, b = x0*y0, c = a+b
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(5));
  auto x0 = store.create_state<double, double>(2.0);
  auto y0 = store.create_state<double, double>(3.0);

  auto a = x0 + y0;  // 5
  auto b = x0 * y0;  // 6
  auto c = a + b;    // 11

  EXPECT_NEAR(c.get(), 11.0, 1e-14);

  gretl::set_as_objective(c);
  store.back_prop();

  // dc/dx0 = 1 + y0 = 4
  // dc/dy0 = 1 + x0 = 3
  EXPECT_NEAR(x0.get_dual(), 4.0, 1e-14);
  EXPECT_NEAR(y0.get_dual(), 3.0, 1e-14);
}

TEST(DAGTopology, SkipConnection)
{
  // x0 -> a -> b -> c, but also x0 -> c directly (skip connection)
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(5));
  auto x0 = store.create_state<double, double>(2.0);

  auto a = gretl::axpb(2.0, x0, 0.0);  // 4
  auto b = gretl::axpb(3.0, a, 0.0);   // 12
  auto c = b + x0;                     // 12 + 2 = 14

  EXPECT_NEAR(c.get(), 14.0, 1e-14);

  gretl::set_as_objective(c);
  store.back_prop();

  // dc/dx0 = dc/db * db/da * da/dx0 + 1 = 1*3*2 + 1 = 7
  EXPECT_NEAR(x0.get_dual(), 7.0, 1e-14);
}

TEST(DAGTopology, WideFanoutThenMerge)
{
  // x0 fans out to 10 branches, all merge back together by summation.
  // Stresses usageCount tracking on x0.
  int W = 10;
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(W + 2)));
  auto x0 = store.create_state<double, double>(1.5);

  State<double> sum = gretl::axpb(1.0, x0, 0.0);
  for (int i = 1; i < W; ++i) {
    auto branch = gretl::axpb(static_cast<double>(i + 1), x0, 0.0);
    sum = sum + branch;
  }
  // sum = x0 + 2*x0 + 3*x0 + ... + W*x0 = x0 * W*(W+1)/2
  double coeff = static_cast<double>(W * (W + 1)) / 2.0;
  EXPECT_NEAR(sum.get(), coeff * 1.5, 1e-12);

  gretl::set_as_objective(sum);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), coeff, 1e-12);
}

TEST(DAGTopology, SkipConnectionTightBudget)
{
  // Skip connection with minimal checkpoint budget.
  // x0 is used both early and late in the graph.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(2));
  auto x0 = store.create_state<double, double>(2.0);

  auto a = gretl::axpb(2.0, x0, 0.0);
  auto b = gretl::axpb(3.0, a, 0.0);
  auto c = gretl::axpb(4.0, b, 0.0);
  // Now use x0 again (skip connection) — x0 must be available
  auto d = c + x0;

  EXPECT_NEAR(d.get(), 4.0 * 3.0 * 2.0 * 2.0 + 2.0, 1e-14);

  gretl::set_as_objective(d);
  store.back_prop();

  // dd/dx0 = 4*3*2 + 1 = 25
  EXPECT_NEAR(x0.get_dual(), 25.0, 1e-14);
}

// ---------------------------------------------------------------------------
// TEST SUITE: ResetAndRerun
// Test reset() and re-evaluation patterns
// ---------------------------------------------------------------------------

TEST(ResetAndRerun, ResetAndRerunGraph)
{
  // Build graph, backprop, reset, change persistent input, re-evaluate, backprop again.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(5));
  auto x0 = store.create_state<double, double>(2.0);

  auto a = gretl::axpb(3.0, x0, 1.0);  // 7
  auto b = gretl::axpb(2.0, a, -1.0);  // 13
  gretl::set_as_objective(b);
  store.back_prop();

  // db/dx0 = 2*3 = 6
  EXPECT_NEAR(x0.get_dual(), 6.0, 1e-14);

  // Reset and re-run with a different x0
  store.reset();
  x0.set(5.0);
  store.reset_for_backprop();
  b.set_dual(1.0);
  store.back_prop();

  // After reset+re-eval: a = 3*5+1=16, b = 2*16-1=31
  EXPECT_NEAR(b.get(), 31.0, 1e-14);
  EXPECT_NEAR(x0.get_dual(), 6.0, 1e-14);  // gradient is independent of x0 for linear graph
}

TEST(ResetAndRerun, ResetGraphAndRebuild)
{
  // Previously crashed: after back_prop(), currentStep_ was 0 but resize()
  // asserted newSize <= currentStep_. Fixed by restoring currentStep_ first.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(5));
  auto x0 = store.create_state<double, double>(2.0);

  // First graph: x0 -> 3*x0+1
  {
    auto a = gretl::axpb(3.0, x0, 1.0);
    gretl::set_as_objective(a);
    store.back_prop();
    EXPECT_NEAR(x0.get_dual(), 3.0, 1e-14);
  }

  // Reset and build a completely different graph
  store.reset_graph();

  // Second graph: x0 -> 5*x0+2
  {
    auto b = gretl::axpb(5.0, x0, 2.0);
    gretl::set_as_objective(b);
    store.back_prop();
    EXPECT_NEAR(x0.get_dual(), 5.0, 1e-14);
  }
}

// ---------------------------------------------------------------------------
// TEST SUITE: NonlinearStress
// Nonlinear operations that stress checkpoint recomputation correctness
// ---------------------------------------------------------------------------

TEST(NonlinearStress, MultiplyChainTightBudget)
{
  // x = x * x iteratively (squaring). Very sensitive to recomputation errors
  // because the function is nonlinear.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(2));
  auto x0 = store.create_state<double, double>(1.1);

  auto x = gretl::axpb(1.0, x0, 0.0);
  int N = 5;
  for (int i = 0; i < N; ++i) {
    x = x * x;  // squaring
  }

  // x0^(2^N)
  double expected = std::pow(1.1, std::pow(2.0, N));
  EXPECT_NEAR(x.get(), expected, 1e-6);

  gretl::set_as_objective(x);
  store.back_prop();

  // d/dx0(x0^(2^N)) = 2^N * x0^(2^N - 1)
  double pow2N = std::pow(2.0, N);
  double expectedGrad = pow2N * std::pow(1.1, pow2N - 1);
  EXPECT_NEAR(x0.get_dual(), expectedGrad, expectedGrad * 1e-6);
}

TEST(NonlinearStress, MixedLinearNonlinear)
{
  // Alternating linear and nonlinear ops.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(4));
  auto x0 = store.create_state<double, double>(0.5);
  auto y0 = store.create_state<double, double>(0.3);

  auto a = x0 + y0;  // 0.8
  auto b = a * x0;   // 0.8 * 0.5 = 0.4
  auto c = b + y0;   // 0.4 + 0.3 = 0.7
  auto d = c * a;    // 0.7 * 0.8 = 0.56

  EXPECT_NEAR(d.get(), 0.56, 1e-14);

  gretl::set_as_objective(d);
  store.back_prop();

  // Numerical gradient check via finite differences
  double eps = 1e-7;

  // Perturb x0
  {
    double x0v = 0.5, y0v = 0.3;
    auto f = [&](double x) {
      double a_ = x + y0v;
      double b_ = a_ * x;
      double c_ = b_ + y0v;
      return c_ * a_;
    };
    double fd = (f(x0v + eps) - f(x0v - eps)) / (2.0 * eps);
    EXPECT_NEAR(x0.get_dual(), fd, 1e-5);
  }

  // Perturb y0
  {
    double x0v = 0.5, y0v = 0.3;
    auto f = [&](double y) {
      double a_ = x0v + y;
      double b_ = a_ * x0v;
      double c_ = b_ + y;
      return c_ * a_;
    };
    double fd = (f(y0v + eps) - f(y0v - eps)) / (2.0 * eps);
    EXPECT_NEAR(y0.get_dual(), fd, 1e-5);
  }
}

// ---------------------------------------------------------------------------
// TEST SUITE: LargeGraphStress
// Push the limits with large graphs and various budget ratios
// ---------------------------------------------------------------------------

TEST(LargeGraphStress, Chain100Budget3)
{
  int N = 100;
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(1.0);

  auto x = x0;
  for (int i = 0; i < N; ++i) {
    x = gretl::axpb(0.99, x, 0.01);
  }

  gretl::set_as_objective(x);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), std::pow(0.99, N), 1e-8);
}

TEST(LargeGraphStress, Chain200Budget5)
{
  int N = 200;
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(5));
  auto x0 = store.create_state<double, double>(1.0);

  auto x = x0;
  for (int i = 0; i < N; ++i) {
    x = gretl::axpb(0.99, x, 0.01);
  }

  gretl::set_as_objective(x);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), std::pow(0.99, N), 1e-6);
}

TEST(LargeGraphStress, NonlinearChain50Budget2)
{
  // Nonlinear chain with very tight budget: exercises checkpoint correctness
  // for nonlinear functions where recomputation must match original.
  int N = 50;
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(2));
  auto x0 = store.create_state<double, double>(0.5);

  auto x = gretl::axpb(1.0, x0, 0.0);
  for (int i = 0; i < N; ++i) {
    // x = x^2 + 0.1 (bounded for x0=0.5)
    auto xsq = x * x;
    x = gretl::axpb(1.0, xsq, 0.1);
  }

  gretl::set_as_objective(x);
  store.back_prop();

  // Verify gradient via finite differences
  double eps = 1e-7;
  auto eval_chain = [&](double x0v) {
    double xv = x0v;
    for (int i = 0; i < N; ++i) {
      xv = xv * xv + 0.1;
    }
    return xv;
  };
  double fd = (eval_chain(0.5 + eps) - eval_chain(0.5 - eps)) / (2.0 * eps);
  // Use relative tolerance since values can be large
  if (std::abs(fd) > 1e-10) {
    EXPECT_NEAR(x0.get_dual() / fd, 1.0, 1e-3);
  }
}

// ---------------------------------------------------------------------------
// TEST SUITE: VectorStateStress
// Same patterns but with vector states, exercising the initialize_zero_dual
// ---------------------------------------------------------------------------

TEST(VectorStateStress, ChainInSubfunc)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  std::vector<double> data = {1.0, 2.0, 3.0};
  auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);

  // Build chain: x = 0.5*x
  auto x = gretl::copy(x0);
  for (int i = 0; i < 10; ++i) {
    x = x * 0.5;
  }

  auto norm = gretl::inner_product(x, x);
  gretl::set_as_objective(norm);
  store.back_prop();

  // norm = (x0/2^10)^2 summed = sum(x0_i^2) / 2^20
  // d(norm)/dx0_i = 2*x0_i / 2^20
  double factor = 1.0 / std::pow(2.0, 20);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(x0.get_dual()[i], 2.0 * data[i] * factor, 1e-12);
  }
}

TEST(VectorStateStress, DiamondWithVectors)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(5));
  std::vector<double> dataA = {1.0, 2.0};
  std::vector<double> dataB = {3.0, 4.0};

  auto a = store.create_state(dataA, gretl::vec::initialize_zero_dual);
  auto b = store.create_state(dataB, gretl::vec::initialize_zero_dual);

  auto c = a + b;                       // {4, 6}
  auto d = a * b;                       // {3, 8}
  auto e = c + d;                       // {7, 14}
  auto f = gretl::inner_product(e, e);  // 49+196=245

  gretl::set_as_objective(f);
  store.back_prop();

  // Check via finite differences
  double eps = 1e-7;
  auto eval_f = [](double a0, double a1, double b0, double b1) {
    double e0 = (a0 + b0) + (a0 * b0);
    double e1 = (a1 + b1) + (a1 * b1);
    return e0 * e0 + e1 * e1;
  };

  double df_da0 = (eval_f(1.0 + eps, 2.0, 3.0, 4.0) - eval_f(1.0 - eps, 2.0, 3.0, 4.0)) / (2.0 * eps);
  double df_da1 = (eval_f(1.0, 2.0 + eps, 3.0, 4.0) - eval_f(1.0, 2.0 - eps, 3.0, 4.0)) / (2.0 * eps);
  double df_db0 = (eval_f(1.0, 2.0, 3.0 + eps, 4.0) - eval_f(1.0, 2.0, 3.0 - eps, 4.0)) / (2.0 * eps);

  EXPECT_NEAR(a.get_dual()[0], df_da0, 1e-5);
  EXPECT_NEAR(a.get_dual()[1], df_da1, 1e-5);
  EXPECT_NEAR(b.get_dual()[0], df_db0, 1e-5);
}

// ---------------------------------------------------------------------------
// TEST SUITE: MultiPersistentState
// Multiple persistent states with various dependency patterns
// ---------------------------------------------------------------------------

TEST(MultiPersistentState, ThreeInputsDeepGraph)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(4));
  auto x = store.create_state<double, double>(1.0);
  auto y = store.create_state<double, double>(2.0);
  auto z = store.create_state<double, double>(3.0);

  // Deep graph mixing all three inputs
  auto a = x + y;  // 3
  auto b = a * z;  // 9
  auto c = b + x;  // 10
  auto d = c * y;  // 20
  auto e = d + z;  // 23
  auto f = e * x;  // 23

  gretl::set_as_objective(f);
  store.back_prop();

  // Numerical gradient check
  double eps = 1e-7;
  auto eval = [](double xv, double yv, double zv) {
    double a_ = xv + yv;
    double b_ = a_ * zv;
    double c_ = b_ + xv;
    double d_ = c_ * yv;
    double e_ = d_ + zv;
    return e_ * xv;
  };

  double df_dx = (eval(1.0 + eps, 2.0, 3.0) - eval(1.0 - eps, 2.0, 3.0)) / (2.0 * eps);
  double df_dy = (eval(1.0, 2.0 + eps, 3.0) - eval(1.0, 2.0 - eps, 3.0)) / (2.0 * eps);
  double df_dz = (eval(1.0, 2.0, 3.0 + eps) - eval(1.0, 2.0, 3.0 - eps)) / (2.0 * eps);

  EXPECT_NEAR(x.get_dual(), df_dx, 1e-5);
  EXPECT_NEAR(y.get_dual(), df_dy, 1e-5);
  EXPECT_NEAR(z.get_dual(), df_dz, 1e-5);
}

TEST(MultiPersistentState, RepeatedUseOfAllInputs)
{
  // All three persistent inputs used at multiple points in the graph.
  // Exercises passthrough and lastStepUsed tracking.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x = store.create_state<double, double>(0.5);
  auto y = store.create_state<double, double>(0.3);
  auto z = store.create_state<double, double>(0.7);

  auto a = x * y;  // early use of x, y
  auto b = a + z;  // early use of z
  auto c = b * x;  // x used again (skip)
  auto d = c + y;  // y used again (skip)
  auto e = d * z;  // z used again (skip)

  gretl::set_as_objective(e);
  store.back_prop();

  double eps = 1e-7;
  auto eval = [](double xv, double yv, double zv) {
    double a_ = xv * yv;
    double b_ = a_ + zv;
    double c_ = b_ * xv;
    double d_ = c_ + yv;
    return d_ * zv;
  };

  EXPECT_NEAR(x.get_dual(), (eval(0.5 + eps, 0.3, 0.7) - eval(0.5 - eps, 0.3, 0.7)) / (2 * eps), 1e-5);
  EXPECT_NEAR(y.get_dual(), (eval(0.5, 0.3 + eps, 0.7) - eval(0.5, 0.3 - eps, 0.7)) / (2 * eps), 1e-5);
  EXPECT_NEAR(z.get_dual(), (eval(0.5, 0.3, 0.7 + eps) - eval(0.5, 0.3, 0.7 - eps)) / (2 * eps), 1e-5);
}

// ---------------------------------------------------------------------------
// TEST SUITE: EdgeCases
// Boundary conditions and unusual patterns
// ---------------------------------------------------------------------------

TEST(EdgeCases, SingleStepGraph)
{
  // Minimal graph: one persistent state, one derived state.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(1));
  auto x0 = store.create_state<double, double>(5.0);
  auto y = gretl::axpb(2.0, x0, 3.0);

  EXPECT_NEAR(y.get(), 13.0, 1e-14);

  gretl::set_as_objective(y);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), 2.0, 1e-14);
}

TEST(EdgeCases, TwoStepGraph)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(1));
  auto x0 = store.create_state<double, double>(5.0);
  auto a = gretl::axpb(2.0, x0, 1.0);
  auto b = gretl::axpb(3.0, a, -1.0);

  EXPECT_NEAR(b.get(), 3.0 * (2.0 * 5.0 + 1.0) - 1.0, 1e-14);

  gretl::set_as_objective(b);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), 6.0, 1e-14);
}

TEST(EdgeCases, StateUsedOnceVsManyTimes)
{
  // Compare: state used as upstream once vs. multiple times in the same operation's dependency.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(4));
  auto x0 = store.create_state<double, double>(3.0);

  // x0 * x0 = x0^2  (x0 used twice as upstream)
  auto sq = x0 * x0;
  EXPECT_NEAR(sq.get(), 9.0, 1e-14);

  gretl::set_as_objective(sq);
  store.back_prop();

  // d(x0^2)/dx0 = 2*x0 = 6
  EXPECT_NEAR(x0.get_dual(), 6.0, 1e-14);
}

TEST(EdgeCases, DeepChainSingleBudget)
{
  // Budget of exactly 1 with a deep chain.
  // This is the absolute minimum and forces full recomputation on every reverse step.
  int N = 30;
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(1));
  auto x0 = store.create_state<double, double>(1.0);

  auto x = x0;
  for (int i = 0; i < N; ++i) {
    x = gretl::axpb(0.9, x, 0.1);
  }

  gretl::set_as_objective(x);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), std::pow(0.9, N), 1e-8);
}

// ---------------------------------------------------------------------------
// TEST SUITE: TemporaryStateInSubFunction
// Specifically exercise the pattern of creating states in called functions
// where the State objects are temporaries that go out of scope.
// ---------------------------------------------------------------------------

// Helper: creates a temporary state, uses it, returns something derived from it
static State<double> create_use_and_discard(const State<double>& input, double scale)
{
  auto temp = gretl::axpb(scale, input, 0.0);  // temp goes out of scope after return
  auto temp2 = gretl::axpb(0.5, temp, 0.0);    // temp2 goes out of scope too
  return gretl::axpb(1.0, temp2, 1.0);         // return final
}

TEST(TemporaryStateInSubFunction, BasicPattern)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(4.0);

  // Call sub-function 3 times in sequence
  auto r1 = create_use_and_discard(x0, 2.0);  // (2*4)*0.5 + 1 = 5
  auto r2 = create_use_and_discard(r1, 3.0);  // (3*5)*0.5 + 1 = 8.5
  auto r3 = create_use_and_discard(r2, 1.0);  // (1*8.5)*0.5 + 1 = 5.25

  EXPECT_NEAR(r3.get(), 5.25, 1e-14);

  gretl::set_as_objective(r3);
  store.back_prop();

  // Each call: f(x) = scale*x*0.5 + 1, df/dx = scale*0.5
  // Chain: dr3/dx0 = (1*0.5) * (3*0.5) * (2*0.5) = 0.5 * 1.5 * 1.0 = 0.75
  EXPECT_NEAR(x0.get_dual(), 0.75, 1e-12);
}

TEST(TemporaryStateInSubFunction, LoopedSubFuncCalls)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(2));
  auto x0 = store.create_state<double, double>(2.0);

  auto x = x0;
  int N = 10;
  for (int i = 0; i < N; ++i) {
    x = create_use_and_discard(x, 1.0);
    // Each: x -> 1.0*x*0.5 + 1 = 0.5*x + 1
  }

  double expected = 2.0;
  for (int i = 0; i < N; ++i) expected = 0.5 * expected + 1.0;
  EXPECT_NEAR(x.get(), expected, 1e-10);

  gretl::set_as_objective(x);
  store.back_prop();

  // Each sub-function has 3 graph steps internally but df/dx = 0.5 per call
  // Total: 0.5^N
  EXPECT_NEAR(x0.get_dual(), std::pow(0.5, N), 1e-10);
}

TEST(TemporaryStateInSubFunction, MixedScopeTempsAndPersisted)
{
  // Some states held externally, others are temporaries in sub-functions.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(4));
  auto x0 = store.create_state<double, double>(3.0);

  auto held = gretl::axpb(2.0, x0, 0.0);  // 6.0, held in this scope

  // Sub-function creates and discards temporaries
  auto fromSub = create_use_and_discard(held, 1.0);  // 0.5*6 + 1 = 4.0

  // Use both held and fromSub
  auto result = held + fromSub;  // 6 + 4 = 10

  EXPECT_NEAR(result.get(), 10.0, 1e-14);

  gretl::set_as_objective(result);
  store.back_prop();

  // dresult/dx0 = dheld/dx0 + dfromSub/dx0
  //             = 2 + (0.5 * 2) = 2 + 1 = 3
  EXPECT_NEAR(x0.get_dual(), 3.0, 1e-14);
}

// ---------------------------------------------------------------------------
// TEST SUITE: DestructorTiming
// Focus on ordering of destructor calls relative to graph state
// ---------------------------------------------------------------------------

TEST(DestructorTiming, StateDestroyedAfterBackprop)
{
  // State objects going out of scope after back_prop has completed.
  // Their destructors call try_to_free, which should handle the
  // post-backprop state gracefully.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(1.0);

  {
    auto a = gretl::axpb(2.0, x0, 1.0);
    auto b = gretl::axpb(3.0, a, -1.0);
    auto c = b + a;

    gretl::set_as_objective(c);
    store.back_prop();

    EXPECT_NEAR(x0.get_dual(), 8.0, 1e-14);  // dc/dx0 = 3*2 + 2 = 8
    // a, b, c go out of scope here — destructors must not crash
  }

  SUCCEED();
}

TEST(DestructorTiming, InterleavedCreationAndDestruction)
{
  // Create states, let some go out of scope, create more.
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(5));
  auto x0 = store.create_state<double, double>(1.0);

  // Use a lambda to create a state derived from an intermediate that goes out of scope
  auto result = [&]() {
    auto a = gretl::axpb(2.0, x0, 0.0);  // 2.0
    return gretl::axpb(3.0, a, 0.0);     // 6.0
    // a goes out of scope — but returned state still depends on it in the graph
  }();

  auto final_result = gretl::axpb(4.0, result, 0.0);  // 24.0
  EXPECT_NEAR(final_result.get(), 24.0, 1e-14);

  gretl::set_as_objective(final_result);
  store.back_prop();

  EXPECT_NEAR(x0.get_dual(), 24.0, 1e-14);
}

TEST(DestructorTiming, StateOutlivesSharedOwner)
{
  // Regression test for teardown ordering when the last DataStore owner is a
  // shared_ptr held by another object, but a copied State survives longer.
  //
  // Before the lifetime-token fix, the escaped state's destructor would call
  // try_to_free() through a dangling DataStore* after owner destruction. ASan
  // reliably reported this as a heap-use-after-free at test teardown.
  struct SharedOwner {
    std::shared_ptr<DataStore> store = std::make_shared<DataStore>(std::make_unique<gretl::WangCheckpointStrategy>(3));
  };

  std::unique_ptr<State<double>> escaped;

  {
    auto owner = std::make_unique<SharedOwner>();
    auto x0 = owner->store->create_state<double, double>(2.0);
    auto y = gretl::axpb(3.0, x0, 1.0);

    EXPECT_NEAR(y.get(), 7.0, 1e-14);

    escaped = std::make_unique<State<double>>(y);
  }

  // No explicit assertions needed here. The regression is in teardown:
  // escaped is destroyed after owner/store are already gone.
  SUCCEED();
}

TEST(DestructorTiming, LastExternalHandleDestructionStillEvictsInactiveState)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(2.0);

  gretl::Int step = 0;

  {
    auto held = gretl::axpb(3.0, x0, 1.0);
    step = held.step();

    // Materialize both primal and dual so the test can verify they are dropped.
    EXPECT_NEAR(held.get(), 7.0, 1e-14);
    EXPECT_NEAR(held.get_dual(), 0.0, 1e-14);

    store.active_[step] = false;
    store.usageCount_[step] = 0;

    // With an external handle alive, try_to_free() should not evict yet.
    store.try_to_free(step);
    EXPECT_TRUE(store.states_[step]->primal());
    EXPECT_TRUE(store.duals_[step]);
  }

  // Once the last external handle dies, its destructor should retry eviction.
  EXPECT_FALSE(store.states_[step]->primal());
  EXPECT_FALSE(store.duals_[step]);
}

TEST(AssignmentOperator, ReassigningLastExternalHandleStillEvictsInactiveState)
{
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(3));
  auto x0 = store.create_state<double, double>(2.0);

  auto held = gretl::axpb(3.0, x0, 1.0);
  gretl::Int old_step = held.step();

  EXPECT_NEAR(held.get(), 7.0, 1e-14);
  EXPECT_NEAR(held.get_dual(), 0.0, 1e-14);

  store.active_[old_step] = false;
  store.usageCount_[old_step] = 0;

  // The old step is inactive, but the external handle still keeps it alive.
  store.try_to_free(old_step);
  EXPECT_TRUE(store.states_[old_step]->primal());
  EXPECT_TRUE(store.duals_[old_step]);

  held = x0;

  EXPECT_FALSE(store.states_[old_step]->primal());
  EXPECT_FALSE(store.duals_[old_step]);
}

// ---------------------------------------------------------------------------
// Helper: high-resolution timer
// ---------------------------------------------------------------------------
static double elapsed_ms(std::chrono::steady_clock::time_point start)
{
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// ---------------------------------------------------------------------------
// TEST SUITE: PerformanceScaling
// Measure how construction + backprop time scales with graph size and budget.
// These tests print timing info and verify correctness. They expose
// super-linear scaling in passthrough tracking, checkpoint search, and
// recomputation overhead.
// ---------------------------------------------------------------------------

// Count forward evaluations to measure recomputation overhead
static int g_eval_count = 0;

static State<double> counted_step(const State<double>& x)
{
  auto y = x.clone({x});

  y.set_eval([](const gretl::UpstreamStates& ups, gretl::DownstreamState& ds) {
    ++g_eval_count;
    ds.set(0.99 * ups[0].get<double>() + 0.01);
  });

  y.set_vjp([](gretl::UpstreamStates& ups, const gretl::DownstreamState& ds) {
    ups[0].get_dual<double, double>() += 0.99 * ds.get_dual<double, double>();
  });

  return y.finalize();
}

TEST(PerformanceScaling, LinearChainRecomputationCount)
{
  // Measure how many forward evaluations happen during backprop
  // for various chain lengths and checkpoint budgets.
  // Optimal (Wang 2009) is O(N * log(N) / S) recomputations for S checkpoints.
  struct Config {
    int N;
    size_t budget;
  };
  std::vector<Config> configs = {
      {50, 3}, {50, 5}, {50, 10}, {50, 50}, {200, 3}, {200, 5}, {200, 10}, {200, 50}, {500, 5}, {500, 10}, {500, 50},
  };

  std::cout << "\n--- Recomputation counts: N steps, S budget, fwd_evals (backprop), ratio=evals/N ---\n";

  for (auto& cfg : configs) {
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(cfg.budget));
    auto x0 = store.create_state<double, double>(1.0);

    g_eval_count = 0;
    auto x = x0;
    for (int i = 0; i < cfg.N; ++i) {
      x = counted_step(x);
    }
    int fwd_evals = g_eval_count;

    g_eval_count = 0;
    gretl::set_as_objective(x);
    store.back_prop();
    int backprop_evals = g_eval_count;

    double ratio = static_cast<double>(backprop_evals) / cfg.N;
    std::cout << "  N=" << cfg.N << " S=" << cfg.budget << " fwd=" << fwd_evals << " back=" << backprop_evals
              << " ratio=" << ratio << "\n";

    // Gradient should still be correct
    EXPECT_NEAR(x0.get_dual(), std::pow(0.99, cfg.N), std::pow(0.99, cfg.N) * 1e-6);

    // Sanity: backprop evals should be >= N-1 (minimum: everything in memory)
    // and forward pass should be exactly N
    EXPECT_EQ(fwd_evals, cfg.N);
  }
  std::cout << "---\n";
}

TEST(PerformanceScaling, ConstructionTimeScaling)
{
  // Measure graph construction time as N grows.
  // Key concern: passthrough loop in add_state is O(distance_to_last_use)
  // per upstream, which can make construction O(N^2) for skip connections.
  std::cout << "\n--- Construction time scaling (linear chain, no skip connections) ---\n";

  std::vector<int> sizes = {100, 500, 1000, 2000, 5000};
  double prev_ms = 0;

  for (int N : sizes) {
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(10));
    auto x0 = store.create_state<double, double>(1.0);

    auto start = std::chrono::steady_clock::now();
    auto x = x0;
    for (int i = 0; i < N; ++i) {
      x = gretl::axpb(0.99, x, 0.01);
    }
    double ms = elapsed_ms(start);

    double ratio = (prev_ms > 0) ? ms / prev_ms : 0;
    std::cout << "  N=" << N << " construct=" << ms << "ms" << (prev_ms > 0 ? " ratio=" + std::to_string(ratio) : "")
              << "\n";
    prev_ms = ms;

    // Should still produce correct results
    gretl::set_as_objective(x);
    store.back_prop();
    EXPECT_NEAR(x0.get_dual(), std::pow(0.99, N), std::pow(0.99, N) * 1e-4);
  }
  std::cout << "---\n";
}

TEST(PerformanceScaling, BackpropTimeVsBudget)
{
  // Fixed graph size, vary budget. Measures the tradeoff between
  // memory (checkpoint slots) and recomputation time.
  int N = 500;
  std::cout << "\n--- Backprop time vs budget (N=" << N << ") ---\n";

  std::vector<size_t> budgets = {2, 3, 5, 10, 20, 50, 100, 500};

  for (size_t budget : budgets) {
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(budget));
    auto x0 = store.create_state<double, double>(1.0);

    auto x = x0;
    for (int i = 0; i < N; ++i) {
      x = counted_step(x);
    }

    g_eval_count = 0;
    auto start = std::chrono::steady_clock::now();
    gretl::set_as_objective(x);
    store.back_prop();
    double ms = elapsed_ms(start);

    std::cout << "  budget=" << budget << " backprop=" << ms << "ms"
              << " recomps=" << g_eval_count << "\n";

    EXPECT_NEAR(x0.get_dual(), std::pow(0.99, N), std::pow(0.99, N) * 1e-4);
  }
  std::cout << "---\n";
}

TEST(PerformanceScaling, SkipConnectionPassthroughOverhead)
{
  // Build a chain where the initial persistent state is used at every step
  // (persistent states skip the passthrough loop, so this should be fast).
  // Then build a chain where a NON-persistent state at step 1 is used
  // at every step (creating long passthrough lists).
  // Compare construction times to measure passthrough overhead.
  int N = 1000;
  std::cout << "\n--- Skip connection passthrough overhead (N=" << N << ") ---\n";

  // Case 1: persistent x0 used at every step (no passthrough overhead)
  {
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(N + 2)));
    auto x0 = store.create_state<double, double>(1.0);

    auto start = std::chrono::steady_clock::now();
    auto x = gretl::axpb(1.0, x0, 0.0);
    for (int i = 1; i < N; ++i) {
      x = x + x0;  // persistent upstream: no passthroughs
    }
    double ms = elapsed_ms(start);
    std::cout << "  persistent_skip: " << ms << "ms\n";

    gretl::set_as_objective(x);
    store.back_prop();
    // x = N*x0, dx/dx0 = N
    EXPECT_NEAR(x0.get_dual(), static_cast<double>(N), 1e-6);
  }

  // Case 2: non-persistent step 1 used at every step (passthrough overhead)
  {
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(N + 2)));
    auto x0 = store.create_state<double, double>(1.0);
    auto base = gretl::axpb(1.0, x0, 0.0);  // step 1 (non-persistent)

    auto start = std::chrono::steady_clock::now();
    auto x = gretl::axpb(1.0, base, 0.0);
    for (int i = 1; i < N; ++i) {
      x = x + base;  // non-persistent upstream: passthroughs grow linearly
    }
    double ms = elapsed_ms(start);
    std::cout << "  nonpersist_skip: " << ms << "ms\n";

    gretl::set_as_objective(x);
    store.back_prop();
    EXPECT_NEAR(x0.get_dual(), static_cast<double>(N), 1e-6);
  }

  std::cout << "---\n";
}

TEST(PerformanceScaling, CheckpointSetOperationOverhead)
{
  // The CheckpointManager uses std::set<Checkpoint>.
  // erase_step() does a linear scan O(S) per call.
  // contains_step() also does O(S) linear scan.
  // For large budgets, this could become a bottleneck.
  // Measure backprop time with a huge budget to isolate this cost.
  std::cout << "\n--- Checkpoint set overhead (large budget) ---\n";

  std::vector<int> sizes = {100, 500, 1000, 2000};

  for (int N : sizes) {
    // Budget = N (everything fits, no recomputation)
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(N)));
    auto x0 = store.create_state<double, double>(1.0);

    auto x = x0;
    for (int i = 0; i < N; ++i) {
      x = gretl::axpb(0.99, x, 0.01);
    }

    auto start = std::chrono::steady_clock::now();
    gretl::set_as_objective(x);
    store.back_prop();
    double ms = elapsed_ms(start);

    std::cout << "  N=S=" << N << " backprop=" << ms << "ms\n";

    EXPECT_NEAR(x0.get_dual(), std::pow(0.99, N), std::pow(0.99, N) * 1e-4);
  }
  std::cout << "---\n";
}

TEST(PerformanceScaling, LargeVectorStateScaling)
{
  // Measure performance with large vector states to see if the
  // type-erased std::any copies dominate runtime.
  std::cout << "\n--- Large vector state scaling ---\n";

  std::vector<size_t> vec_sizes = {10, 100, 1000, 10000};
  int N = 100;

  for (size_t S : vec_sizes) {
    std::vector<double> data(S, 1.0);

    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(10));
    auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);

    auto start = std::chrono::steady_clock::now();
    auto x = gretl::copy(x0);
    for (int i = 0; i < N; ++i) {
      x = x * 0.99;
    }
    auto norm = gretl::inner_product(x, x);
    gretl::set_as_objective(norm);
    store.back_prop();
    double ms = elapsed_ms(start);

    std::cout << "  vec_size=" << S << " N=" << N << " total=" << ms << "ms\n";

    // norm = S * (0.99^N)^2
    double expected_norm = static_cast<double>(S) * std::pow(0.99, 2 * N);
    EXPECT_NEAR(norm.get(), expected_norm, expected_norm * 1e-6);
  }
  std::cout << "---\n";
}

TEST(PerformanceScaling, WideFanoutScaling)
{
  // Measure how fan-out width affects construction and backprop time.
  // Each branch adds passthroughs and usageCount updates.
  std::cout << "\n--- Wide fan-out scaling ---\n";

  std::vector<int> widths = {5, 10, 20, 50, 100};

  for (int W : widths) {
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(2 * W + 5)));
    auto x0 = store.create_state<double, double>(1.0);

    auto start = std::chrono::steady_clock::now();
    State<double> sum = gretl::axpb(1.0, x0, 0.0);
    for (int i = 1; i < W; ++i) {
      auto branch = gretl::axpb(static_cast<double>(i + 1), x0, 0.0);
      sum = sum + branch;
    }
    double construct_ms = elapsed_ms(start);

    start = std::chrono::steady_clock::now();
    gretl::set_as_objective(sum);
    store.back_prop();
    double backprop_ms = elapsed_ms(start);

    double coeff = static_cast<double>(W * (W + 1)) / 2.0;
    std::cout << "  W=" << W << " construct=" << construct_ms << "ms"
              << " backprop=" << backprop_ms << "ms\n";

    EXPECT_NEAR(x0.get_dual(), coeff, 1e-8);
  }
  std::cout << "---\n";
}

TEST(PerformanceScaling, DeepDAGWithMultipleInputs)
{
  // A realistic-ish DAG pattern: two inputs, alternating linear operations,
  // deep chain. Measures scaling with depth.
  // Uses bounded linear ops (scale < 1) to avoid overflow.
  std::cout << "\n--- Deep DAG with 2 inputs ---\n";

  std::vector<int> depths = {50, 100, 200, 500, 1000};

  for (int N : depths) {
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(10));
    auto x = store.create_state<double, double>(0.5);
    auto y = store.create_state<double, double>(0.3);

    auto start = std::chrono::steady_clock::now();
    auto a = x + y;                                                // 0.8
    auto b = gretl::axpb(0.5, x, 0.0) + gretl::axpb(0.3, y, 0.0);  // 0.34
    for (int i = 0; i < N; ++i) {
      auto c = gretl::axpb(0.6, a, 0.0) + gretl::axpb(0.3, b, 0.0);  // bounded
      b = gretl::axpb(0.3, a, 0.0) + gretl::axpb(0.5, b, 0.0);       // bounded
      a = c;
    }
    auto result = a + b;
    double construct_ms = elapsed_ms(start);

    start = std::chrono::steady_clock::now();
    gretl::set_as_objective(result);
    store.back_prop();
    double backprop_ms = elapsed_ms(start);

    std::cout << "  depth=" << N << " construct=" << construct_ms << "ms"
              << " backprop=" << backprop_ms << "ms\n";

    // Verify finite gradients (bounded recurrence)
    EXPECT_TRUE(std::isfinite(x.get_dual()));
    EXPECT_TRUE(std::isfinite(y.get_dual()));
  }
  std::cout << "---\n";
}

TEST(PerformanceScaling, RepeatedResetAndBackprop)
{
  // Measure the cost of reset + re-evaluation cycles.
  // This is the pattern for iterative optimization.
  int N = 100;
  int iters = 20;
  DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(10));
  auto x0 = store.create_state<double, double>(1.0);

  auto x = x0;
  for (int i = 0; i < N; ++i) {
    x = gretl::axpb(0.99, x, 0.01);
  }

  std::cout << "\n--- Repeated reset+backprop (N=" << N << ", iters=" << iters << ") ---\n";

  gretl::set_as_objective(x);
  store.back_prop();
  double grad0 = x0.get_dual();

  auto start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < iters; ++iter) {
    store.reset();
    store.reset_for_backprop();
    x.set_dual(1.0);
    store.back_prop();
  }
  double ms = elapsed_ms(start);

  std::cout << "  total=" << ms << "ms"
            << " per_iter=" << ms / iters << "ms\n";
  std::cout << "---\n";

  // Gradient should be unchanged each time (linear graph)
  EXPECT_NEAR(x0.get_dual(), grad0, 1e-12);
}

// ---------------------------------------------------------------------------
// TEST SUITE: VectorBottleneck
// Detailed timing breakdown for State<vector<double>> to identify where
// the remaining performance bottlenecks are after Phase 1 (move overloads).
// ---------------------------------------------------------------------------

// Helper: a vector scale operation using const ref + move (best practice)
static VectorState vec_scale_move(const VectorState& a, double s)
{
  VectorState b = a.clone({a});

  b.set_eval([s](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const gretl::Vector& A = upstreams[0].get<gretl::Vector>();
    gretl::Vector C(A);  // copy-construct (avoids zero-init of Vector(sz))
    for (auto& v : C) {
      v *= s;
    }
    downstream.set(std::move(C));  // move into primal
  });

  b.set_vjp([s](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const gretl::Vector& Cbar = downstream.get_dual<gretl::Vector, gretl::Vector>();
    gretl::Vector& Abar = upstreams[0].get_dual<gretl::Vector, gretl::Vector>();
    for (size_t i = 0; i < Abar.size(); ++i) {
      Abar[i] += s * Cbar[i];
    }
  });

  return b.finalize();
}

// Helper: a vector scale using old pattern (copy input + copy output)
static VectorState vec_scale_copy(const VectorState& a, double s)
{
  VectorState b = a.clone({a});

  b.set_eval([s](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    gretl::Vector C = upstreams[0].get<gretl::Vector>();  // copy input
    for (auto& v : C) {
      v *= s;
    }
    downstream.set(C);  // copy output (no move)
  });

  b.set_vjp([s](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const gretl::Vector& Cbar = downstream.get_dual<gretl::Vector, gretl::Vector>();
    gretl::Vector& Abar = upstreams[0].get_dual<gretl::Vector, gretl::Vector>();
    for (size_t i = 0; i < Abar.size(); ++i) {
      Abar[i] += s * Cbar[i];
    }
  });

  return b.finalize();
}

TEST(VectorBottleneck, ProfileByPhase)
{
  // Break down total time into construction vs backprop for varying vector sizes.
  // Chain length fixed at N=100, budget=10.
  int N = 100;
  size_t budget = 10;

  std::cout << "\n--- Vector bottleneck profile (N=" << N << ", budget=" << budget << ") ---\n";
  std::cout << "  vec_size | construct_ms | backprop_ms | total_ms | bytes_per_vec\n";

  std::vector<size_t> vec_sizes = {100, 1000, 10000, 50000, 100000};

  for (size_t S : vec_sizes) {
    gretl::Vector data(S, 1.0);

    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(budget));
    auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);

    auto start = std::chrono::steady_clock::now();
    auto x = gretl::copy(x0);
    for (int i = 0; i < N; ++i) {
      x = vec_scale_move(x, 0.99);
    }
    double construct_ms = elapsed_ms(start);

    auto norm = gretl::inner_product(x, x);

    start = std::chrono::steady_clock::now();
    gretl::set_as_objective(norm);
    store.back_prop();
    double backprop_ms = elapsed_ms(start);

    size_t bytes = S * sizeof(double);
    std::cout << "  " << std::setw(8) << S << " | " << std::setw(12) << construct_ms << " | " << std::setw(11)
              << backprop_ms << " | " << std::setw(8) << (construct_ms + backprop_ms) << " | " << std::setw(13) << bytes
              << "\n";

    // Verify correctness
    double expected_norm = static_cast<double>(S) * std::pow(0.99, 2 * N);
    EXPECT_NEAR(norm.get(), expected_norm, expected_norm * 1e-6);
  }
  std::cout << "---\n";
}

TEST(VectorBottleneck, MoveVsCopy)
{
  // Compare the move-enabled eval path vs the copy path.
  // This directly measures the benefit of Phase 1 move overloads.
  int N = 100;
  size_t budget = 10;

  std::cout << "\n--- Move vs Copy comparison (N=" << N << ", budget=" << budget << ") ---\n";
  std::cout << "  vec_size | move_total_ms | copy_total_ms | speedup\n";

  std::vector<size_t> vec_sizes = {1000, 10000, 50000};

  for (size_t S : vec_sizes) {
    gretl::Vector data(S, 1.0);

    // Move path
    double move_ms;
    {
      DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(budget));
      auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);

      auto start = std::chrono::steady_clock::now();
      auto x = gretl::copy(x0);
      for (int i = 0; i < N; ++i) {
        x = vec_scale_move(x, 0.99);
      }
      auto norm = gretl::inner_product(x, x);
      gretl::set_as_objective(norm);
      store.back_prop();
      move_ms = elapsed_ms(start);

      double expected = static_cast<double>(S) * std::pow(0.99, 2 * N);
      EXPECT_NEAR(norm.get(), expected, expected * 1e-6);
    }

    // Copy path
    double copy_ms;
    {
      DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(budget));
      auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);

      auto start = std::chrono::steady_clock::now();
      auto x = gretl::copy(x0);
      for (int i = 0; i < N; ++i) {
        x = vec_scale_copy(x, 0.99);
      }
      auto norm = gretl::inner_product(x, x);
      gretl::set_as_objective(norm);
      store.back_prop();
      copy_ms = elapsed_ms(start);

      double expected = static_cast<double>(S) * std::pow(0.99, 2 * N);
      EXPECT_NEAR(norm.get(), expected, expected * 1e-6);
    }

    double speedup = copy_ms / move_ms;
    std::cout << "  " << std::setw(8) << S << " | " << std::setw(13) << move_ms << " | " << std::setw(13) << copy_ms
              << " | " << std::setw(7) << speedup << "x\n";
  }
  std::cout << "---\n";
}

TEST(VectorBottleneck, CloneOverhead)
{
  // Measure the cost of clone() for vector states. clone() always copies
  // the primal via make_shared<any>(*any_cast<T>(...)), so this is a
  // remaining copy that move overloads don't help.
  std::cout << "\n--- Clone overhead for vector states ---\n";
  std::cout << "  vec_size | clone_100x_ms | per_clone_us\n";

  std::vector<size_t> vec_sizes = {100, 1000, 10000, 50000};
  int N = 100;

  for (size_t S : vec_sizes) {
    gretl::Vector data(S, 1.0);

    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(N + 5)));
    auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);

    auto start = std::chrono::steady_clock::now();
    auto x = x0;
    for (int i = 0; i < N; ++i) {
      // clone is called inside vec_scale_move via clone({a})
      x = vec_scale_move(x, 1.0);
    }
    double total_ms = elapsed_ms(start);
    double per_clone_us = total_ms / N * 1000.0;

    std::cout << "  " << std::setw(8) << S << " | " << std::setw(13) << total_ms << " | " << std::setw(12)
              << per_clone_us << "\n";
  }
  std::cout << "---\n";
}

TEST(VectorBottleneck, CheckpointRecomputeVsNoRecompute)
{
  // Compare budget=N (all in memory, no recomputation) vs budget=5 (heavy recomputation).
  // This isolates the cost of checkpoint-driven recomputation for large vector states.
  int N = 100;

  std::cout << "\n--- Checkpoint recompute cost for vectors (N=" << N << ") ---\n";
  std::cout << "  vec_size | budget=N_ms | budget=5_ms | recomp_overhead\n";

  std::vector<size_t> vec_sizes = {1000, 10000, 50000};

  for (size_t S : vec_sizes) {
    gretl::Vector data(S, 1.0);

    // budget = N (no recomputation)
    double no_recomp_ms;
    {
      DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(N + 5)));
      auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);
      auto x = gretl::copy(x0);
      for (int i = 0; i < N; ++i) {
        x = vec_scale_move(x, 0.99);
      }
      auto norm = gretl::inner_product(x, x);
      gretl::set_as_objective(norm);

      auto start = std::chrono::steady_clock::now();
      store.back_prop();
      no_recomp_ms = elapsed_ms(start);
    }

    // budget = 5 (heavy recomputation)
    double recomp_ms;
    {
      DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(5));
      auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);
      auto x = gretl::copy(x0);
      for (int i = 0; i < N; ++i) {
        x = vec_scale_move(x, 0.99);
      }
      auto norm = gretl::inner_product(x, x);
      gretl::set_as_objective(norm);

      auto start = std::chrono::steady_clock::now();
      store.back_prop();
      recomp_ms = elapsed_ms(start);
    }

    double overhead = recomp_ms / no_recomp_ms;
    std::cout << "  " << std::setw(8) << S << " | " << std::setw(11) << no_recomp_ms << " | " << std::setw(11)
              << recomp_ms << " | " << std::setw(15) << overhead << "x\n";
  }
  std::cout << "---\n";
}

TEST(VectorBottleneck, GetPrimalCopyCost)
{
  // Measure the cost of get_primal<Vector> (which returns const ref, no copy)
  // vs the copy that happens inside eval when reading upstream.get<Vector>()
  // followed by modification. This isolates the read side.
  int N = 200;
  size_t S = 10000;

  std::cout << "\n--- get_primal read cost (N=" << N << ", vec_size=" << S << ") ---\n";

  gretl::Vector data(S, 1.0);

  // Pattern 1: Read-only (inner_product reads but doesn't copy vectors)
  double readonly_ms;
  {
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(N + 5)));
    auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);
    auto x = gretl::copy(x0);
    // Chain of inner_products: each reads 2 vectors but writes 1 double
    std::vector<State<double>> norms;
    for (int i = 0; i < N; ++i) {
      norms.push_back(gretl::inner_product(x, x));
    }
    // Sum all norms via a chain
    auto total = norms[0];
    for (int i = 1; i < N; ++i) {
      total = total + norms[static_cast<size_t>(i)];
    }
    gretl::set_as_objective(total);
    auto start = std::chrono::steady_clock::now();
    store.back_prop();
    readonly_ms = elapsed_ms(start);
  }

  // Pattern 2: Read + copy + write (scale operation copies the vector)
  double readwrite_ms;
  {
    DataStore store(std::make_unique<gretl::WangCheckpointStrategy>(static_cast<size_t>(N + 5)));
    auto x0 = store.create_state(data, gretl::vec::initialize_zero_dual);
    auto x = gretl::copy(x0);
    for (int i = 0; i < N; ++i) {
      x = vec_scale_move(x, 1.0);  // copies upstream vector, scales, moves result
    }
    auto norm = gretl::inner_product(x, x);
    gretl::set_as_objective(norm);
    auto start = std::chrono::steady_clock::now();
    store.back_prop();
    readwrite_ms = elapsed_ms(start);
  }

  std::cout << "  read-only (inner_product chain): " << readonly_ms << "ms\n";
  std::cout << "  read+copy+write (scale chain):   " << readwrite_ms << "ms\n";
  std::cout << "  copy+write overhead:             " << (readwrite_ms - readonly_ms) << "ms"
            << " (" << (readwrite_ms / readonly_ms) << "x)\n";
  std::cout << "---\n";
}
