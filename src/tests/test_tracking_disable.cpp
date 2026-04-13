#include <gtest/gtest.h>
#include "gretl/data_store.hpp"
#include "gretl/state.hpp"
#include "gretl/create_state.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"
#include "gretl/double_state.hpp"

using namespace gretl;

namespace {

State<double> picard_step(const State<double>& x, const State<double>& p, bool vjp_implemented = true)
{
  if (vjp_implemented) {
    return create_state<double, double>([](const double&) { return 0.0; },
                                        [](const double& x_val, const double& p_val) {
                                          // simple iteration: x = x * 0.5 + p
                                          return x_val * 0.5 + p_val;
                                        },
                                        [](const double& /*x_val*/, const double& /*p_val*/, const double& /*f_val*/,
                                           double& dx, double& dp, const double& df) {
                                          dx += 0.5 * df;
                                          dp += 1.0 * df;
                                        },
                                        x, p);
  } else {
    return create_state<double, double>(
        [](const double&) { return 0.0; },
        [](const double& x_val, const double& p_val) {
          // simple iteration: x = x * 0.5 + p
          return x_val * 0.5 + p_val;
        },
        [](const double&, const double&, const double&, double&, double&, const double&) {
          gretl_assert_msg(false, "VJP should not be called for stop-gradient nodes");
        },
        x, p);
  }
}

}  // namespace

TEST(GraphTracking, PicardIterationNoOpVjp)
{
  DataStore ds(std::make_unique<WangCheckpointStrategy>(3));

  // Parameter p
  auto p = ds.create_state<double, double>(0.1);

  // Initial guess x0
  auto x = ds.create_state<double, double>(1.0);

  // Iterate 10 times without tracking gradients (stop-gradient nodes)
  ds.set_gradients_enabled(false);
  for (int i = 0; i < 10; ++i) {
    x = picard_step(x, p, false);
  }

  // Re-enable gradients for the final step
  ds.set_gradients_enabled(true);

  // One final iteration to connect the parameter sensitivity
  auto x_final = picard_step(x, p, true);
  auto obj = set_as_objective(x_final);
  ds.finalize_graph();
  ds.back_prop();

  // Since x_final = 0.5 * x_10 + p,
  // dx_final / dp = 1.0 (from the direct dependency of x_final on p)
  // The dependency through x_10 is killed because x_10 has no-op VJP.
  EXPECT_EQ(p.get_dual(), 1.0);
}
