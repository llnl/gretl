// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "state_base.hpp"
#include "upstream_state.hpp"

namespace gretl {

void StateBase::evaluate_forward()
{
  DownstreamState ds(&data_store(), step());
  UpstreamStates upstreams(data_store(), data_store().upstreamSteps_[step()]);
  data_store().evals_[step()](upstreams, ds);
  data_store().erase_step_state_data(step());
}

void StateBase::evaluate_vjp()
{
  const DownstreamState ds(&data_store(), step());
  UpstreamStates upstreams(data_store(), data_store().upstreamSteps_[step()]);
  data_store().vjps_[step()](upstreams, ds);
}

}  // namespace gretl
