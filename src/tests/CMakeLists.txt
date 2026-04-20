# Copyright (c) Lawrence Livermore National Security, LLC and
# other Gretl Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(gretl_test_sources
    test_gretl_checkpoint.cpp
    test_gretl_checkpoint_compare.cpp
    test_gretl_dynamics.cpp
    test_gretl_graph.cpp
    test_gretl_robustness.cpp
    test_persistent_scope.cpp
    test_tracking_disable.cpp)

foreach(test ${gretl_test_sources})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable(NAME       ${test_name}
                       SOURCES    ${test}
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON gretl gtest
                       FOLDER     gretl/tests )

    blt_add_test(NAME    ${test_name}
                 COMMAND ${test_name} )
endforeach()

