// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Test inplace_vector serialization


#include <deal.II/base/std_cxx26/inplace_vector.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <numeric>
#include <vector>

#include "../tests.h"

#include "inplace_vector_common.h"

template <typename T>
void
test_serialization()
{
  std::array<T, 3> as{};
  if constexpr (std::is_same_v<T, int>)
    std::iota(as.begin(), as.end(), -11);
  if constexpr (std::is_same_v<T, std::vector<int>>)
    {
      as[0] = {1, 1, 2};
      as[1] = {42};
      as[2] = {3, 5};
    }

  std_cxx26::inplace_vector<T, 32> vec(3), vec2(3);
  std::copy(as.begin(), as.end(), vec.begin());
  std::ostringstream              out;
  boost::archive::binary_oarchive oarchive(out);
  oarchive << vec;
  const auto serialization = out.str();
  deallog << "size = " << serialization.size() << std::endl;

  std::istringstream              in(serialization);
  boost::archive::binary_iarchive iarchive(in);
  iarchive >> vec2;

  AssertThrow(vec == vec2, ExcInternalError());
}

int
main()
{
  initlog();

  deallog.push("serialization");
  deallog << std::endl;
  deallog.push("A");
  test_serialization<A>();
  deallog.pop();

  deallog << std::endl;
  deallog.push("int");
  test_serialization<int>();
  deallog.pop();

  deallog << std::endl;
  deallog.push("vector<int>");
  test_serialization<std::vector<int>>();
  deallog.pop();

  deallog.pop();
}
