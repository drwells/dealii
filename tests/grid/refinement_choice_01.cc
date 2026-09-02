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

#include <deal.II/grid/refinement_choice.h>

#include <sstream>
#include <vector>

#include "../tests.h"

// Test RefinementChoice's basic functionality (boolean operations)

template <int dim>
std::vector<RefinementChoice<dim>>
get_hypercube_choices()
{
  if constexpr (dim == 1)
    return {RefinementChoices::None<1>, RefinementChoices::Isotropic<1>};
  if constexpr (dim == 2)
    return {RefinementChoices::None<2>,
            RefinementChoices::CutX<2>,
            RefinementChoices::CutY<2>,
            RefinementChoices::Isotropic<2>};
  if constexpr (dim == 3)
    return {RefinementChoices::None<3>,
            RefinementChoices::CutX<3>,
            RefinementChoices::CutY<3>,
            RefinementChoices::CutXY<3>,
            RefinementChoices::CutZ<3>,
            RefinementChoices::CutXZ<3>,
            RefinementChoices::CutYZ<3>,
            RefinementChoices::Isotropic<3>};
}

std::vector<RefinementChoice<3>>
get_tetrahedron_choices()
{
  return {RefinementChoices::None<3>,
          RefinementChoices::Cut49<3>,
          RefinementChoices::Cut57<3>,
          RefinementChoices::Cut68<3>,
          RefinementChoices::Isotropic<3>};
}

template <int dim>
void
test_hypercube(const RefinementChoice<dim>               choice,
               const std::vector<RefinementChoice<dim>> &other_choices)
{
  const auto str = choice.to_string();
  deallog << str << std::endl;
  deallog << "  ~" << str << " = " << (~choice).to_string() << std::endl;
  deallog << "  bool(" << str << ") = " << std::boolalpha << bool(choice)
          << std::endl;
  for (const auto &other : other_choices)
    {
      // only some unions are allowed (e.g., not Cut49 | Cut57)
      try
        {
          const auto result = choice | other;
          deallog << "  " << str << " | " << other.to_string() << " = "
                  << result.to_string() << std::endl;
        }
      catch (const ExceptionBase &e)
        {
          std::ostringstream message;
          e.print_info(message);
          deallog << message.str() << std::endl;
        }
    }
  for (const auto &other : other_choices)
    deallog << "  " << str << " & " << other.to_string() << " = "
            << (choice & other).to_string() << std::endl;
}

int
main()
{
  initlog();

#ifndef Assert
  // TODO clean this up
#  warning "This test assumes Assert() is defined"
#endif

  deal_II_exceptions::disable_abort_on_exception();

  for (const auto &choice : get_hypercube_choices<1>())
    test_hypercube(choice, get_hypercube_choices<1>());
  for (const auto &choice : get_hypercube_choices<2>())
    test_hypercube(choice, get_hypercube_choices<2>());
  for (const auto &choice : get_hypercube_choices<3>())
    test_hypercube(choice, get_hypercube_choices<3>());
  for (const auto &choice : get_tetrahedron_choices())
    test_hypercube(choice, get_tetrahedron_choices());
}
