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

#include <deal.II/base/geometry_info.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/refinement_choice.h>
#include <deal.II/grid/tria.h>

#include <algorithm>
#include <vector>

#include "../tests.h"

// Test RefinementChoice's basic functionality (boolean operations)

template <int dim>
std::vector<RefinementChoice<dim>>
get_hypercube_choices()
{
  if constexpr (dim == 0)
    return {RefinementChoices::None<0>, RefinementChoices::Isotropic<0>};
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
          RefinementChoices::Cut68<3>};
}

// Unions of distinct specific tetrahedral refinements are not defined.
template <int dim>
bool
union_is_defined(const RefinementChoice<dim> a, const RefinementChoice<dim> b)
{
  if constexpr (dim == 3)
    {
      const auto tets            = get_tetrahedron_choices();
      const auto is_specific_tet = [&](const RefinementChoice<3> c) {
        return std::find(tets.begin() + 1, tets.end(), c) != tets.end();
      };
      if (is_specific_tet(a) && is_specific_tet(b) && a != b)
        return false;
    }
  return true;
}

template <int dim>
void
test(const RefinementChoice<dim>               choice,
     const std::vector<RefinementChoice<dim>> &other_choices)
{
  const auto str = choice.to_string();
  deallog << dim << "d " << str << " (" << int(choice) << ")" << std::endl;
  deallog << "  ~" << str << " = " << (~choice).to_string() << std::endl;
  deallog << "  bool(" << str << ") = " << std::boolalpha << bool(choice)
          << std::endl;
  deallog << "  is_anisotropic(" << str << ") = " << choice.is_anisotropic()
          << std::endl;
  for (const auto &other : other_choices)
    if (union_is_defined(choice, other))
      deallog << "  " << str << " | " << other.to_string() << " = "
              << (choice | other).to_string() << std::endl;
  for (const auto &other : other_choices)
    deallog << "  " << str << " & " << other.to_string() << " = "
            << (choice & other).to_string() << std::endl;
}

template <int dim>
void
test_conversions(const std::vector<RefinementChoice<dim>> &choices)
{
  for (const auto &choice : choices)
    {
      const auto ref_case = static_cast<RefinementCase<dim>>(choice);
      deallog << dim << "d " << choice.to_string()
              << " RefinementCase = " << int(ref_case)
              << " round trip = " << RefinementChoice<dim>(ref_case).to_string()
              << std::endl;
    }
}

template <int dim>
void
test_reference_cell(const ReferenceCell<dim> &reference_cell)
{
  deallog << reference_cell.to_string() << ":";
  for (const auto &choice : reference_cell.refinement_choices())
    deallog << " " << choice.to_string() << " ("
            << reference_cell.n_children(choice) << " children, index "
            << reference_cell.refinement_choice_index(choice) << ")";
  deallog << std::endl;
}

// Check CellAccessor::refinement_choice_set() and
// TriaAccessor::refinement_choice() for a single-cell Triangulation
template <int dim>
void
print_flag(const Triangulation<dim> &tria, const std::string &description)
{
  deallog << dim << "d " << tria.begin_active()->reference_cell().to_string()
          << " " << description << ": "
          << tria.begin_active()->refinement_choice_set().to_string()
          << std::endl;
}

void
test_cell_refinement_choice()
{
  {
    Triangulation<1> tria;
    GridGenerator::hyper_cube(tria);
    print_flag(tria, "unflagged");
    tria.begin_active()->set_refine_flag();
    print_flag(tria, "isotropic flag");
  }
  {
    Triangulation<2> tria;
    GridGenerator::hyper_cube(tria);
    tria.begin_active()->set_refine_flag(RefinementCase<2>::cut_x);
    print_flag(tria, "cut_x flag");
  }
  {
    Triangulation<3> tria;
    GridGenerator::hyper_cube(tria);
    tria.begin_active()->set_refine_flag(RefinementCase<3>::cut_xz);
    print_flag(tria, "cut_xz flag");
    tria.begin_active()->clear_refine_flag();
    print_flag(tria, "cleared flag");
    tria.begin_active()->set_refine_flag(RefinementCase<3>::cut_xz);
    tria.execute_coarsening_and_refinement();
    deallog << "3d Hex refined with cut_xz: "
            << tria.begin(0)->refinement_choice().to_string() << ", face 0: "
            << tria.begin(0)->face(0)->refinement_choice().to_string()
            << ", face 4: "
            << tria.begin(0)->face(4)->refinement_choice().to_string()
            << ", child 0: "
            << tria.begin(0)->child(0)->refinement_choice().to_string()
            << std::endl;
  }
  {
    Triangulation<2> tria;
    GridGenerator::reference_cell(tria, ReferenceCells::Triangle);
    tria.begin_active()->set_refine_flag();
    print_flag(tria, "isotropic flag");
  }
  {
    Triangulation<3> tria;
    GridGenerator::reference_cell(tria, ReferenceCells::Tetrahedron);
    const auto cell = tria.begin_active();
    print_flag(tria, "unflagged");
    cell->set_refine_flag();
    cell->set_refine_choice(
      static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_57));
    print_flag(tria, "flag, then choice cut_tet_57");
    cell->clear_refine_flag();
    cell->clear_refine_choice();
    cell->set_refine_choice(
      static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_49));
    cell->set_refine_flag();
    print_flag(tria, "choice cut_tet_49, then flag");
    cell->clear_refine_flag();
    cell->clear_refine_choice();
    cell->set_refine_flag();
    print_flag(tria, "flag only (shortest new line)");
    cell->clear_refine_flag();
    cell->clear_refine_choice();
    cell->set_refine_flag();
    cell->set_refine_choice(
      static_cast<std::uint8_t>(IsotropicRefinementChoice::cut_tet_57));
    tria.execute_coarsening_and_refinement();
    deallog << "3d Tet refined with cut_tet_57: "
            << tria.begin(0)->refinement_choice().to_string() << ", face 0: "
            << tria.begin(0)->face(0)->refinement_choice().to_string()
            << std::endl;
  }
}

int
main()
{
  initlog();

  for (const auto &choice : get_hypercube_choices<0>())
    test(choice, get_hypercube_choices<0>());
  for (const auto &choice : get_hypercube_choices<1>())
    test(choice, get_hypercube_choices<1>());
  for (const auto &choice : get_hypercube_choices<2>())
    test(choice, get_hypercube_choices<2>());
  for (const auto &choice : get_hypercube_choices<3>())
    test(choice, get_hypercube_choices<3>());
  for (const auto &choice : get_tetrahedron_choices())
    test(choice, get_tetrahedron_choices());

  // Check cut_axis():
  for (unsigned int i = 0; i < 1; ++i)
    deallog << "1d cut_axis(" << i
            << ") = " << RefinementChoice<1>::cut_axis(i).to_string()
            << std::endl;
  for (unsigned int i = 0; i < 2; ++i)
    deallog << "2d cut_axis(" << i
            << ") = " << RefinementChoice<2>::cut_axis(i).to_string()
            << std::endl;
  for (unsigned int i = 0; i < 3; ++i)
    deallog << "3d cut_axis(" << i
            << ") = " << RefinementChoice<3>::cut_axis(i).to_string()
            << std::endl;
  static_assert(RefinementChoice<3>::cut_axis(2) == RefinementChoices::CutZ<3>);

  // Check ReferenceCell::is_compatible_refinement_choice():
  for (const auto &reference_cell : {ReferenceCells::Line})
    for (const auto &choice : get_hypercube_choices<1>())
      deallog << reference_cell.to_string() << " " << choice.to_string()
              << " compatible = "
              << reference_cell.is_compatible_refinement_choice(choice)
              << std::endl;
  for (const auto &reference_cell :
       {ReferenceCells::Triangle, ReferenceCells::Quadrilateral})
    for (const auto &choice : get_hypercube_choices<2>())
      deallog << reference_cell.to_string() << " " << choice.to_string()
              << " compatible = "
              << reference_cell.is_compatible_refinement_choice(choice)
              << std::endl;
  for (const auto &reference_cell : {ReferenceCells::Tetrahedron,
                                     ReferenceCells::Pyramid,
                                     ReferenceCells::Wedge,
                                     ReferenceCells::Hexahedron})
    {
      auto choices = get_hypercube_choices<3>();
      for (const auto &choice : get_tetrahedron_choices())
        if (choice != RefinementChoices::None<3>)
          choices.push_back(choice);
      for (const auto &choice : choices)
        deallog << reference_cell.to_string() << " " << choice.to_string()
                << " compatible = "
                << reference_cell.is_compatible_refinement_choice(choice)
                << std::endl;
    }

  // Check conversions to and from
  // RefinementCase:
  test_conversions(get_hypercube_choices<0>());
  test_conversions(get_hypercube_choices<1>());
  test_conversions(get_hypercube_choices<2>());
  test_conversions(get_hypercube_choices<3>());
  test_conversions(get_tetrahedron_choices());

  // Check ReferenceCell::refinement_choices() and ReferenceCell::n_children():
  test_reference_cell(ReferenceCells::Vertex);
  test_reference_cell(ReferenceCells::Line);
  test_reference_cell(ReferenceCells::Triangle);
  test_reference_cell(ReferenceCells::Quadrilateral);
  test_reference_cell(ReferenceCells::Tetrahedron);
  test_reference_cell(ReferenceCells::Pyramid);
  test_reference_cell(ReferenceCells::Wedge);
  test_reference_cell(ReferenceCells::Hexahedron);

  test_cell_refinement_choice();

  // Check that the integer values match those of RefinementCase.
  static_assert(int(RefinementChoices::Isotropic<1>) == 1);
  static_assert(int(RefinementChoices::Isotropic<2>) == 3);
  static_assert(int(RefinementChoices::Isotropic<3>) == 7);
  static_assert(int(RefinementChoices::CutXZ<3>) == 5);
}
