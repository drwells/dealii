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

#include <deal.II/grid/refinement_choice.h>

DEAL_II_NAMESPACE_OPEN

template <int dim>
RefinementChoice<dim>::RefinementChoice(const IsotropicRefinementChoice choice)
  : kind(0)
{
  switch (choice)
    {
      case IsotropicRefinementChoice::isotropic_refinement:
        *this = RefinementChoices::None<dim>;
        return;
      case IsotropicRefinementChoice::cut_tet_68:
      case IsotropicRefinementChoice::cut_tet_57:
      case IsotropicRefinementChoice::cut_tet_49:
        if constexpr (dim == 3)
          {
            if (choice == IsotropicRefinementChoice::cut_tet_68)
              *this = RefinementChoices::Cut68<3>;
            else if (choice == IsotropicRefinementChoice::cut_tet_57)
              *this = RefinementChoices::Cut57<3>;
            else
              *this = RefinementChoices::Cut49<3>;
          }
        else
          DEAL_II_NOT_IMPLEMENTED();
        return;
    }

  DEAL_II_NOT_IMPLEMENTED();
}



template <int dim>
RefinementChoice<dim>::
operator IsotropicRefinementChoice() const
{
  if (*this == RefinementChoices::None<dim>)
    return IsotropicRefinementChoice::isotropic_refinement;

  if constexpr (dim == 3)
    {
      if (*this == RefinementChoices::Cut68<3>)
        return IsotropicRefinementChoice::cut_tet_68;
      if (*this == RefinementChoices::Cut57<3>)
        return IsotropicRefinementChoice::cut_tet_57;
      if (*this == RefinementChoices::Cut49<3>)
        return IsotropicRefinementChoice::cut_tet_49;
    }

  Assert(false,
         ExcMessage("Only None, Cut49, Cut57, and Cut68 can be converted to "
                    "IsotropicRefinementChoice."));
  return IsotropicRefinementChoice::isotropic_refinement;
}



template <int dim>
RefinementChoice<dim>::RefinementChoice(
  const RefinementCase<dim> &refinement_case)
  : kind(0)
{
  // Compare stored values rather than enumeration values: RefinementCase
  // stores its value in a bit field, so in 0d isotropic_refinement (0xFF) is
  // truncated when it is stored.
  const auto stored_value = [](const RefinementCase<dim> &rc) {
    return static_cast<std::uint8_t>(rc);
  };
  if (stored_value(refinement_case) ==
      stored_value(RefinementCase<dim>::no_refinement))
    *this = RefinementChoices::None<dim>;
  else if (stored_value(refinement_case) ==
           stored_value(RefinementCase<dim>::isotropic_refinement))
    *this = RefinementChoices::Isotropic<dim>;
  else
    {
      // The integer values of the anisotropic RefinementCases and
      // RefinementChoices are equal (the constructor checks validity).
      Assert(dim > 1, ExcInternalError());
      *this = RefinementChoice<dim>(static_cast<std::uint8_t>(refinement_case));
    }
}



template <int dim>
RefinementChoice<dim>::
operator RefinementCase<dim>() const
{
  if (*this == RefinementChoices::None<dim>)
    return RefinementCase<dim>::no_refinement;
  else if (*this == RefinementChoices::Isotropic<dim> || this->is_tetrahedral())
    return RefinementCase<dim>::isotropic_refinement;
  else
    return RefinementCase<dim>(this->kind);
}



template <int dim>
std::string
RefinementChoice<dim>::to_string() const
{
  if (*this == RefinementChoices::None<dim>)
    return "None";
  if (*this == RefinementChoices::Isotropic<dim>)
    return "Isotropic";

  if constexpr (dim == 2)
    {
      if (*this == RefinementChoices::CutX<2>)
        return "CutX";
      if (*this == RefinementChoices::CutY<2>)
        return "CutY";
    }
  else if constexpr (dim == 3)
    {
      if (*this == RefinementChoices::CutX<3>)
        return "CutX";
      if (*this == RefinementChoices::CutY<3>)
        return "CutY";
      if (*this == RefinementChoices::CutXY<3>)
        return "CutXY";
      if (*this == RefinementChoices::CutZ<3>)
        return "CutZ";
      if (*this == RefinementChoices::CutXZ<3>)
        return "CutXZ";
      if (*this == RefinementChoices::CutYZ<3>)
        return "CutYZ";
      if (*this == RefinementChoices::Cut49<3>)
        return "Cut49";
      if (*this == RefinementChoices::Cut57<3>)
        return "Cut57";
      if (*this == RefinementChoices::Cut68<3>)
        return "Cut68";
    }

  DEAL_II_ASSERT_UNREACHABLE();
  return "";
}



template class RefinementChoice<0>;
template class RefinementChoice<1>;
template class RefinementChoice<2>;
template class RefinementChoice<3>;

DEAL_II_NAMESPACE_CLOSE
