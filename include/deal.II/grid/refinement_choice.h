// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2020 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#ifndef dealii_refinement_choice_h
#define dealii_refinement_choice_h


#include <deal.II/base/config.h>

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>

#include <cstdint>
#include <string>


DEAL_II_NAMESPACE_OPEN

// Forward declarations
template <int dim>
class RefinementChoice;

// This should work. Some TODOs:
//
// 1. Deprecate all GeometryInfo stuff which uses RefinementCase so that we
//    don't need to update it to RefinementChoice. This requires moving a lot of
//    stuff over to ReferenceCell

namespace internal
{
  /**
   * A helper function to create a RefinementChoice object from an integer.
   * RefinementChoice objects are "singletons" (actually, "multitons" -- there
   * are multiple, but they are only a handful and these are all that can be
   * used). What is then necessary is to have a way to create these with their
   * internal id to distinguish the few possible ones in existence. We could do
   * this via a public constructor of RefinementChoice, but that would allow
   * users to create ones outside the range we envision, and we don't want to do
   * that. Rather, the constructor that takes an integer is made `private` but
   * we have this one function in an internal namespace that is a friend of the
   * class and can be used to create the objects.
   */
  template <int dim>
  constexpr RefinementChoice<dim>
  make_refinement_choice_from_int(const std::uint8_t kind);
} // namespace internal

#if 0
  // TODO to avoid cyclic dependencies, ReferenceCell should depend on this
  // class and not the other way around. Hence this method needs to be moved to
  // ReferenceCell.

  /**
   * Return a boolean indicating whether or not the ReferenceCell may be refined
   * with @p choice.
   */
  template <int dim>
  constexpr bool
  ReferenceCell<dim>::is_supported(const RefinementChoice<dim> &choice) const;
#endif

template <int dim>
class RefinementChoice
{
public:
  /**
   * Default constructor. Creates an object equal to
   * RefinementChoice<dim>::None.
   */
  constexpr RefinementChoice();

  /**
   * Convert the RefinementChoice to a bool. This is equal to testing for
   * inequality with RefinementChoices::None.
   */
  constexpr operator bool() const;

  /**
   * Conversion operator to the underlying integral type.
   */
  constexpr operator std::uint8_t() const;

  /**
   * Conversion operator to an int.
   *
   * @note Since this class has two other integer conversions (bool and
   * std::uint8_t) it needs a dedicated int conversion operator so that
   * constexpr RefinementChoices may be used as a switch statement case values.
   */
  constexpr operator int() const;

  /**
   * Comparison operator.
   */
  constexpr bool
  operator==(const RefinementChoice<dim> &other_choice) const;

  /**
   * Comparison operator.
   */
  constexpr bool
  operator!=(const RefinementChoice<dim> &other_choice) const;

  constexpr std::string
  to_string() const;

  /**
   * Create a new RefinementChoice which is the union of the current object
   * and @p other_choice.
   */
  constexpr RefinementChoice<dim>
  operator|(const RefinementChoice<dim> &other_choice) const;

  /**
   * Create a new RefinementChoice which is the intersection of the current
   * object and @p other_choice.
   */
  constexpr RefinementChoice<dim>
  operator&(const RefinementChoice<dim> &other_choice) const;

  /**
   * Invert the RefinementChoice.
   */
  constexpr RefinementChoice<dim>
  operator~() const;

  /**
   * Return whether or not the refinement is anisotropic.
   */
  constexpr bool
  is_anisotropic() const;

  /**
   * Return the internal line for tetrahedral refinement. This function may only
   * be called on Cut49, Cut57, and Cut68.
   */
  constexpr unsigned int
  new_tetrahedron_line() const;

private:
  /**
   * The variable that stores what this object actually corresponds to.
   */
  std::uint8_t kind;

  /**
   * Constructor.
   */
  constexpr RefinementChoice(const std::uint8_t kind);

  /**
   * A kind of constructor -- not quite private because it can be called by
   * anyone, but at least hidden in an internal namespace.
   */
  template <int dim_>
  friend constexpr RefinementChoice<dim_>
  internal::make_refinement_choice_from_int(const std::uint8_t);
};


/*----------------------------- inline functions -----------------------------*/


template <int dim>
inline constexpr RefinementChoice<dim>::RefinementChoice()
  : kind(0)
{}



template <int dim>
inline constexpr RefinementChoice<dim>::RefinementChoice(
  const std::uint8_t kind)
  : kind(kind)
{
  static_assert(dim <= 3, "Only implemented for 1d, 2d, and 3d");
  // To prevent circular dependencies, this constructor also encodes the integer
  // value of several RefinementChoices so that we can both use this constructor
  // to create constexpr values and check their validity.
#if DEAL_II_WITH_CXX20
  // Prevent users from creating illegal combinations of dim and kind. This
  // duplicates the values of kind encoded in RefinementChoices.
  //
  // Since Assert() is a runtime check we have to duplicate things to get good
  // compiler error messages for, e.g., CutX<1> (which looks like the old
  // RefinementPossibilities::cut_x, which was equal to
  // RefinementPossibilities::isotropic_refinement) which we do not allow
  if consteval
    {
      if (dim == 1)
        static_assert(kind == 0 || kind == 7,
                      "The only valid RefinementChoices in 1d are None and "
                      "Isotropic.");
      if (dim == 2)
        static_assert(kind < 3 || kind == 7,
                      ExcMessage(
                        "The only valid RefinementChoices in 2d are None, "
                        "CutX, CutY, and Isotropic."));
      if (dim == 3)
        static_assert(kind < 8 || kind == 15 || kind == 23 || kind == 31,
                      "This RefinementChoice isn't implemented in 3d.");
    }
  else
    {
      if (dim == 1)
        Assert(kind == 0 || kind == 7,
               ExcMessage("The only valid RefinementChoices in 1d are None and "
                          "Isotropic."));
      else if (dim == 2)
        Assert(kind < 3 || kind == 7,
               ExcMessage("The only valid RefinementChoices in 2d are None, "
                          "CutX, CutY, and Isotropic."));
      else if (dim == 3)
        Assert(kind < 8 || kind == 15 || kind == 23 || kind == 31,
               ExcNotImplemented());
      else
        DEAL_II_NOT_IMPLEMENTED();
    }
#endif
}



namespace internal
{
  template <int dim>
  inline constexpr RefinementChoice<dim>
  make_refinement_choice_from_int(const std::uint8_t kind)
  {
    return RefinementChoice<dim>(kind);
  }
} // namespace internal



/**
 * A namespace in which we define objects that correspond to specific refinement
 * choices. The objects defined here are a complete enumeration of all possible
 * refinement choices that can be used in deal.II.
 */
namespace RefinementChoices
{
  /**
   * Do not refine the cell (the default).
   */
  template <int dim>
  constexpr RefinementChoice<dim> None =
    internal::make_refinement_choice_from_int<dim>(0);

  /**
   * Refine a quadrilateral or hexahedron by cutting in the $x$ direction.
   *
   * @note This may only be used in 2d or 3d. To refine a 1d cell use
   * RefinementChoices::Isotropic.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutX =
    internal::make_refinement_choice_from_int<dim>(1);

  /**
   * Refine a quadrilateral or hexahedron by cutting in the $y$ direction.
   *
   * @note This may only be used in 2d or 3d.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutY =
    internal::make_refinement_choice_from_int<dim>(2);

  /**
   * Refine a hexahedron by cutting in the $x$ and $y$ directions.
   *
   * @note This may only be used in 3d. To refine a 2d cell use
   * RefinementChoices::Isotropic.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutXY =
    internal::make_refinement_choice_from_int<dim>(3);

  /**
   * Refine a hexahedron by cutting in the $z$ direction.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutZ =
    internal::make_refinement_choice_from_int<dim>(4);

  /**
   * Refine a hexahedron by cutting in the $x$ and $z$ directions.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutXZ =
    internal::make_refinement_choice_from_int<dim>(5);

  /**
   * Refine a hexahedron by cutting in the $y$ and $z$ directions.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutYZ =
    internal::make_refinement_choice_from_int<dim>(6);

  /**
   * Refine a cell isotropically.
   *
   * @note There are three distinct ways to isotropically refine a tetrahedron.
   * If this RefinementChoice is selected for a tetrahedron, then Triangulation
   * will pick the RefinementChoice (one of Cut49, Cut57, or Cut68) which
   * creates cells with the best aspect ratios.
   */
  template <int dim>
  constexpr RefinementChoice<dim> Isotropic =
    internal::make_refinement_choice_from_int<dim>(7);

  /**
   * Refine a Tetrahedron isotropically, with the internal line placed between
   * the fourth and ninth vertices.
   */
  template <int dim>
  constexpr RefinementChoice<dim> Cut49 =
    internal::make_refinement_choice_from_int<dim>(15);

  /**
   * Refine a Tetrahedron isotropically, with the internal line placed between
   * the fifth and seventh vertices.
   */
  template <int dim>
  constexpr RefinementChoice<dim> Cut57 =
    internal::make_refinement_choice_from_int<dim>(23);

  /**
   * Refine a Tetrahedron isotropically, with the internal line placed between
   * the sixth and eighth vertices.
   */
  template <int dim>
  constexpr RefinementChoice<dim> Cut68 =
    internal::make_refinement_choice_from_int<dim>(31);
} // namespace RefinementChoices



template <int dim>
inline constexpr RefinementChoice<dim>::operator bool() const
{
  static_assert(RefinementChoices::None<dim>.kind == 0);
  return this->kind != 0;
}



template <int dim>
inline constexpr RefinementChoice<dim>::operator std::uint8_t() const
{
  return this->kind;
}



template <int dim>
inline constexpr RefinementChoice<dim>::operator int() const
{
  static_assert(RefinementChoices::None<dim>.kind == 0);
  return kind;
}



template <int dim>
inline constexpr bool
RefinementChoice<dim>::operator==(const RefinementChoice<dim> &other) const
{
  return this->kind == other.kind;
}



template <int dim>
inline constexpr bool
RefinementChoice<dim>::operator!=(const RefinementChoice<dim> &other) const
{
  return !(*this == other);
}



template <int dim>
inline constexpr std::string
RefinementChoice<dim>::to_string() const
{
  if (*this == RefinementChoices::None<dim>)
    return "None";
  if (*this == RefinementChoices::Isotropic<dim>)
    return "Isotropic";

  if constexpr (dim == 1)
    {
      DEAL_II_NOT_IMPLEMENTED();
    }
  else if constexpr (dim == 2)
    switch (this->kind)
      {
        case RefinementChoices::CutX<2>:
          return "CutX";
        case RefinementChoices::CutY<2>:
          return "CutY";
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
  else if constexpr (dim == 3)
    switch (this->kind)
      {
        case RefinementChoices::CutX<3>:
          return "CutX";
        case RefinementChoices::CutY<3>:
          return "CutY";
        case RefinementChoices::CutXY<3>:
          return "CutXY";
        case RefinementChoices::CutZ<3>:
          return "CutZ";
        case RefinementChoices::CutXZ<3>:
          return "CutXZ";
        case RefinementChoices::CutYZ<3>:
          return "CutYZ";
        case RefinementChoices::Cut49<3>:
          return "Cut49";
        case RefinementChoices::Cut57<3>:
          return "Cut57";
        case RefinementChoices::Cut68<3>:
          return "Cut68";
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
  else
    {
      DEAL_II_NOT_IMPLEMENTED();
      return "";
    }

  DEAL_II_ASSERT_UNREACHABLE();
  return "";
}



template <int dim>
inline constexpr RefinementChoice<dim>
RefinementChoice<dim>::operator|(
  const RefinementChoice<dim> &other_choice) const
{
  // Canonicalize by ensuring this->kind < other_choice.kind
  if (*this == other_choice)
    return *this;
  if (this->kind > other_choice.kind)
    return other_choice.operator|(*this);
  Assert(this->kind < other_choice.kind, ExcInternalError());

  if (*this == RefinementChoices::None<dim>)
    return other_choice;

  if constexpr (dim == 1)
    {
      // 1d must be None or Isotropic, which is already covered by the
      // dimension-independent code. We already checked if *this is None, so (by
      // sorting) both must be Isotropic (which we also already checked).
      DEAL_II_ASSERT_UNREACHABLE();
      return RefinementChoices::None<1>;
    }
  else if constexpr (dim == 2)
    {
      // All other cases are handled by the dimension-independent code. *this
      // cannot be None at this point. By sorting, either *this is CutX and
      // other_choice is CutY or Isotropic, or *this is CutY in which case
      // other_choice must also be Isotropic. Either way the result must be
      // Isotropic.
      Assert((*this == RefinementChoices::CutX<2> &&
              other_choice == RefinementChoices::CutY<2>) ||
               other_choice == RefinementChoices::Isotropic<2>,
             ExcInternalError());
      return RefinementChoices::Isotropic<2>;
    }
  else if constexpr (dim == 3)
    {
      // Handle hexahedra first:
      if (other_choice.kind <= RefinementChoices::Isotropic<3>.kind)
        {
          Assert(this->kind <= RefinementChoices::Isotropic<3>.kind,
                 ExcInternalError());
          return RefinementChoice<3>(this->kind | other_choice.kind);
        }
      // Then tetrahedra:
      else
        {
          Assert(this->kind >= RefinementChoices::Isotropic<3>.kind,
                 ExcMessage("Specific tetrahedral refinemens (e.g., Cut49) "
                            "cannot be combined with anisotropic refinement "
                            "(e.g., CutX)."));
          // Checking other_choice is redundant but it avoids mentioning the
          // sorting step in the assertion message
          Assert(*this == RefinementChoices::Isotropic<3> ||
                   other_choice == RefinementChoices::Isotropic<3>,
                 ExcMessage("The union of specific tetrahedral refinements, "
                            "e.g., Cut49 with Cut57, is not defined since "
                            "these are mutually exclusive operations."));
          return other_choice;
        }
    }
  else
    DEAL_II_NOT_IMPLEMENTED();

  DEAL_II_ASSERT_UNREACHABLE();
  return RefinementChoices::None<dim>;
}



template <int dim>
inline constexpr RefinementChoice<dim>
RefinementChoice<dim>::operator&(
  const RefinementChoice<dim> &other_choice) const
{
  // Like operator|(), canonicalize by ensuring this->kind < other_choice.kind
  if (*this == other_choice)
    return *this;
  if (this->kind > other_choice.kind)
    return other_choice.operator&(*this);
  Assert(this->kind < other_choice.kind, ExcInternalError());

  if (*this == RefinementChoices::None<dim>)
    return *this;

  if constexpr (dim == 1)
    {
      // 1d must be None or Isotropic, which is already covered by the
      // dimension-independent code. We already checked if *this is None, so (by
      // sorting) both must be None (which we also already checked).
      DEAL_II_ASSERT_UNREACHABLE();
      return RefinementChoices::None<1>;
    }
  else if constexpr (dim == 2)
    {
      // All other cases are handled by the dimension-independent code. *this
      // cannot be None at this point. By sorting, either *this is CutX and
      // other_choice is CutY or Isotropic, or *this is CutY in which case
      // other_choice must also be Isotropic. Either way the result must be
      // Isotropic.
      Assert(*this == RefinementChoices::CutX<2> ||
               *this == RefinementChoices::CutY<2>,
             ExcInternalError());
      if (*this == RefinementChoices::CutX<2>)
        {
          Assert(other_choice == RefinementChoices::CutY<2> ||
                   other_choice == RefinementChoices::Isotropic<2>,
                 ExcInternalError());
          return other_choice == RefinementChoices::CutY<2> ?
                   RefinementChoices::None<2> :
                   RefinementChoices::CutX<2>;
        }
      else
        {
          Assert(other_choice == RefinementChoices::Isotropic<2>,
                 ExcInternalError());
          return *this;
        }
    }
  else if constexpr (dim == 3)
    {
      // Handle hexahedra first:
      if (other_choice.kind <= RefinementChoices::Isotropic<3>.kind)
        {
          Assert(this->kind <= RefinementChoices::Isotropic<3>.kind,
                 ExcInternalError());
          return RefinementChoice<3>(this->kind & other_choice.kind);
        }
      // Then tetrahedra:
      else
        {
          // The only reasonable option here is to 'downgrade' to an unspecified
          // isotropic refinement:
          return RefinementChoices::Isotropic<3>;
        }
    }
  else
    DEAL_II_NOT_IMPLEMENTED();

  DEAL_II_ASSERT_UNREACHABLE();
  return RefinementChoices::None<dim>;
}



template <int dim>
inline constexpr RefinementChoice<dim>
RefinementChoice<dim>::operator~() const
{
  // Handle common cases first:
  if (*this == RefinementChoices::None<dim>)
    return RefinementChoices::Isotropic<dim>;
  if (*this == RefinementChoices::Isotropic<dim>)
    return RefinementChoices::None<dim>;

  if constexpr (dim == 1)
    {
      DEAL_II_NOT_IMPLEMENTED();
    }
  else if constexpr (dim == 2)
    {
      switch (this->kind)
        {
          case RefinementChoices::CutX<2>:
            return RefinementChoices::CutY<2>;
          case RefinementChoices::CutY<2>:
            return RefinementChoices::CutX<2>;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
    }
  else if constexpr (dim == 3)
    {
      if (this->kind < RefinementChoices::Isotropic<3>.kind)
        {
          return internal::make_refinement_choice_from_int<3>(
            ~(this->kind) & RefinementChoices::Isotropic<3>.kind);
        }
      else
        {
          Assert(*this == RefinementChoices::Cut49<3> ||
                   *this == RefinementChoices::Cut57<3> ||
                   *this == RefinementChoices::Cut68<3>,
                 ExcInternalError());
          return RefinementChoices::Isotropic<3>;
        }
    }

  DEAL_II_ASSERT_UNREACHABLE();
  return RefinementChoices::None<dim>;
}



template <int dim>
inline constexpr bool
RefinementChoice<dim>::is_anisotropic() const
{
  if constexpr (dim == 1)
    return false;
  else if constexpr (dim == 2)
    return !(*this == RefinementChoices::None<2> ||
           *this == RefinementChoices::Isotropic<2>);
  else if constexpr (dim == 3)
    return !(*this == RefinementChoices::None<3> ||
             this->kind >= RefinementChoices::Isotropic<3>.kind);
  else
    {
      DEAL_II_ASSERT_UNREACHABLE();
      return false;
    }
}



template <int dim>
inline constexpr unsigned int
RefinementChoice<dim>::new_tetrahedron_line() const
{
  AssertDimension(dim, 3);
  switch (*this)
    {
      case RefinementChoices::Cut49<3>:
        return 2;
      case RefinementChoices::Cut57<3>:
        return 1;
      case RefinementChoices::Cut68<3>:
        return 0;
      default:
        DEAL_II_NOT_IMPLEMENTED();
    }

  DEAL_II_ASSERT_UNREACHABLE();
  return numbers::invalid_unsigned_int;
}


DEAL_II_NAMESPACE_CLOSE

#endif
