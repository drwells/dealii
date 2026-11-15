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

#ifndef dealii_refinement_choice_h
#define dealii_refinement_choice_h


#include <deal.II/base/config.h>

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <cstdint>
#include <string>


DEAL_II_NAMESPACE_OPEN

// Forward declarations
template <int dim>
class RefinementChoice;

template <int dim>
class RefinementCase;



/**
 * Enum of different choices for istropic refinement.
 * There are 3 different ways to refine a tetrahedral, here we save the
 * different possibilities. It is different to RefinementPossibilities are these
 * are options in the case that an isotropic refinement is conducted.
 *
 * @deprecated Use RefinementChoice (e.g., RefinementChoices::Cut68) instead.
 */
enum class DEAL_II_DEPRECATED_EARLY_WITH_COMMENT(
  "Use RefinementChoice instead.") IsotropicRefinementChoice : std::uint8_t
{
  /**
   * Decide where to perform a cut based on the shortest edge length.
   */
  isotropic_refinement = 0,
  /**
   * Perform a cut in the along the edge (6,8).
   */
  cut_tet_68 = 1,
  /**
   * Perform a cut in the along the edge (5,7).
   */
  cut_tet_57 = 2,
  /**
   * Perform a cut in the along the edge (4,9).
   */
  cut_tet_49 = 3,
};



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



/**
 * A class describing the ways in which a cell (or, more generally, any
 * geometric object of dimension @p dim) may be refined.
 *
 * Like ReferenceCell, objects of this type should not be created in user code:
 * rather, there is a small and fixed number of these objects defined in the
 * RefinementChoices namespace (e.g., RefinementChoices::Isotropic) which
 * enumerate every possibility. This class unifies the description of
 * refinement for all reference cells: in particular, it replaces both
 * RefinementCase (which only describes refinement of hypercubes) and
 * IsotropicRefinementChoice (which only describes the three ways in which a
 * tetrahedron may be refined).
 *
 * Not every RefinementChoice can be used with every reference cell. For
 * example, RefinementChoices::CutX only makes sense for quadrilaterals and
 * hexahedra. ReferenceCell provides functions which check if a given
 * RefinementChoice can be used with that ReferenceCell.
 *
 * @note For hypercubes the integer values of these objects are equal to the
 * corresponding values of RefinementCase.
 */
template <int dim>
class RefinementChoice
{
public:
  /**
   * Default constructor. Creates an object equal to
   * RefinementChoices::None.
   */
  constexpr RefinementChoice();

  /**
   * Convert an IsotropicRefinementChoice into a RefinementChoice. Since
   * IsotropicRefinementChoice::isotropic_refinement does not correspond to a
   * specific refinement of a tetrahedron, it is converted to
   * RefinementChoices::None (i.e., no choice has been made yet).
   *
   * @note The three specific values of IsotropicRefinementChoice may only be
   * converted to RefinementChoice<3>.
   */
  explicit RefinementChoice(const IsotropicRefinementChoice choice);

  /**
   * Convert a RefinementChoice to an IsotropicRefinementChoice. This is the
   * inverse of the constructor above: RefinementChoices::None is converted to
   * IsotropicRefinementChoice::isotropic_refinement. All values other than
   * None, Cut49, Cut57, and Cut68 cannot be converted.
   */
  explicit
  operator IsotropicRefinementChoice() const;

  /**
   * Convert a RefinementCase into a RefinementChoice. Since RefinementCase
   * only describes refinement of hypercubes, the result is never one of the
   * tetrahedral RefinementChoices.
   */
  explicit RefinementChoice(const RefinementCase<dim> &refinement_case);

  /**
   * Convert a RefinementChoice into a RefinementCase. Since RefinementCase
   * cannot describe the three ways in which a tetrahedron may be refined,
   * RefinementChoices::Cut49, RefinementChoices::Cut57, and
   * RefinementChoices::Cut68 are all converted to
   * RefinementCase<dim>::isotropic_refinement.
   */
  explicit
  operator RefinementCase<dim>() const;

  /**
   * Convert the RefinementChoice to a bool. This is equal to testing for
   * inequality with RefinementChoices::None.
   */
  constexpr explicit
  operator bool() const;

  /**
   * Conversion operator to the underlying integral type.
   *
   * @note This conversion (and the one to int) is explicit: otherwise,
   * expressions like `static_cast<RefinementCase<dim>>(choice)` would use
   * RefinementCase's constructor which takes an integer instead of the
   * conversion operator to RefinementCase, which gives the wrong result for,
   * e.g., RefinementChoices::Cut49.
   */
  constexpr explicit
  operator std::uint8_t() const;

  /**
   * Conversion operator to an int.
   */
  constexpr explicit
  operator int() const;

  /**
   * An enumeration type whose only purpose is to make RefinementChoice
   * objects usable in switch statements, e.g.,
   * @code
   * switch (choice)
   *   {
   *     case RefinementChoices::CutX<3>:
   *       // ...
   *       break;
   *     case RefinementChoices::Cut49<3>:
   *       // ...
   *       break;
   *     default:
   *       break;
   *   }
   * @endcode
   * Unlike an integer type, a scoped enumeration cannot be implicitly
   * converted to an integer, so this conversion does not allow other classes'
   * constructors which take an integer (e.g., the one of RefinementCase) to be
   * used instead of the conversion operators above.
   */
  enum class SwitchValue : std::uint8_t
  {
  };

  /**
   * Conversion operator to SwitchValue, which allows RefinementChoice objects
   * to be used in switch statements.
   */
  constexpr
  operator SwitchValue() const;

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

  /**
   * Return a string representation of the present object, e.g., "CutX".
   */
  std::string
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
   * Return the RefinementChoice corresponding to cutting a hypercube along the
   * axis @p i. For example, if <code>i == 0</code> and <code>dim == 3</code>
   * then the returned value is RefinementChoices::CutX<3>. Since 1d cells
   * can only be refined in one way, in 1d this function returns
   * RefinementChoices::Isotropic<1>.
   *
   * This is the equivalent of RefinementCase::cut_axis().
   */
  static constexpr RefinementChoice<dim>
  cut_axis(const unsigned int i);

  /**
   * Return the internal line for tetrahedral refinement. This function may only
   * be called on Cut49, Cut57, and Cut68.
   *
   * @note The returned value is the index of the new interior line in the
   * list of possible interior lines (6, 8), (5, 7), and (4, 9) used by
   * Triangulation and ReferenceCell: i.e., Cut68 returns 0, Cut57 returns 1,
   * and Cut49 returns 2. This is unrelated to the array index computed by
   * ReferenceCell::refinement_choice_index(), which follows the order of
   * ReferenceCell::refinement_choices().
   */
  constexpr unsigned int
  new_tetrahedron_line() const;

  /**
   * Return the amount of memory occupied by an object of this type.
   */
  static constexpr std::size_t
  memory_consumption();

  /**
   * Read or write the data of this object to or from a stream for the purpose
   * of serialization using the [BOOST serialization
   * library](https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/index.html).
   */
  template <class Archive>
  void
  serialize(Archive &archive, const unsigned int version);

private:
  /**
   * The value of RefinementChoice::kind corresponding to isotropic refinement.
   * For hypercubes in dimensions one through three this is the same value
   * used by RefinementCase<dim>::isotropic_refinement.
   */
  static constexpr std::uint8_t isotropic_kind = dim == 0 ? 1 : (1 << dim) - 1;

  /**
   * The variable that stores what this object actually corresponds to.
   */
  std::uint8_t kind;

  /**
   * Constructor.
   */
  constexpr explicit RefinementChoice(const std::uint8_t kind);

  /**
   * Return whether or not the present object is one of the RefinementChoices
   * specific to tetrahedra.
   */
  constexpr bool
  is_tetrahedral() const;

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
  static_assert(0 <= dim && dim <= 3, "Only implemented for 0d - 3d.");
  // Prevent users from creating illegal combinations of dim and kind. These
  // checks duplicate the values of kind encoded in RefinementChoices. Since
  // this constructor is constexpr, these checks will also cause compilation
  // errors for invalid constexpr objects.
  if constexpr (dim == 0)
    Assert(kind <= isotropic_kind,
           ExcMessage("The only valid RefinementChoices in 0d are None and "
                      "Isotropic."));
  else if constexpr (dim == 1)
    Assert(kind <= isotropic_kind,
           ExcMessage("The only valid RefinementChoices in 1d are None and "
                      "Isotropic."));
  else if constexpr (dim == 2)
    Assert(kind <= isotropic_kind,
           ExcMessage("The only valid RefinementChoices in 2d are None, "
                      "CutX, CutY, and Isotropic."));
  else if constexpr (dim == 3)
    Assert(kind <= isotropic_kind || (8 <= kind && kind <= 10),
           ExcMessage("This RefinementChoice isn't implemented in 3d."));
}



namespace internal
{
  template <int dim>
  inline constexpr RefinementChoice<dim>
  make_refinement_choice_from_int(const std::uint8_t kind)
  {
    return RefinementChoice<dim>(kind);
  }



  /**
   * Helper function for creating anisotropic RefinementChoice objects, which
   * only exist in 2d and 3d.
   */
  template <int dim>
  constexpr RefinementChoice<dim>
  make_anisotropic_refinement_choice_from_int(const std::uint8_t kind)
  {
    static_assert(dim == 2 || dim == 3,
                  "Anisotropic RefinementChoices only exist in 2d and 3d.");
    return make_refinement_choice_from_int<dim>(kind);
  }



  /**
   * Helper function for creating RefinementChoice objects which are only
   * valid in 3d.
   */
  template <int dim>
  constexpr RefinementChoice<dim>
  make_3d_refinement_choice_from_int(const std::uint8_t kind)
  {
    static_assert(dim == 3, "This RefinementChoice only exists in 3d.");
    return make_refinement_choice_from_int<dim>(kind);
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
    internal::make_anisotropic_refinement_choice_from_int<dim>(1);

  /**
   * Refine a quadrilateral or hexahedron by cutting in the $y$ direction.
   *
   * @note This may only be used in 2d or 3d.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutY =
    internal::make_anisotropic_refinement_choice_from_int<dim>(2);

  /**
   * Refine a hexahedron by cutting in the $x$ and $y$ directions.
   *
   * @note This may only be used in 3d. To refine a 2d cell use
   * RefinementChoices::Isotropic.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutXY =
    internal::make_3d_refinement_choice_from_int<dim>(3);

  /**
   * Refine a hexahedron by cutting in the $z$ direction.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutZ =
    internal::make_3d_refinement_choice_from_int<dim>(4);

  /**
   * Refine a hexahedron by cutting in the $x$ and $z$ directions.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutXZ =
    internal::make_3d_refinement_choice_from_int<dim>(5);

  /**
   * Refine a hexahedron by cutting in the $y$ and $z$ directions.
   */
  template <int dim>
  constexpr RefinementChoice<dim> CutYZ =
    internal::make_3d_refinement_choice_from_int<dim>(6);

  /**
   * Refine a cell isotropically.
   *
   * @note There are three distinct ways to isotropically refine a tetrahedron
   * and, as such, this RefinementChoice cannot be used with tetrahedra. Use one
   * of RefinementChoices::Cut49, RefinementChoices::Cut57, or
   * RefinementChoices::Cut68 instead.
   *
   * @note In 1d, 2d, and 3d, this value is equal to
   * RefinementCase<dim>::isotropic_refinement.
   */
  template <int dim>
  constexpr RefinementChoice<dim> Isotropic =
    internal::make_refinement_choice_from_int<dim>(dim == 0 ? 1 :
                                                              (1 << dim) - 1);

  /**
   * Refine a Tetrahedron isotropically, with the internal line placed between
   * the fourth and ninth vertices.
   */
  template <int dim>
  constexpr RefinementChoice<dim> Cut49 =
    internal::make_3d_refinement_choice_from_int<dim>(8);

  /**
   * Refine a Tetrahedron isotropically, with the internal line placed between
   * the fifth and seventh vertices.
   */
  template <int dim>
  constexpr RefinementChoice<dim> Cut57 =
    internal::make_3d_refinement_choice_from_int<dim>(9);

  /**
   * Refine a Tetrahedron isotropically, with the internal line placed between
   * the sixth and eighth vertices.
   */
  template <int dim>
  constexpr RefinementChoice<dim> Cut68 =
    internal::make_3d_refinement_choice_from_int<dim>(10);
} // namespace RefinementChoices



template <int dim>
inline constexpr RefinementChoice<dim>::
operator bool() const
{
  return this->kind != 0;
}



template <int dim>
inline constexpr RefinementChoice<dim>::
operator std::uint8_t() const
{
  return this->kind;
}



template <int dim>
inline constexpr RefinementChoice<dim>::
operator int() const
{
  return kind;
}



template <int dim>
inline constexpr RefinementChoice<dim>::
operator SwitchValue() const
{
  return static_cast<SwitchValue>(kind);
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
inline constexpr bool
RefinementChoice<dim>::is_tetrahedral() const
{
  return this->kind > isotropic_kind;
}



template <int dim>
inline constexpr RefinementChoice<dim>
RefinementChoice<dim>::operator|(
  const RefinementChoice<dim> &other_choice) const
{
  // Canonicalize by ensuring this->kind <= other_choice.kind
  if (*this == other_choice)
    return *this;
  if (this->kind > other_choice.kind)
    return other_choice.operator|(*this);

  if (*this == RefinementChoices::None<dim>)
    return other_choice;

  // Hypercube refinements are stored as bit fields:
  if (!other_choice.is_tetrahedral())
    return RefinementChoice<dim>(this->kind | other_choice.kind);

  // Tetrahedral refinements can only be combined with None (handled above) or
  // themselves (also handled above).
  Assert(this->is_tetrahedral(),
         ExcMessage("Tetrahedral refinements (e.g., Cut49) cannot be combined "
                    "with refinements of other cells (e.g., CutX or "
                    "Isotropic)."));
  Assert(false,
         ExcMessage("The union of specific tetrahedral refinements, e.g., "
                    "Cut49 with Cut57, is not defined since these are mutually "
                    "exclusive operations."));
  return RefinementChoices::None<dim>;
}



template <int dim>
inline constexpr RefinementChoice<dim>
RefinementChoice<dim>::operator&(
  const RefinementChoice<dim> &other_choice) const
{
  // Like operator|(), canonicalize by ensuring this->kind <= other_choice.kind
  if (*this == other_choice)
    return *this;
  if (this->kind > other_choice.kind)
    return other_choice.operator&(*this);

  if (*this == RefinementChoices::None<dim>)
    return *this;

  // Hypercube refinements are stored as bit fields:
  if (!other_choice.is_tetrahedral())
    return RefinementChoice<dim>(this->kind & other_choice.kind);

  // Tetrahedral refinements can only be combined with None (handled above) or
  // other tetrahedral refinements. Since these are mutually exclusive their
  // intersection is empty.
  Assert(this->is_tetrahedral(),
         ExcMessage("Tetrahedral refinements (e.g., Cut49) cannot be combined "
                    "with refinements of other cells (e.g., CutX or "
                    "Isotropic)."));
  return RefinementChoices::None<dim>;
}



template <int dim>
inline constexpr RefinementChoice<dim>
RefinementChoice<dim>::operator~() const
{
  if (this->is_tetrahedral())
    return RefinementChoices::None<dim>;
  else
    return RefinementChoice<dim>(~(this->kind) & isotropic_kind);
}



template <int dim>
inline constexpr bool
RefinementChoice<dim>::is_anisotropic() const
{
  return !(*this == RefinementChoices::None<dim> ||
           *this == RefinementChoices::Isotropic<dim> ||
           this->is_tetrahedral());
}



template <int dim>
inline constexpr RefinementChoice<dim>
RefinementChoice<dim>::cut_axis(const unsigned int i)
{
  // Since there are no axes in 0d this also prevents calling this function
  // in that case.
  AssertIndexRange(i, dim);
  // As with RefinementCase, cutting along an axis sets the bit corresponding to
  // that axis.
  return RefinementChoice<dim>(static_cast<std::uint8_t>(1u << i));
}



template <int dim>
template <class Archive>
inline void
RefinementChoice<dim>::serialize(Archive &archive, const unsigned int)
{
  archive & kind;
}



template <int dim>
inline constexpr std::size_t
RefinementChoice<dim>::memory_consumption()
{
  return sizeof(RefinementChoice<dim>);
}



template <int dim>
inline constexpr unsigned int
RefinementChoice<dim>::new_tetrahedron_line() const
{
  AssertDimension(dim, 3);
  if (*this == internal::make_refinement_choice_from_int<dim>(8))
    return 2;
  else if (*this == internal::make_refinement_choice_from_int<dim>(9))
    return 1;
  else if (*this == internal::make_refinement_choice_from_int<dim>(10))
    return 0;

  DEAL_II_NOT_IMPLEMENTED();
  return numbers::invalid_unsigned_int;
}


DEAL_II_NAMESPACE_CLOSE

#endif
