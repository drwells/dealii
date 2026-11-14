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

// Test the RefinementChoice versions of
// FiniteElement::get_prolongation_matrix() and
// FiniteElement::get_restriction_matrix():
//
// 1. A user-defined element which only overrides the new (RefinementChoice)
//    version of get_prolongation_matrix() is called by both the new and the
//    (deprecated) old interfaces.
// 2. Prolongation reproduces a quadratic function exactly on every child for
//    every RefinementChoice of quadrilaterals, hexahedra, triangles, and
//    tetrahedra.
// 3. For tetrahedra, the old interface (where RefinementCase values 1, 2, and 3
//    correspond to Cut68, Cut57, and Cut49) returns the same matrices as the
//    new one.

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <map>
#include <mutex>

#include "../tests.h"


// An element defined outside the library which computes its prolongation
// matrices lazily and stores them itself.
template <int dim>
class UserQ : public FE_Q<dim>
{
public:
  UserQ(const unsigned int degree)
    : FE_Q<dim>(degree)
  {}

  const FullMatrix<double> &
  get_prolongation_matrix(
    const unsigned int           child,
    const RefinementChoice<dim> &refinement_choice) const override
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++n_calls;
    auto &matrices = cache[int(refinement_choice)];
    if (matrices.empty())
      FETools::compute_embedding_matrices(*this, refinement_choice, matrices);
    return matrices[child];
  }

  // Make the deprecated version which takes a RefinementCase available too.
  using FE_Q<dim>::get_prolongation_matrix;

  mutable unsigned int n_calls = 0;

private:
  mutable std::mutex                                     mutex;
  mutable std::map<int, std::vector<FullMatrix<double>>> cache;
};



double
max_difference(const FullMatrix<double> &a, const FullMatrix<double> &b)
{
  double result = 0.0;
  for (unsigned int i = 0; i < a.m(); ++i)
    for (unsigned int j = 0; j < a.n(); ++j)
      result = std::max(result, std::abs(a(i, j) - b(i, j)));
  return result;
}



template <int dim>
void
test_user_element()
{
  const UserQ<dim>          user_fe(2);
  const FE_Q<dim>           fe_q(2);
  const FiniteElement<dim> &user_fe_base   = user_fe;
  const ReferenceCell<dim>  reference_cell = fe_q.reference_cell();

  double error = 0.0;
  for (const auto &choice : reference_cell.refinement_choices())
    for (unsigned int c = 0; c < reference_cell.n_children(choice); ++c)
      {
        const FullMatrix<double> &expected =
          fe_q.get_prolongation_matrix(c, choice);
        // new interface through the base class and the derived class
        error = std::max(
          error,
          max_difference(expected,
                         user_fe_base.get_prolongation_matrix(c, choice)));
        error =
          std::max(error,
                   max_difference(expected,
                                  user_fe.get_prolongation_matrix(c, choice)));
        // old interface through the base class and the derived class
        const auto refinement_case = static_cast<RefinementCase<dim>>(choice);
        error = std::max(error,
                         max_difference(expected,
                                        user_fe_base.get_prolongation_matrix(
                                          c, refinement_case)));
        error = std::max(
          error,
          max_difference(expected,
                         user_fe.get_prolongation_matrix(c, refinement_case)));
      }

  // the default argument of the old interface still works
  for (unsigned int c = 0; c < reference_cell.n_isotropic_children(); ++c)
    error = std::max(error,
                     max_difference(fe_q.get_prolongation_matrix(c),
                                    user_fe.get_prolongation_matrix(c)));

  deallog << "UserQ<" << dim << ">: override called "
          << (user_fe.n_calls > 0 ? "yes" : "no") << ", max difference "
          << (error < 1e-12 ? "OK" : "FAILED") << std::endl;
}



// Refine a reference cell with the given RefinementChoice
template <int dim>
void
refine_reference_cell(Triangulation<dim>          &tria,
                      const ReferenceCell<dim>    &reference_cell,
                      const RefinementChoice<dim> &choice)
{
  GridGenerator::reference_cell(tria, reference_cell);
  const auto refinement_case =
    internal::legacy_refinement_case_from_refinement_choice(reference_cell,
                                                            choice);
  if (reference_cell == ReferenceCells::get_simplex<dim>() && dim == 3)
    {
      tria.begin_active()->set_refine_flag(
        RefinementCase<dim>::isotropic_refinement);
      tria.begin_active()->set_refine_choice(
        static_cast<std::uint8_t>(refinement_case));
    }
  else
    tria.begin_active()->set_refine_flag(refinement_case);
  tria.execute_coarsening_and_refinement();
}



template <int dim>
double
quadratic(const Point<dim> &p)
{
  double result = 1.0;
  for (unsigned int d = 0; d < dim; ++d)
    result += (d + 1) * p[d] + p[d] * p[(d + 1) % dim];
  return result;
}



template <int dim>
void
test_exactness(const FiniteElement<dim> &fe)
{
  const ReferenceCell<dim> reference_cell = fe.reference_cell();
  const auto              &mapping =
    reference_cell.template get_default_linear_mapping<dim>();

  // nodal values of the quadratic on the parent, which is the reference cell
  Vector<double> parent_values(fe.n_dofs_per_cell());
  for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    parent_values[i] = quadratic(fe.get_unit_support_points()[i]);

  for (const auto &choice : reference_cell.refinement_choices())
    {
      Triangulation<dim> tria;
      refine_reference_cell(tria, reference_cell, choice);

      FEValues<dim> fe_values(mapping,
                              fe,
                              Quadrature<dim>(fe.get_unit_support_points()),
                              update_quadrature_points);

      double         error = 0.0;
      Vector<double> child_values(fe.n_dofs_per_cell());
      for (unsigned int c = 0; c < reference_cell.n_children(choice); ++c)
        {
          fe.get_prolongation_matrix(c, choice).vmult(child_values,
                                                      parent_values);
          fe_values.reinit(tria.begin(0)->child(c));
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            error =
              std::max(error,
                       std::abs(child_values[i] -
                                quadratic(fe_values.quadrature_point(i))));
        }

      deallog << fe.get_name() << " " << choice.to_string() << " (index "
              << reference_cell.refinement_choice_index(choice)
              << "): " << (error < 1e-12 ? "exact" : "NOT EXACT") << std::endl;
    }
}



void
test_tetrahedron_legacy_interface()
{
  const FE_SimplexP<3>    fe(2);
  const ReferenceCell<3>  reference_cell = fe.reference_cell();
  const FiniteElement<3> &fe_base        = fe;
  for (const auto &choice : reference_cell.refinement_choices())
    {
      const auto refinement_case =
        internal::legacy_refinement_case_from_refinement_choice(reference_cell,
                                                                choice);
      double error = 0.0;
      for (unsigned int c = 0; c < reference_cell.n_children(choice); ++c)
        {
          error =
            std::max(error,
                     max_difference(
                       fe_base.get_prolongation_matrix(c, choice),
                       fe_base.get_prolongation_matrix(c, refinement_case)));
          error = std::max(
            error,
            max_difference(fe_base.get_restriction_matrix(c, choice),
                           fe_base.get_restriction_matrix(c, refinement_case)));
        }
      deallog << "Tet " << choice.to_string() << " = legacy RefinementCase "
              << int(refinement_case) << ": "
              << (error == 0.0 ? "same matrices" : "DIFFERENT") << std::endl;
    }
}



int
main()
{
  initlog();

  test_user_element<2>();
  test_user_element<3>();

  test_exactness(FE_Q<2>(2));
  test_exactness(FE_Q<3>(2));
  test_exactness(FE_SimplexP<2>(2));
  test_exactness(FE_SimplexP<3>(2));

  test_tetrahedron_legacy_interface();
}
