/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */

// Verify convergence rates for various simplex elements

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_project.h>

#include "../tests.h"

template <int dim>
class LinearFunction : public Function<dim>
{
public:
  LinearFunction()
    : Function<dim>(1)
  {}

  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    return p[1];
  }
};

template <int dim>
void
test(const unsigned int degree)
{
  FE_SimplexP<dim> fe(degree);
  deallog << std::endl
          << "----------------------------------------"
          << "----------------------------------------" << std::endl
          << "                                "
          << " FE = " << fe.get_name() << std::endl
          << std::endl;
  QGaussSimplex<dim> quadrature(degree + 1);

  double previous_error = 1.0;

  for (unsigned int r = 0; r < 4; ++r)
    {
      Triangulation<dim> tria_hex, tria_flat, tria;
#if 0
      // having two cells is nice for debugging
      // GridGenerator::subdivided_hyper_cube_with_simplices(tria, 1);

      std::vector<Point<dim>> points;
      if (dim == 2)
        {
          points.emplace_back(0, 0);
          points.emplace_back(1, 0);
          points.emplace_back(0, 1);
          points.emplace_back(-1, 0);
        }
      else
        {
          points.emplace_back(0, 0, 0);
          points.emplace_back(1, 0, 0);
          points.emplace_back(0, 1, 0);
          points.emplace_back(0, 0, 1);
          points.emplace_back(-1, 0, 0);
        }
      std::vector<CellData<dim>> cell_data;
      if (dim == 2)
        {
          cell_data.emplace_back();
          cell_data.back().vertices = {0, 1, 2};
          cell_data.emplace_back();
          cell_data.back().vertices = {3, 0, 2};
        }
      else
        {
          cell_data.emplace_back();
          cell_data.back().vertices = {0, 1, 2, 3};
          cell_data.emplace_back();
          cell_data.back().vertices = {4, 0, 2, 3};
        }

      GridTools::invert_cells_with_negative_measure(points, cell_data);
      tria.create_triangulation(points, cell_data, SubCellData());
#else
      GridGenerator::hyper_cube(tria_hex);
      tria_hex.refine_global(r + 1);
      GridGenerator::flatten_triangulation(tria_hex, tria_flat);
      GridGenerator::convert_hypercube_to_simplex_mesh(tria_flat, tria);
#endif
      deallog << "Number of cells = " << tria.n_active_cells() << std::endl;

      ReferenceCell   reference_cell = tria.begin_active()->reference_cell();
      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      Vector<double>                 cell_errors(tria.n_active_cells());
      Vector<double>                 solution(dof_handler.n_dofs());
      Functions::CosineFunction<dim> function;
      AffineConstraints<double>      constraints;
      constraints.close();
      const auto &mapping =
        reference_cell.template get_default_linear_mapping<dim>();

      VectorTools::project(
        mapping, dof_handler, constraints, quadrature, function, solution);

      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        function,
                                        cell_errors,
                                        quadrature,
                                        VectorTools::Linfty_norm);
      std::vector<Point<dim>> support_points(dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(mapping,
                                           dof_handler,
                                           support_points);
      const double max_error =
        *std::max_element(cell_errors.begin(), cell_errors.end());
      deallog << "max error = " << max_error << std::endl;
      if (max_error != 0.0)
        deallog << "ratio = " << previous_error / max_error << std::endl;
      previous_error = max_error;

      Quadrature<dim> nodal_points(fe.get_unit_support_points());
      FEValues<dim>   fe_values(mapping,
                              fe,
                              nodal_points,
                              update_quadrature_points | update_values);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
              AssertThrow(std::abs(fe_values.shape_value(i, j) -
                                   double(i == j)) < 1e-12,
                          ExcInternalError());

          if (r == 0)
            {
              for (unsigned int l : cell->line_indices())
                {
                  deallog << "line " << l << std::endl
                          << "  "
                          << "vertex " << cell->line(l)->vertex_index(0)
                          << " = " << cell->line(l)->vertex(0) << std::endl
                          << "  "
                          << "vertex " << cell->line(l)->vertex_index(1)
                          << " = " << cell->line(l)->vertex(1) << std::endl
                          << "  "
                          << "orientation = " << cell->line_orientation(l)
                          << std::endl;
                }

              deallog << "vertices =" << std::endl
                      << "  " << cell->vertex_index(0) << ": "
                      << cell->vertex(0) << std::endl
                      << "  " << cell->vertex_index(1) << ": "
                      << cell->vertex(1) << std::endl
                      << "  " << cell->vertex_index(2) << ": "
                      << cell->vertex(2) << std::endl;
              if (dim == 3)
                deallog << "  " << cell->vertex_index(3) << ": "
                        << cell->vertex(3) << std::endl;

              if (dim == 3)
                {
                  for (unsigned int face_n = 0; face_n < cell->n_faces();
                       ++face_n)
                    {
                      auto face = cell->face(face_n);
                      deallog << "face " << face_n << ": "
                              << face->vertex_index(0) << ", "
                              << face->vertex_index(1);
                      if (dim == 3)
                        deallog << ", " << face->vertex_index(2);
                      deallog << ": orientation = "
                              << int(cell->combined_face_orientation(face_n))
                              << std::endl;
                      for (unsigned int line_n = 0; line_n < face->n_lines();
                           ++line_n)
                        {
                          auto line = face->line(line_n);
                          deallog << "  line " << line_n << ": "
                                  << line->vertex_index(0) << ", "
                                  << line->vertex_index(1) << ": orientation = "
                                  << cell->line_orientation(line_n)
                                  << std::endl;
                        }
                    }
                }

              std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
              cell->get_dof_indices(cell_dofs);
              for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                deallog << "DoF " << cell_dofs[i] << std::endl
                        << "  support point " << fe_values.quadrature_point(i)
                        << std::endl;
            }
        }

#if 0
      if (dim == 2)
        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          solution    = 0.0;
          solution[3] = 1.0;
          data_out.add_data_vector(solution, "u");
          data_out.build_patches(2);

          std::ofstream output("out-" + std::to_string(degree) + "-" +
                               std::to_string(r) + ".vtu");
          data_out.write_vtu(output);
        }
#endif
    }
}

int
main()
{
  initlog();

#if 0
  test<2>(1);
  test<2>(2);
#endif
  test<2>(3);

#if 0
  test<3>(1);
  test<3>(2);
#endif
  test<3>(3);
}
