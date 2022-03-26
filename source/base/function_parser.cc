// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


#include <deal.II/base/function_parser.h>
#include <deal.II/base/mu_parser_internal.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>

#include <cmath>
#include <map>

#ifdef DEAL_II_WITH_MUPARSER
#  include <muParser.h>
#endif

DEAL_II_NAMESPACE_OPEN


template <int dim>
const std::vector<std::string> &
FunctionParser<dim>::get_expressions() const
{
  return expressions;
}



template <int dim>
FunctionParser<dim>::FunctionParser(const unsigned int n_components,
                                    const double       initial_time,
                                    const double       h)
  : AutoDerivativeFunction<dim>(h, n_components, initial_time)
  , initialized(false)
  , n_vars(0)
{}


template <int dim>
FunctionParser<dim>::FunctionParser(const std::string &expression,
                                    const std::string &constants,
                                    const std::string &variable_names,
                                    const double       h)
  : AutoDerivativeFunction<dim>(
      h,
      Utilities::split_string_list(expression, ';').size())
  , initialized(false)
  , n_vars(0)
{
  auto constants_map = Patterns::Tools::Convert<ConstMap>::to_value(
    constants,
    Patterns::Map(Patterns::Anything(),
                  Patterns::Double(),
                  0,
                  Patterns::Map::max_int_value,
                  ",",
                  "="));
  initialize(variable_names,
             expression,
             constants_map,
             Utilities::split_string_list(variable_names, ",").size() ==
               dim + 1);
}


#ifdef DEAL_II_WITH_MUPARSER

template <int dim>
void
FunctionParser<dim>::initialize(const std::string &             variables,
                                const std::vector<std::string> &expressions,
                                const std::map<std::string, double> &constants,
                                const bool time_dependent)
{
  this->fp.clear(); // this will reset all thread-local objects

  this->constants   = constants;
  this->var_names   = Utilities::split_string_list(variables, ',');
  this->expressions = expressions;
  AssertThrow(((time_dependent) ? dim + 1 : dim) == var_names.size(),
              ExcMessage("Wrong number of variables"));

  // We check that the number of components of this function matches the
  // number of components passed in as a vector of strings.
  AssertThrow(this->n_components == expressions.size(),
              ExcInvalidExpressionSize(this->n_components, expressions.size()));

  // Now we define how many variables we expect to read in.  We distinguish
  // between two cases: Time dependent problems, and not time dependent
  // problems. In the first case the number of variables is given by the
  // dimension plus one. In the other case, the number of variables is equal
  // to the dimension. Once we parsed the variables string, if none of this is
  // the case, then an exception is thrown.
  if (time_dependent)
    n_vars = dim + 1;
  else
    n_vars = dim;

  // create a parser object for the current thread we can then query in
  // value() and vector_value(). this is not strictly necessary because a user
  // may never call these functions on the current thread, but it gets us
  // error messages about wrong formulas right away
  init_muparser();

  // finally set the initialization bit
  initialized = true;
}

namespace internal
{
  namespace FunctionParserImplementation
  {
    namespace
    {
      /**
       * PIMPL for mu::Parser.
       */
      class Parser : public internal::muParserBase
      {
      public:
        operator mu::Parser &()
        {
          return parser;
        }

        operator const mu::Parser &() const
        {
          return parser;
        }

      protected:
        mu::Parser parser;
      };
    } // namespace
  }   // namespace FunctionParserImplementation
} // namespace internal


template <int dim>
void
FunctionParser<dim>::init_muparser() const
{
  // check that we have not already initialized the parser on the
  // current thread, i.e., that the current function is only called
  // once per thread
  Assert(fp.get().size() == 0, ExcInternalError());

  // initialize the objects for the current thread (fp.get() and
  // vars.get())
  fp.get().reserve(this->n_components);
  vars.get().resize(var_names.size());
  for (unsigned int component = 0; component < this->n_components; ++component)
    {
      fp.get().emplace_back(
        std::make_unique<internal::FunctionParserImplementation::Parser>());
      mu::Parser &parser =
        dynamic_cast<internal::FunctionParserImplementation::Parser &>(
          *fp.get().back());

      for (const auto &constant : constants)
        {
          parser.DefineConst(constant.first, constant.second);
        }

      for (unsigned int iv = 0; iv < var_names.size(); ++iv)
        parser.DefineVar(var_names[iv], &vars.get()[iv]);

      // define some compatibility functions:
      parser.DefineFun("if", internal::FunctionParser::mu_if, true);
      parser.DefineOprt("|", internal::FunctionParser::mu_or, 1);
      parser.DefineOprt("&", internal::FunctionParser::mu_and, 2);
      parser.DefineFun("int", internal::FunctionParser::mu_int, true);
      parser.DefineFun("ceil", internal::FunctionParser::mu_ceil, true);
      parser.DefineFun("cot", internal::FunctionParser::mu_cot, true);
      parser.DefineFun("csc", internal::FunctionParser::mu_csc, true);
      parser.DefineFun("floor", internal::FunctionParser::mu_floor, true);
      parser.DefineFun("sec", internal::FunctionParser::mu_sec, true);
      parser.DefineFun("log", internal::FunctionParser::mu_log, true);
      parser.DefineFun("pow", internal::FunctionParser::mu_pow, true);
      parser.DefineFun("erf", internal::FunctionParser::mu_erf, true);
      parser.DefineFun("erfc", internal::FunctionParser::mu_erfc, true);
      parser.DefineFun("rand_seed",
                       internal::FunctionParser::mu_rand_seed,
                       true);
      parser.DefineFun("rand", internal::FunctionParser::mu_rand, true);

      try
        {
          // muparser expects that functions have no
          // space between the name of the function and the opening
          // parenthesis. this is awkward because it is not backward
          // compatible to the library we used to use before muparser
          // (the fparser library) but also makes no real sense.
          // consequently, in the expressions we set, remove any space
          // we may find after function names
          std::string transformed_expression = expressions[component];

          for (const auto &current_function_name :
               internal::FunctionParser::function_names)
            {
              const unsigned int function_name_length =
                current_function_name.size();

              std::string::size_type pos = 0;
              while (true)
                {
                  // try to find any occurrences of the function name
                  pos = transformed_expression.find(current_function_name, pos);
                  if (pos == std::string::npos)
                    break;

                  // replace whitespace until there no longer is any
                  while ((pos + function_name_length <
                          transformed_expression.size()) &&
                         ((transformed_expression[pos + function_name_length] ==
                           ' ') ||
                          (transformed_expression[pos + function_name_length] ==
                           '\t')))
                    transformed_expression.erase(
                      transformed_expression.begin() + pos +
                      function_name_length);

                  // move the current search position by the size of the
                  // actual function name
                  pos += function_name_length;
                }
            }

          // now use the transformed expression
          parser.SetExpr(transformed_expression);
        }
      catch (mu::ParserError &e)
        {
          std::cerr << "Message:  <" << e.GetMsg() << ">\n";
          std::cerr << "Formula:  <" << e.GetExpr() << ">\n";
          std::cerr << "Token:    <" << e.GetToken() << ">\n";
          std::cerr << "Position: <" << e.GetPos() << ">\n";
          std::cerr << "Errc:     <" << e.GetCode() << ">" << std::endl;
          AssertThrow(false, ExcParseError(e.GetCode(), e.GetMsg()));
        }
    }
}



template <int dim>
void
FunctionParser<dim>::initialize(const std::string &                  vars,
                                const std::string &                  expression,
                                const std::map<std::string, double> &constants,
                                const bool time_dependent)
{
  initialize(vars,
             Utilities::split_string_list(expression, ';'),
             constants,
             time_dependent);
}



template <int dim>
double
FunctionParser<dim>::value(const Point<dim> & p,
                           const unsigned int component) const
{
  Assert(initialized == true, ExcNotInitialized());
  AssertIndexRange(component, this->n_components);

  // initialize the parser if that hasn't happened yet on the current thread
  if (fp.get().size() == 0)
    init_muparser();

  for (unsigned int i = 0; i < dim; ++i)
    vars.get()[i] = p(i);
  if (dim != n_vars)
    vars.get()[dim] = this->get_time();

  try
    {
      Assert(dynamic_cast<internal::FunctionParserImplementation::Parser *>(
               fp.get()[component].get()),
             ExcInternalError());
      // using dynamic_cast in the next line is about 6% slower than
      // static_cast, so use the assertion above for debugging and disable
      // clang-tidy:
      mu::Parser &parser =
        static_cast<internal::FunctionParserImplementation::Parser &>( // NOLINT
          *fp[component]);
      return parser.Eval();
    }
  catch (mu::ParserError &e)
    {
      std::cerr << "Message:  <" << e.GetMsg() << ">\n";
      std::cerr << "Formula:  <" << e.GetExpr() << ">\n";
      std::cerr << "Token:    <" << e.GetToken() << ">\n";
      std::cerr << "Position: <" << e.GetPos() << ">\n";
      std::cerr << "Errc:     <" << e.GetCode() << ">" << std::endl;
      AssertThrow(false, ExcParseError(e.GetCode(), e.GetMsg()));
      return 0.0;
    }
}



template <int dim>
void
FunctionParser<dim>::vector_value(const Point<dim> &p,
                                  Vector<double> &  values) const
{
  Assert(initialized == true, ExcNotInitialized());
  Assert(values.size() == this->n_components,
         ExcDimensionMismatch(values.size(), this->n_components));


  // initialize the parser if that hasn't happened yet on the current thread
  if (fp.get().size() == 0)
    init_muparser();

  for (unsigned int i = 0; i < dim; ++i)
    vars.get()[i] = p(i);
  if (dim != n_vars)
    vars.get()[dim] = this->get_time();

  for (unsigned int component = 0; component < this->n_components; ++component)
    {
      // Same comment in value() applies here too:
      Assert(dynamic_cast<internal::FunctionParserImplementation::Parser *>(
               fp.get()[component].get()),
             ExcInternalError());
      mu::Parser &parser =
        static_cast<internal::FunctionParserImplementation::Parser &>( // NOLINT
          *fp[component]);

      values(component) = parser.Eval();
    }
}

#else


template <int dim>
void
FunctionParser<dim>::initialize(const std::string &,
                                const std::vector<std::string> &,
                                const std::map<std::string, double> &,
                                const bool)
{
  AssertThrow(false, ExcNeedsFunctionparser());
}

template <int dim>
void
FunctionParser<dim>::initialize(const std::string &,
                                const std::string &,
                                const std::map<std::string, double> &,
                                const bool)
{
  AssertThrow(false, ExcNeedsFunctionparser());
}



template <int dim>
double
FunctionParser<dim>::value(const Point<dim> &, unsigned int) const
{
  AssertThrow(false, ExcNeedsFunctionparser());
  return 0.;
}


template <int dim>
void
FunctionParser<dim>::vector_value(const Point<dim> &, Vector<double> &) const
{
  AssertThrow(false, ExcNeedsFunctionparser());
}


#endif

// Explicit Instantiations.

template class FunctionParser<1>;
template class FunctionParser<2>;
template class FunctionParser<3>;

DEAL_II_NAMESPACE_CLOSE
