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


// Test std_cxx26::inplace_vector's assignment functions. Similar to
// inplace_vector_01 (same utilities)


#include <deal.II/base/std_cxx26/inplace_vector.h>

#include <numeric>
#include <vector>

#include "../tests.h"

// Struct for recording which constructor is called and counting total ctor/dtor
// calls
struct A
{
  static int n_ctors;
  static int n_dtors;

  A()
  {
    ++n_ctors;
    deallog << "A::A()" << std::endl;
  }

  A(A &&)
  {
    ++n_ctors;
    deallog << "A::A(A&&)" << std::endl;
  }

  A(const A &)
  {
    ++n_ctors;
    deallog << "A::A(const A&)" << std::endl;
  }

  ~A()
  {
    ++n_dtors;
    deallog << "A::~A()" << std::endl;
  }

  A&
  operator=(const A&)
  {
    deallog << "A::operator=(const A&)" << std::endl;
    return *this;
  }

  A&
  operator=(A&&)
  {
    deallog << "A::operator=(A&&)" << std::endl;
    return *this;
  }
};

int A::n_ctors = 0;
int A::n_dtors = 0;

template <typename T>
void
print_counts()
{
  if constexpr (std::is_same_v<T, A>)
    deallog << "ctors = " << A::n_ctors << " dtors = " << A::n_dtors
            << std::endl;
}

template <typename T>
void
print(const T &t)
{
  deallog << t;
}

template <typename T>
void
print(const std::vector<T> &vec)
{
  deallog << "{";
  if (vec.size() > 0)
    {
      print(vec[0]);
      for (std::size_t i = 1; i < vec.size(); ++i)
        {
          deallog << ", ";
          print(vec[i]);
        }
    }
  deallog << "}";
}

template <typename T, std::size_t N>
void
print(const std_cxx26::inplace_vector<T, N> &vec)
{
  if constexpr (!std::is_same_v<T, A>)
    {
      if (vec.size() > 0)
        {
          print(vec[0]);
          for (std::size_t i = 1; i < vec.size(); ++i)
            {
              deallog << ", ";
              print(vec[i]);
            }
        }
    }
  else
    deallog << "size : " << vec.size();
}

template <typename T>
void
test_assignments()
{
  constexpr bool is_a = std::is_same_v<T, A>;

  std::array<T, 3> as{};
  if constexpr (std::is_same_v<T, int>)
    std::iota(as.begin(), as.end(), -11);
  if constexpr (std::is_same_v<T, std::vector<int>>)
    {
      as[0] = {1, 1, 2};
      as[1] = {42};
      as[2] = {3, 5};
    }

  deallog << "copy assignment" << std::endl;
  {
    std_cxx26::inplace_vector<T, 16> vec1, vec2(2);
    vec2[0] = as[0];
    vec2[1] = as[1];
    vec1    = vec2;

    print(vec1);
    deallog << std::endl;
    print(vec2);
    deallog << std::endl;
  }
  print_counts<T>();
  deallog << std::endl;

  deallog << "move assignment" << std::endl;
  {
    std_cxx26::inplace_vector<T, 16> vec1, vec2(2);
    vec2[0] = as[0];
    vec2[1] = as[1];
    vec1    = std::move(vec2);

    print(vec1);
    deallog << std::endl;
    print(vec2);
    deallog << std::endl;
  }
  print_counts<T>();
  deallog << std::endl;

  deallog << "initializer list assignment" << std::endl;
  {
    std_cxx26::inplace_vector<T, 16> vec;
    vec = {as[0], as[1], as[2]};
    print(vec);
    deallog << std::endl;
  }
  print_counts<T>();
  deallog << std::endl;

  deallog << "range assignment" << std::endl;
  {
    std_cxx26::inplace_vector<T, 16> vec;
    vec.assign(as.begin(), as.end());
    print(vec);
    deallog << std::endl;
  }
  print_counts<T>();
  deallog << std::endl;

  deallog << "count assignment" << std::endl;
  {
    std_cxx26::inplace_vector<T, 16> vec;
    vec.assign(8, as[0]);
    print(vec);
    deallog << std::endl;
  }
  print_counts<T>();
  deallog << std::endl;

  deallog << "initializer list assignment 2" << std::endl;
  {
    std_cxx26::inplace_vector<T, 16> vec;
    vec.assign({as[0], as[1], as[2]});
    print(vec);
    deallog << std::endl;
  }
  print_counts<T>();
  deallog << std::endl;
}

int
main()
{
  initlog();

  deallog.push("assignments");
  deallog << std::endl;
  deallog.push("A");
  test_assignments<A>();
  deallog.pop();

  deallog << std::endl;
  deallog.push("int");
  test_assignments<int>();
  deallog.pop();

  deallog << std::endl;
  deallog.push("vector<int>");
  test_assignments<std::vector<int>>();
  deallog.pop();

  deallog.pop();
}
