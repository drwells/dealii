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


// Test std_cxx26::inplace_vector's constructors.


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
test_ctors()
{
  constexpr bool is_a = std::is_same_v<T, A>;

  deallog << "default ctor" << std::endl;
  {
    std_cxx26::inplace_vector<T, 16> vec;
    print(vec);
    deallog << std::endl;
  }
  print_counts<T>();
  deallog << std::endl;

  deallog << "count default ctor" << std::endl;
  {
    std_cxx26::inplace_vector<T, 16> vec(2);
    print(vec);
    deallog << std::endl;
  }
  print_counts<T>();
  deallog << std::endl;

  deallog << "count copy ctor" << std::endl;
  {
    T a{};
    if constexpr (std::is_same_v<T, int>)
      a = 42;
    if constexpr (std::is_same_v<T, std::vector<int>>)
      a = {3, 5};
    std_cxx26::inplace_vector<T, 16> vec(2, a);
    print(vec);
    deallog << std::endl;
  }
  print_counts<T>();
  deallog << std::endl;

  deallog << "iterator range, copy, and move ctor" << std::endl;
  {
    std::array<T, 2> as{};
    if constexpr (std::is_same_v<T, int>)
      std::iota(as.begin(), as.end(), -11);
    if constexpr (std::is_same_v<T, std::vector<int>>)
      {
        as[0] = {1, 1, 2};
        as[1] = {3, 5};
      }
    std_cxx26::inplace_vector<T, 16> vec1(as.begin(), as.end());
    print(vec1);
    deallog << std::endl;
    deallog << "copy ctor" << std::endl;
    std_cxx26::inplace_vector<T, 16> vec2(vec1);
    print(vec2);
    deallog << std::endl;
    deallog << "and move ctor" << std::endl;
    std_cxx26::inplace_vector<T, 16> vec3(std::move(vec1));
    deallog << "moved-from:" << std::endl;
    print(vec1);
    deallog << std::endl;
    deallog << "moved-to:" << std::endl;
    print(vec3);
    deallog << std::endl;
  }
  print_counts<T>();

  deallog << "initializer_list ctor" << std::endl;
  {
    std::array<T, 2> as{};
    if constexpr (std::is_same_v<T, int>)
      std::iota(as.begin(), as.end(), -11);
    if constexpr (std::is_same_v<T, std::vector<int>>)
      {
        as[0] = {1, 1, 2};
        as[1] = {3, 5};
      }
    std_cxx26::inplace_vector<T, 16> vec(
      std::initializer_list<T>{as[0], as[1]});
    print(vec);
    deallog << std::endl;
  }
  print_counts<T>();
}

int
main()
{
  initlog();

  deallog.push("ctors");
  deallog << std::endl;
  deallog.push("A");
  test_ctors<A>();
  deallog.pop();

  deallog << std::endl;
  deallog.push("int");
  test_ctors<int>();
  deallog.pop();

  deallog << std::endl;
  deallog.push("vector<int>");
  test_ctors<std::vector<int>>();
  deallog.pop();

  deallog.pop();
}
