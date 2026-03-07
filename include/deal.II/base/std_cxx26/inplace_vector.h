// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------
#ifndef dealii_cxx26_inplace_vector_h
#define dealii_cxx26_inplace_vector_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_HAVE_CXX26
#  include <inplace_vector>
#else
#  include <algorithm>
#  include <array>
#  include <initializer_list>
#  include <iterator>
#  include <limits>
#  include <type_traits>
#  include <utility>
#endif

// boost::serialization::make_array used to be in array.hpp, but was
// moved to a different file in BOOST 1.64
#include <boost/serialization/split_free.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION >= 106400
#  include <boost/serialization/array_wrapper.hpp>
#else
#  include <boost/serialization/array.hpp>
#endif

DEAL_II_NAMESPACE_OPEN

namespace std_cxx26
{
#ifndef DEAL_II_HAVE_CXX26
  /**
   * C++17-implementation of a subset of std::inplace_vector.
   *
   * Differences:
   *
   * 1. The number of elements is limited to the maximum value of an
   *    `unsigned short`, which is typically 65535.
   * 2. Since placement new() is not constexpr prior to C++26, the majority of
   *    the constructors and assignment operators are not constexpr.
   * 3. The range constructor is not present sinces ranges were not available
   *    prior to C++20.
   *
   * See https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2747r2.html
   * for more information.
   */
  template <typename T, std::size_t N>
  class inplace_vector
  {
  public:
    /**
     * Types.
     */
    /** @{ */
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type &;
    using const_reference = const value_type &;
    using pointer         = value_type *;
    using const_pointer   = const value_type *;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    /** @} */

    /**
     * Constructors and destructor.
     */
    /** @{ */
    constexpr inplace_vector() noexcept
      : n_elements(0)
    {}

    explicit inplace_vector(size_type n)
      : n_elements(0)
    {
      internal_resize<true>(n);
    }

    inplace_vector(size_type n, const T &value)
      : n_elements(0)
    {
      if (n > N)
        throw std::bad_alloc();
      internal_append(SingleInputIterator(&value, 0),
                      SingleInputIterator(&value, n));
    }

    template <class InputIterator>
    inplace_vector(InputIterator first, InputIterator last)
      : n_elements(0)
    {
      // If we have two integers (e.g., inplace_vector(42, 42)) then we should
      // convert to the correct types and call the other constructor
      if constexpr (std::is_integral_v<InputIterator>)
        *this =
          inplace_vector(static_cast<size_type>(first), static_cast<T>(last));
      else
        internal_append<InputIterator, true, false>(first, last);
    }

    inplace_vector(const inplace_vector &other)
      : n_elements(0)
    {
      internal_append(other.begin(), other.end());
    }

    inplace_vector(inplace_vector &&other) noexcept(
      N == 0 || std::is_nothrow_move_constructible_v<T>)
      : n_elements(0)
    {
      internal_append<iterator, false, true>(other.begin(), other.end());
      other.clear();
    }

    constexpr inplace_vector(std::initializer_list<T> other)
      : n_elements(0)
    {
      if (other.size() > N)
        throw std::bad_alloc();
      internal_append(other.begin(), other.end());
    }

#  ifdef DEAL_II_WITH_CXX20
    constexpr
#  endif
      ~inplace_vector()
    {
      clear();
    }
    /** @} */

    /**
     * Assignment operators.
     */
    /** @{ */
    inplace_vector &
    operator=(const inplace_vector &other)
    {
      internal_assign(other.begin(), other.end());

      return *this;
    }

    inplace_vector &
    operator=(inplace_vector &&other) noexcept(
      N == 0 || (std::is_nothrow_move_assignable_v<T> &&
                 std::is_nothrow_move_constructible_v<T>))
    {
      internal_assign<iterator, false, true>(other.begin(), other.end());

      return *this;
    }

    inplace_vector &
    operator=(std::initializer_list<T> other)
    {
      if (other.size() > N)
        throw std::bad_alloc();

      internal_assign(other.begin(), other.end());

      return *this;
    }
    /** @} */

    /**
     * Assignment functions.
     */
    /** @{ */
    template <class InputIterator>
    void
    assign(InputIterator first, InputIterator last)
    {
      internal_assign(first, last);
    }

    void
    assign(size_type n, const T &value)
    {
      if (n > N)
        throw std::bad_alloc();

      internal_assign(SingleInputIterator(&value, 0),
                      SingleInputIterator(&value, n));
    }

    void
    assign(std::initializer_list<T> other)
    {
      if (other.size() > N)
        throw std::bad_alloc();

      internal_assign(other.begin(), other.end());
    }
    /** @} */

    /**
     * Comparison.
     */
    constexpr bool
    operator==(inplace_vector &other) const
    {
      return (size() == other.size()) &&
             std::equal(begin(), end(), other.begin());
    }

    constexpr bool
    operator!=(inplace_vector &other) const
    {
      return !(*this == other);
    }

    constexpr bool
    operator<(const inplace_vector &other) const
    {
      return std::lexicographical_compare(begin(),
                                          end(),
                                          other.begin(),
                                          other.end());
    }

    /** @} */

    /**
     * Iterators.
     */
    /** @{ */
    constexpr iterator
    begin() noexcept
    {
      return std::addressof(elements[0]);
    }

    constexpr iterator
    end() noexcept
    {
      return std::addressof(elements[0]) + size();
    }

    constexpr const_iterator
    begin() const noexcept
    {
      return std::addressof(elements[0]);
    }

    constexpr const_iterator
    end() const noexcept
    {
      return std::addressof(elements[0]) + size();
    }

    constexpr reverse_iterator
    rbegin() noexcept
    {
      return reverse_iterator(end());
    }

    constexpr reverse_iterator
    rend() noexcept
    {
      return reverse_iterator(begin());
    }

    constexpr const_reverse_iterator
    rbegin() const noexcept
    {
      return reverse_iterator(end());
    }

    constexpr const_reverse_iterator
    rend() const noexcept
    {
      return reverse_iterator(begin());
    }

    constexpr const_iterator
    cbegin() const noexcept
    {
      return std::addressof(elements[0]);
    }

    constexpr const_iterator
    cend() const noexcept
    {
      return std::addressof(elements[0]) + size();
    }

    constexpr const_reverse_iterator
    crbegin() const noexcept
    {
      return reverse_iterator(cend());
    }

    constexpr const_reverse_iterator
    crend() const noexcept
    {
      return reverse_iterator(cbegin());
    }
    /** @} */

    /**
     * Capacity.
     */
    /** @{ */
    constexpr bool
    empty() const noexcept
    {
      return size() == 0;
    }

    constexpr size_type
    size() const noexcept
    {
      Assert(n_elements <= N, ExcInternalError());
      return n_elements;
    }

    static constexpr size_type
    max_size() noexcept
    {
      return N;
    }

    static constexpr size_type
    capacity() noexcept
    {
      return N;
    }

    void
    resize(size_type n)
    {
      internal_resize(n);
    }

    void
    resize(size_type n, const T &value)
    {
      internal_resize(n, value);
    }

    static constexpr void
    reserve(size_type n)
    {
      if (n > N)
        throw std::bad_alloc();
    }

    static constexpr void
    shrink_to_fit() noexcept
    {}
    /** @} */

    /**
     * Element access.
     */
    /** @{ */
    reference
    operator[](size_type n)
    {
      AssertIndexRange(n, size());
      return elements[n];
    }

    const_reference
    operator[](size_type n) const
    {
      AssertIndexRange(n, size());
      return elements[n];
    }

    reference
    at(size_type n)
    {
      if (!(n < size()))
        throw std::out_of_range();
      return elements[n];
    }

    const_reference
    at(size_type n) const
    {
      if (!(n < size()))
        throw std::out_of_range();
      return elements[n];
    }

    reference
    front()
    {
      Assert(!empty(), ExcEmptyObject());
      return elements[0];
    }

    const_reference
    front() const
    {
      Assert(!empty(), ExcEmptyObject());
      return elements[0];
    }

    reference
    back()
    {
      Assert(!empty(), ExcEmptyObject());
      return elements[size() - 1];
    }

    const_reference
    back() const
    {
      Assert(!empty(), ExcEmptyObject());
      return elements[size() - 1];
    }
    /** @} */

    /**
     * Data access.
     */
    /** @{ */
    T *
    data() noexcept
    {
      return begin();
    }

    const T *
    data() const noexcept
    {
      return cbegin();
    }
    /** @} */

    /**
     * Modifiers.
     */
    /** @{ */
    template <class... Args>
    reference
    emplace_back(Args &&...args)
    {
      internal_resize<true>(size() + 1, std::forward<Args>(args)...);
    }

    reference
    push_back(const T &value)
    {
      internal_resize<true>(size() + 1, value);
    }

    reference
    push_back(T &&value)
    {
      internal_resize<true>(size() + 1, std::forward(value));
    }

    void
    pop_back()
    {
      internal_resize(size() - 1);
      Assert(!empty(), ExcEmptyObject());
    }

    template <class... Args>
    pointer
    try_emplace_back(Args &&...args)
    {
      if (size() == N)
        return nullptr;
      internal_resize(size() + 1, std::forward<Args>(args)...);
      return std::addressof(back());
    }

    pointer
    try_push_back(const T &value)
    {
      if (size() == N)
        return nullptr;
      internal_resize<true>(size() + 1, value);
      return std::addressof(back());
    }

    pointer
    try_push_back(T &&value)
    {
      if (size() == N)
        return nullptr;
      internal_resize<true>(size() + 1, std::forward(value));
      return std::addressof(back());
    }

    template <class... Args>
    reference
    unchecked_emplace_back(Args &&...args)
    {
      internal_resize(size() + 1, std::forward<Args>(args)...);
      Assert(size() < capacity(), ExcCapacityExceeded());
      return back();
    }

    reference
    unchecked_push_back(const T &value)
    {
      internal_resize(size() + 1, value);
      Assert(size() < capacity(), ExcCapacityExceeded());
      return back();
    }

    reference
    unchecked_push_back(T &&value)
    {
      internal_resize(size() + 1, std::forward(value));
      Assert(size() < capacity(), ExcCapacityExceeded());
      return back();
    }

    template <class... Args>
    iterator
    emplace(const_iterator position, Args &&...args)
    {
      const auto index = position - cbegin();
      // Since args may reference *this we have to construct the new object
      // first: do that in the buffer and then rotate so it is in the correct
      // place
      internal_resize<true>(size() + 1, std::forward<Args>(args)...);

      std::rotate(begin() + index, end() - 1, end());

      return begin() + index;
    }

    iterator
    insert(const_iterator position, const T &value)
    {
      return emplace(position, value);
    }

    iterator
    insert(const_iterator position, T &&value)
    {
      return emplace(position, std::move(value));
    }

    iterator
    insert(const_iterator position, size_type n, const T &value)
    {
      return insert(position,
                    SingleInputIterator(&value, 0),
                    SingleInputIterator(&value, n));
    }

    template <typename InputIterator>
    iterator
    insert(const_iterator position, InputIterator first, InputIterator last)
    {
      const auto index = position - cbegin();
      AssertIndexRange(index, size());
      const auto n_new_elements =
        internal_append<InputIterator, true>(first, last);
      Assert(position + n_new_elements <= end(), ExcInternalError());
      std::rotate(position, position + n_new_elements, end());
      return begin() + index;
    }

    iterator
    insert(const_iterator position, std::initializer_list<T> other)
    {
      return insert(position, other.begin(), other.end());
    }

    iterator
    erase(const_iterator position)
    {
      const auto index = position - cbegin();
      AssertIndexRange(index, size());
      Assert(begin() + index + 1 <= end(), ExcInternalError());
      std::rotate(begin() + index, begin() + index + 1, end());
      pop_back();

      return begin() + index;
    }

    iterator
    erase(const_iterator first, const_iterator last)
    {
      const auto index = first - begin();
      std::rotate(first, last, end());
      internal_resize(size() - (last - first));

      return begin() + index;
    }

    void constexpr clear() noexcept
    {
      internal_resize(0);
    }

    /** @} */
  private:
    /**
     * Wrapper class for an iterator-like interface to a single value.
     *
     * Most constructors and assign() overloads work with ranges of iterators.
     * SingleInputIterator allows for calling those functions (or common
     * utilities) from the functions which takes counts and values as well.
     */
    struct SingleInputIterator
    {
      using value_type        = const T;
      using difference_type   = std::ptrdiff_t;
      using reference         = const T &;
      using pointer           = const T *;
      using iterator_category = std::input_iterator_tag;

      SingleInputIterator(const T *value, size_type index)
        : ptr(value)
        , index(index)
      {}

      reference
      operator*() const
      {
        return *ptr;
      }

      bool
      operator==(const SingleInputIterator &other) const
      {
        return ptr == other.ptr && index == other.index;
      }

      bool
      operator!=(const SingleInputIterator &other) const
      {
        return !(*this == other);
      }

      SingleInputIterator &
      operator++()
      {
        ++index;
        return *this;
      }

      const T *ptr;

      std::size_t index;
    };

    /**
     * Common function for resizing the vector.
     */
    template <bool check = false, typename... Args>
    constexpr void
    internal_resize(const size_type n, Args &&...args) noexcept(!check)
    {
      static_assert(std::is_constructible_v<T, Args...>);
      Assert(check || n == 0, ExcInternalError());
      Assert(n <= capacity(), ExcCapacityExceeded());
      if constexpr (check)
        if (n > N)
          throw std::bad_alloc();

      // Here and elsewhere we avoid std::uninitialized_copy() etc. so that we
      // maintain the invariant that n_elements is always correct
      if (n < size())
        for (size_type i = size(); i > n; --i)
          {
            Assert(size() > 0, ExcInternalError());
            (end() - 1)->~T();
            --n_elements;
          }
      else
        for (size_type i = size(); i < n; ++i)
          {
            AssertIndexRange(size(), capacity());
            new (end()) T(args...);
            ++n_elements;
          }
    }

    /**
     * Common function for assigning values to the vector from a range of input
     * iterators, first by copying and then by placement new.
     */
    template <typename InputIterator, bool check = false, bool move = false>
    void
    internal_assign(InputIterator first, InputIterator last)
    {
      static_assert(std::is_convertible_v<decltype(*first), T>);
      size_type i = 0;
      while (i < size() && first != last)
        {
          if constexpr (move)
            elements[i] = std::move(*first);
          else
            elements[i] = *first;
          ++i;
          ++first;
        }
      if (first == last)
        resize(i);
      else
        internal_append<InputIterator, check, move>(first, last);
    }

    /**
     * Common function for creating or moving new values at the end of the
     * vector.
     */
    template <typename InputIterator, bool check = false, bool move = false>
    constexpr size_type
    internal_append(InputIterator first, InputIterator last)
    {
      static_assert(std::is_convertible_v<decltype(*first), T>);

      // TODO: for constexpr-ification with C++17 we need to find a way to copy
      // trivial values over without using new() (since that is not constexpr
      // until C++26)

      size_type count = 0;
      while (first != last)
        {
          AssertIndexRange(size(), capacity());
          if constexpr (check)
            if (size() == N)
              throw std::bad_alloc();
          if constexpr (move)
            new (end()) T(std::move(*first));
          else
            new (end()) T(*first);
          ++n_elements;
          ++first;
          ++count;
        }
      return count;
    }

    /**
     * Prevent initialization of unused elements by placing the array in a
     * union.
     *
     * @note we use std::array to utilize its specialization for N = 0.
     */
    union
    {
      std::array<T, N> elements;
    };

    static constexpr bool use_smaller_type =
      N <= std::numeric_limits<unsigned char>::max();
    using buffer_size_type =
      std::conditional_t<use_smaller_type, unsigned char, unsigned short>;
    static_assert(
      N <= std::numeric_limits<unsigned short>::max(),
      "This class only supports objects of size <= the maximum size of an "
      "unsigned short (typically 65535).");

    /**
     * Present number of elements in the buffer.
     */
    buffer_size_type n_elements;
  };

#else
  using std::inplace_vector;
#endif

  template <class Archive, typename T, std::size_t N>
  inline void
  serialize(Archive                         &ar,
            std_cxx26::inplace_vector<T, N> &t,
            const unsigned int               file_version)
  {
    boost::serialization::split_free(ar, t, file_version);
  }

  /**
   * Write the data of this object to a stream for the purpose of
   * serialization using the [BOOST serialization
   * library](https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/index.html).
   */
  template <class Archive, typename T, std::size_t N>
  inline void
  save(Archive                              &ar,
       const std_cxx26::inplace_vector<T, N> vec,
       const unsigned int /*version*/)
  {
    const auto vec_size = vec.size();
    ar        &vec_size;
    if (vec_size > 0)
      ar &boost::serialization::make_array(vec.data(), vec_size);
  }

  template <class Archive, typename T, std::size_t N>
  inline void
  load(Archive                        &ar,
       std_cxx26::inplace_vector<T, N> vec,
       const unsigned int /*version*/)
  {
    decltype(vec.size()) vec_size;
    ar                  &vec_size;
    vec.resize(vec_size);
    if (vec_size > 0)
      {
        ar &boost::serialization::make_array(vec.data(), vec_size);
      }
  }
} // namespace std_cxx26

DEAL_II_NAMESPACE_CLOSE

#endif
