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
#  include <deal.II/base/exception_macros.h>
#  include <deal.II/base/exceptions.h>

#  include <algorithm>
#  include <array>
#  include <initializer_list>
#  include <iterator>
#  include <limits>
#  include <type_traits>
#  include <utility>
#endif

#include <boost/serialization/split_free.hpp>

DEAL_II_NAMESPACE_OPEN

namespace std_cxx26
{
#ifndef DEAL_II_HAVE_CXX26
  DeclExceptionMsg(ExcCapacityExceeded,
                   "The current operation requires more capacity than the "
                   "container can provide.");

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
      other.resize(0);

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
      if constexpr (std::is_integral_v<InputIterator>)
        assign(static_cast<size_type>(first), last);
      else
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
      return const_reverse_iterator(end());
    }

    constexpr const_reverse_iterator
    rend() const noexcept
    {
      return const_reverse_iterator(begin());
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
      return const_reverse_iterator(cend());
    }

    constexpr const_reverse_iterator
    crend() const noexcept
    {
      return const_reverse_iterator(cbegin());
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
      internal_resize<true>(n);
    }

    void
    resize(size_type n, const T &value)
    {
      internal_resize<true>(n, value);
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
        throw std::out_of_range("inplace_vector::at(): index " +
                                std::to_string(n) + " out of range");
      return elements[n];
    }

    const_reference
    at(size_type n) const
    {
      if (!(n < size()))
        throw std::out_of_range("inplace_vector::at(): index " +
                                std::to_string(n) + " out of range");
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
      return back();
    }

    reference
    push_back(const T &value)
    {
      internal_resize<true>(size() + 1, value);
      return back();
    }

    reference
    push_back(T &&value)
    {
      internal_resize<true>(size() + 1, std::forward<T>(value));
      return back();
    }

    void
    pop_back()
    {
      Assert(!empty(), ExcEmptyObject());
      internal_resize<true>(size() - 1);
    }

    template <class... Args>
    pointer
    try_emplace_back(Args &&...args)
    {
      if (size() == capacity())
        return nullptr;
      internal_resize<true>(size() + 1, std::forward<Args>(args)...);
      return std::addressof(back());
    }

    pointer
    try_push_back(const T &value)
    {
      if (size() == capacity())
        return nullptr;
      internal_resize<true>(size() + 1, value);
      return std::addressof(back());
    }

    pointer
    try_push_back(T &&value)
    {
      if (size() == capacity())
        return nullptr;
      internal_resize<true>(size() + 1, std::forward<T>(value));
      return std::addressof(back());
    }

    template <class... Args>
    reference
    unchecked_emplace_back(Args &&...args)
    {
      Assert(size() < capacity(), ExcCapacityExceeded());
      internal_resize<false>(size() + 1, std::forward<Args>(args)...);
      return back();
    }

    reference
    unchecked_push_back(const T &value)
    {
      Assert(size() < capacity(), ExcCapacityExceeded());
      internal_resize<false>(size() + 1, value);
      return back();
    }

    reference
    unchecked_push_back(T &&value)
    {
      Assert(size() < capacity(), ExcCapacityExceeded());
      internal_resize<false>(size() + 1, std::forward<T>(value));
      return back();
    }

    template <class... Args>
    iterator
    emplace(const_iterator position, Args &&...args)
    {
      const auto index = position - cbegin();
      AssertIndexRange(index, size());
      // Since args may reference *this we have to construct the new object
      // first: do that in the buffer and then rotate so it is in the correct
      // place
      internal_resize<true>(size() + 1, std::forward<Args>(args)...);

      Assert(begin() < end(), ExcInternalError());
      Assert(begin() + index <= end() - 1, ExcInternalError());
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
      if constexpr (std::is_integral_v<InputIterator>)
        {
          return insert(position,
                        static_cast<size_type>(first),
                        static_cast<T>(last));
        }
      else
        {
          // TODO we need a nicer way to check for valid iterators
          const auto index = position - cbegin();
          Assert(position == cend() || index < size(), ExcMessage("out of range"));
          const auto n_new_elements =
            internal_append<InputIterator, true>(first, last);
          Assert(position + n_new_elements <= end(), ExcInternalError());
          std::rotate(begin() + index, begin() + index + n_new_elements, end());
          return begin() + index;
        }
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
      const auto first_index = first - begin();
      const auto last_index  = last - cbegin();
      Assert(first_index <= last_index,
             ExcMessage("The given range is not valid."));
      std::rotate(begin() + first_index, begin() + last_index, end());
      internal_resize<true>(size() - (last - first));

      return begin() + first_index;
    }

    void
    swap(inplace_vector &other) noexcept(
      N == 0 ||
      std::is_nothrow_swappable_v<T> && std::is_nothrow_move_constructible_v<T>)
    {
      auto      &smaller      = size() < other.size() ? *this : other;
      const auto smaller_size = smaller.size();
      auto      &larger       = size() < other.size() ? other : *this;

      using std::swap;
      for (std::size_t i = 0; i < smaller.size(); ++i)
        swap(smaller[i], larger[i]);
      for (std::size_t i = smaller.size(); i < larger.size(); ++i)
        smaller.push_back(std::move(larger[i]));
      AssertDimension(smaller.size(), larger.size());
      larger.resize(smaller_size);
    }

    friend void
    swap(inplace_vector &x,
         inplace_vector &y) noexcept(N == 0 ||
                                     (std::is_nothrow_swappable_v<T> &&
                                      std::is_nothrow_move_constructible_v<T>))
    {
      x.swap(y);
    }

    constexpr void
    clear() noexcept
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
      Assert(n <= capacity(), ExcCapacityExceeded());
      static_assert(std::is_constructible_v<T, Args...>);
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
            new (end()) T(std::forward<Args>(args)...);
            ++n_elements;
          }
    }

    /**
     * Common function for assigning values to the vector from a range of input
     * iterators, first by copying and then by placement new.
     */
    template <typename InputIterator, bool check = false, bool move = false>
    void
    internal_assign(InputIterator first, InputIterator last) noexcept(
      !check && (move ? std::is_nothrow_move_assignable_v<T> :
                        std::is_nothrow_copy_assignable_v<T>))
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
    internal_append(InputIterator first, InputIterator last) noexcept(
      !check && (move ? std::is_nothrow_move_constructible_v<T> :
                        std::is_nothrow_copy_constructible_v<T>))
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

    using buffer_size_type =
      std::conditional_t<N <= std::numeric_limits<unsigned char>::max(),
                         unsigned char,
                         unsigned short>;
    static_assert(
      N <= std::numeric_limits<buffer_size_type>::max(),
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
  save(Archive                               &ar,
       const std_cxx26::inplace_vector<T, N> &vec,
       const unsigned int /*version*/)
  {
    const auto vec_size = vec.size();
    ar        &vec_size;
    if (vec_size > 0)
      for (const auto &v : vec)
        ar &v;
  }

  template <class Archive, typename T, std::size_t N>
  inline void
  load(Archive                         &ar,
       std_cxx26::inplace_vector<T, N> &vec,
       const unsigned int /*version*/)
  {
    decltype(vec.size()) vec_size = 0;
    ar                  &vec_size;
    vec.resize(vec_size);
    for (std::size_t i = 0; i < vec_size; ++i)
      ar &vec[i];
  }
} // namespace std_cxx26

DEAL_II_NAMESPACE_CLOSE

#endif
