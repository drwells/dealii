// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2021 by the deal.II authors
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

#ifndef dealii_petsc_vector_h
#  define dealii_petsc_vector_h

#  include <deal.II/base/config.h>

#  ifdef DEAL_II_WITH_PETSC

#    include <deal.II/base/index_set.h>
#    include <deal.II/base/partitioner.h>
#    include <deal.II/base/subscriptor.h>

#    include <deal.II/lac/exceptions.h>
#    include <deal.II/lac/petsc_vector_base.h>
#    include <deal.II/lac/vector.h>
#    include <deal.II/lac/vector_operation.h>
#    include <deal.II/lac/vector_type_traits.h>

DEAL_II_NAMESPACE_OPEN

// forward declaration
#    ifndef DOXYGEN
namespace PETScWrappers
{
  namespace MPI
  {
    class Vector;
  }
} // namespace PETScWrappers
#    endif


/**
 * @addtogroup PETScWrappers
 * @{
 */
namespace PETScWrappers
{
  /**
   * @cond internal
   */

  /**
   * A namespace for internal implementation details of the PETScWrapper
   * members.
   * @ingroup PETScWrappers
   */
  namespace internal
  {
    /**
     * Since access to PETSc vectors only goes through functions, rather than
     * by obtaining a reference to a vector element, we need a wrapper class
     * that acts as if it was a reference, and basically redirects all
     * accesses (read and write) to member functions of this class.
     *
     * This class implements such a wrapper: it is initialized with a vector
     * and an element within it, and has a conversion operator to extract the
     * scalar value of this element. It also has a variety of assignment
     * operator for writing to this one element.
     * @ingroup PETScWrappers
     */
    class VectorReference
    {
    public:
      /**
       * Declare type for container size.
       */
      using size_type = types::global_dof_index;

    private:
      /**
       * Constructor. It is made private so as to only allow the actual vector
       * class to create it.
       */
      VectorReference(const MPI::Vector &vector, const size_type index);

    public:
      /*
       * Copy constructor.
       */
      VectorReference(const VectorReference &vector) = default;

      /**
       * This looks like a copy operator, but does something different than
       * usual. In particular, it does not copy the member variables of this
       * reference. Rather, it handles the situation where we have two vectors
       * @p v and @p w, and assign elements like in <tt>v(i)=w(i)</tt>. Here,
       * both left and right hand side of the assignment have data type
       * VectorReference, but what we really mean is to assign the vector
       * elements represented by the two references. This operator implements
       * this operation. Note also that this allows us to make the assignment
       * operator const.
       */
      const VectorReference &
      operator=(const VectorReference &r) const;

      /**
       * The same function as above, but for non-const reference objects. The
       * function is needed since the compiler might otherwise automatically
       * generate a copy operator for non-const objects.
       */
      VectorReference &
      operator=(const VectorReference &r);

      /**
       * Set the referenced element of the vector to <tt>s</tt>.
       */
      const VectorReference &
      operator=(const PetscScalar &s) const;

      /**
       * Add <tt>s</tt> to the referenced element of the vector.
       */
      const VectorReference &
      operator+=(const PetscScalar &s) const;

      /**
       * Subtract <tt>s</tt> from the referenced element of the vector.
       */
      const VectorReference &
      operator-=(const PetscScalar &s) const;

      /**
       * Multiply the referenced element of the vector by <tt>s</tt>.
       */
      const VectorReference &
      operator*=(const PetscScalar &s) const;

      /**
       * Divide the referenced element of the vector by <tt>s</tt>.
       */
      const VectorReference &
      operator/=(const PetscScalar &s) const;

      /**
       * Return the real part of the value of the referenced element.
       */
      PetscReal
      real() const;

      /**
       * Return the imaginary part of the value of the referenced element.
       *
       * @note This operation is not defined for real numbers and an exception
       * is thrown.
       */
      PetscReal
      imag() const;

      /**
       * Convert the reference to an actual value, i.e. return the value of
       * the referenced element of the vector.
       */
      operator PetscScalar() const;
      /**
       * Exception
       */
      DeclException3(
        ExcAccessToNonlocalElement,
        int,
        int,
        int,
        << "You tried to access element " << arg1
        << " of a distributed vector, but only elements in range [" << arg2
        << ',' << arg3 << "] are stored locally and can be accessed."
        << "\n\n"
        << "A common source for this kind of problem is that you "
        << "are passing a 'fully distributed' vector into a function "
        << "that needs read access to vector elements that correspond "
        << "to degrees of freedom on ghost cells (or at least to "
        << "'locally active' degrees of freedom that are not also "
        << "'locally owned'). You need to pass a vector that has these "
        << "elements as ghost entries.");
      /**
       * Exception.
       */
      DeclException2(ExcWrongMode,
                     int,
                     int,
                     << "You tried to do a "
                     << (arg1 == 1 ? "'set'" : (arg1 == 2 ? "'add'" : "???"))
                     << " operation but the vector is currently in "
                     << (arg2 == 1 ? "'set'" : (arg2 == 2 ? "'add'" : "???"))
                     << " mode. You first have to call 'compress()'.");

    private:
      /**
       * Point to the vector we are referencing.
       */
      const MPI::Vector &vector;

      /**
       * Index of the referenced element of the vector.
       */
      const size_type index;

      // Make the vector class a friend, so that it can create objects of the
      // present type.
      friend class ::dealii::PETScWrappers::MPI::Vector;
    };
  } // namespace internal
  /**
   * @endcond
   */


  /**
   * Namespace for PETSc classes that work in parallel over MPI, such as
   * distributed vectors and matrices.
   *
   * @ingroup PETScWrappers
   */
  namespace MPI
  {
    /**
     * Implementation of a parallel vector class based on PETSC and using MPI
     * communication to synchronize distributed operations. All the
     * functionality is actually in the base class, except for the calls to
     * generate a parallel vector. This is possible since PETSc only works on
     * an abstract vector type and internally distributes to functions that do
     * the actual work depending on the actual vector type (much like using
     * virtual functions). Only the functions creating a vector of specific
     * type differ, and are implemented in this particular class.
     *
     *
     * <h3>Parallel communication model</h3>
     *
     * The parallel functionality of PETSc is built on top of the Message
     * Passing Interface (MPI). MPI's communication model is built on
     * collective communications: if one process wants something from another,
     * that other process has to be willing to accept this communication. A
     * process cannot query data from another process by calling a remote
     * function, without that other process expecting such a transaction. The
     * consequence is that most of the operations in the base class of this
     * class have to be called collectively. For example, if you want to
     * compute the l2 norm of a parallel vector, @em all processes across
     * which this vector is shared have to call the @p l2_norm function. If
     * you don't do this, but instead only call the @p l2_norm function on one
     * process, then the following happens: This one process will call one of
     * the collective MPI functions and wait for all the other processes to
     * join in on this. Since the other processes don't call this function,
     * you will either get a time-out on the first process, or, worse, by the
     * time the next a call to a PETSc function generates an MPI message on
     * the other processes, you will get a cryptic message that only a subset
     * of processes attempted a communication. These bugs can be very hard to
     * figure out, unless you are well-acquainted with the communication model
     * of MPI, and know which functions may generate MPI messages.
     *
     * One particular case, where an MPI message may be generated unexpectedly
     * is discussed below.
     *
     *
     * <h3>Accessing individual elements of a vector</h3>
     *
     * PETSc does allow read access to individual elements of a vector, but in
     * the distributed case only to elements that are stored locally. We
     * implement this through calls like <tt>d=vec(i)</tt>. However, if you
     * access an element outside the locally stored range, an exception is
     * generated.
     *
     * In contrast to read access, PETSc (and the respective deal.II wrapper
     * classes) allow to write (or add) to individual elements of vectors,
     * even if they are stored on a different process. You can do this
     * writing, for example, <tt>vec(i)=d</tt> or <tt>vec(i)+=d</tt>, or
     * similar operations. There is one catch, however, that may lead to very
     * confusing error messages: PETSc requires application programs to call
     * the compress() function when they switch from adding, to elements to
     * writing to elements. The reasoning is that all processes might
     * accumulate addition operations to elements, even if multiple processes
     * write to the same elements. By the time we call compress() the next
     * time, all these additions are executed. However, if one process adds to
     * an element, and another overwrites to it, the order of execution would
     * yield non-deterministic behavior if we don't make sure that a
     * synchronization with compress() happens in between.
     *
     * In order to make sure these calls to compress() happen at the
     * appropriate time, the deal.II wrappers keep a state variable that store
     * which is the presently allowed operation: additions or writes. If it
     * encounters an operation of the opposite kind, it calls compress() and
     * flips the state. This can sometimes lead to very confusing behavior, in
     * code that may for example look like this:
     * @code
     *   PETScWrappers::MPI::Vector vector;
     *   ...
     *   // do some write operations on the vector
     *   for (unsigned int i=0; i<vector.size(); ++i)
     *     vector(i) = i;
     *
     *   // do some additions to vector elements, but only for some elements
     *   for (unsigned int i=0; i<vector.size(); ++i)
     *     if (some_condition(i) == true)
     *       vector(i) += 1;
     *
     *   // do another collective operation
     *   const double norm = vector.l2_norm();
     * @endcode
     *
     * This code can run into trouble: by the time we see the first addition
     * operation, we need to flush the overwrite buffers for the vector, and
     * the deal.II library will do so by calling compress(). However, it will
     * only do so for all processes that actually do an addition -- if the
     * condition is never true for one of the processes, then this one will
     * not get to the actual compress() call, whereas all the other ones do.
     * This gets us into trouble, since all the other processes hang in the
     * call to flush the write buffers, while the one other process advances
     * to the call to compute the l2 norm. At this time, you will get an error
     * that some operation was attempted by only a subset of processes. This
     * behavior may seem surprising, unless you know that write/addition
     * operations on single elements may trigger this behavior.
     *
     * The problem described here may be avoided by placing additional calls
     * to compress(), or making sure that all processes do the same type of
     * operations at the same time, for example by placing zero additions if
     * necessary.
     *
     * @see
     * @ref GlossGhostedVector "vectors with ghost elements"
     *
     * @ingroup PETScWrappers
     * @ingroup Vectors
     */
    class Vector : public VectorBase
    {
    public:
      /**
       * Declare some of the standard types used in all containers. These types
       * parallel those in the <tt>C++</tt> standard libraries
       * <tt>vector<...></tt> class.
       */
      using reference       = internal::VectorReference;
      using const_reference = const internal::VectorReference;
      using size_type       = types::global_dof_index;

      /**
       * Default constructor. Initialize the vector as empty.
       */
      Vector();

      /**
       * Constructor. Set dimension to @p n and initialize all elements with
       * zero.
       *
       * @arg locally_owned_size denotes the size of the chunk that shall be
       * stored on the present process.
       *
       * @arg communicator denotes the MPI communicator over which the
       * different parts of the vector shall communicate
       *
       * The constructor is made explicit to avoid accidents like this:
       * <tt>v=0;</tt>. Presumably, the user wants to set every element of the
       * vector to zero, but instead, what happens is this call:
       * <tt>v=Vector@<number@>(0);</tt>, i.e. the vector is replaced by one
       * of length zero.
       */
      explicit Vector(const MPI_Comm &communicator,
                      const size_type n,
                      const size_type locally_owned_size);

      /**
       * Copy-constructor from deal.II vectors. Sets the dimension to that of
       * the given vector, and copies all elements.
       *
       * @arg locally_owned_size denotes the size of the chunk that shall be
       * stored on the present process.
       *
       * @arg communicator denotes the MPI communicator over which the
       * different parts of the vector shall communicate
       */
      template <typename Number>
      explicit Vector(const MPI_Comm &              communicator,
                      const dealii::Vector<Number> &v,
                      const size_type               locally_owned_size);


      /**
       * Copy-constructor the values from a PETSc wrapper vector class.
       *
       * @arg local_size denotes the size of the chunk that shall be stored on
       * the present process.
       *
       * @arg communicator denotes the MPI communicator over which the
       * different parts of the vector shall communicate
       *
       * @deprecated The use of objects that are explicitly of type VectorBase
       * is deprecated: use PETScWrappers::MPI::Vector instead.
       */
      DEAL_II_DEPRECATED
      explicit Vector(const MPI_Comm &  communicator,
                      const VectorBase &v,
                      const size_type   local_size);

      /**
       * Construct a new parallel ghosted PETSc vector from IndexSets.
       *
       * Note that @p local must be ascending and 1:1, see
       * IndexSet::is_ascending_and_one_to_one().  In particular, the DoFs in
       * @p local need to be contiguous, meaning you can only create vectors
       * from a DoFHandler with several finite element components if they are
       * not reordered by component (use a PETScWrappers::BlockVector
       * otherwise).  The global size of the vector is determined by
       * local.size(). The global indices in @p ghost are supplied as ghost
       * indices so that they can be read locally.
       *
       * Note that the @p ghost IndexSet may be empty and that any indices
       * already contained in @p local are ignored during construction. That
       * way, the ghost parameter can equal the set of locally relevant
       * degrees of freedom, see step-32.
       *
       * @note This operation always creates a ghosted vector, which is considered
       * read-only.
       *
       * @see
       * @ref GlossGhostedVector "vectors with ghost elements"
       */
      Vector(const IndexSet &local,
             const IndexSet &ghost,
             const MPI_Comm &communicator);

      /**
       * Construct a new parallel PETSc vector without ghost elements from an
       * IndexSet.
       *
       * Note that @p local must be ascending and 1:1, see
       * IndexSet::is_ascending_and_one_to_one().  In particular, the DoFs in
       * @p local need to be contiguous, meaning you can only create vectors
       * from a DoFHandler with several finite element components if they are
       * not reordered by component (use a PETScWrappers::BlockVector
       * otherwise).
       */
      explicit Vector(const IndexSet &local, const MPI_Comm &communicator);

      /**
       * Copy constructor.
       */
      Vector(const Vector &v);

      /**
       * Release all memory and return to a state just like after having
       * called the default constructor.
       */
      virtual void
      clear() override;

      /**
       * Copy the given vector. Resize the present vector if necessary. Also
       * take over the MPI communicator of @p v.
       */
      Vector &
      operator=(const Vector &v);

      /**
       * Set all components of the vector to the given number @p s. Simply
       * pass this down to the base class, but we still need to declare this
       * function to make the example given in the discussion about making the
       * constructor explicit work.
       */
      Vector &
      operator=(const PetscScalar s);

      /**
       * Copy the values of a deal.II vector (as opposed to those of the PETSc
       * vector wrapper class) into this object.
       *
       * Contrary to the case of sequential vectors, this operators requires
       * that the present vector already has the correct size, since we need
       * to have a partition and a communicator present which we otherwise
       * can't get from the source vector.
       */
      template <typename number>
      Vector &
      operator=(const dealii::Vector<number> &v);

      /**
       * Change the dimension of the vector to @p N. It is unspecified how
       * resizing the vector affects the memory allocation of this object;
       * i.e., it is not guaranteed that resizing it to a smaller size
       * actually also reduces memory consumption, or if for efficiency the
       * same amount of memory is used
       *
       * @p locally_owned_size denotes how many of the @p N values shall be
       * stored locally on the present process. for less data.
       *
       * @p communicator denotes the MPI communicator henceforth to be used
       * for this vector.
       *
       * If @p omit_zeroing_entries is false, the vector is filled by zeros.
       * Otherwise, the elements are left an unspecified state.
       */
      void
      reinit(const MPI_Comm &communicator,
             const size_type N,
             const size_type locally_owned_size,
             const bool      omit_zeroing_entries = false);

      /**
       * Change the dimension to that of the vector @p v, and also take over
       * the partitioning into local sizes as well as the MPI communicator.
       * The same applies as for the other @p reinit function.
       *
       * The elements of @p v are not copied, i.e. this function is the same
       * as calling <tt>reinit(v.size(), v.locally_owned_size(),
       * omit_zeroing_entries)</tt>.
       */
      void
      reinit(const Vector &v, const bool omit_zeroing_entries = false);

      /**
       * Reinit as a vector with ghost elements. See the constructor with
       * same signature for more details.
       *
       * @see
       * @ref GlossGhostedVector "vectors with ghost elements"
       */
      void
      reinit(const IndexSet &local,
             const IndexSet &ghost,
             const MPI_Comm &communicator);

      /**
       * Reinit as a vector without ghost elements. See constructor with same
       * signature for more details.
       *
       * @see
       * @ref GlossGhostedVector "vectors with ghost elements"
       */
      void
      reinit(const IndexSet &local, const MPI_Comm &communicator);

      /**
       * Initialize the vector given to the parallel partitioning described in
       * @p partitioner.
       */
      void
      reinit(
        const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner);

      /**
       * Provide access to a given element, both read and write.
       */
      reference
      operator()(const size_type index);

      /**
       * Provide read-only access to an element.
       */
      PetscScalar
      operator()(const size_type index) const;

      /**
       * Provide access to a given element, both read and write.
       *
       * Exactly the same as operator().
       */
      reference
      operator[](const size_type index);

      /**
       * Provide read-only access to an element.
       *
       * Exactly the same as operator().
       */
      PetscScalar
      operator[](const size_type index) const;

      /**
       * Return a reference to the MPI communicator object in use with this
       * vector.
       */
      const MPI_Comm &
      get_mpi_communicator() const override;

      /**
       * Print to a stream. @p precision denotes the desired precision with
       * which values shall be printed, @p scientific whether scientific
       * notation shall be used. If @p across is @p true then the vector is
       * printed in a line, while if @p false then the elements are printed on
       * a separate line each.
       *
       * @note This function overloads the one in the base class to ensure
       * that the right thing happens for parallel vectors that are
       * distributed across processors.
       */
      void
      print(std::ostream &     out,
            const unsigned int precision  = 3,
            const bool         scientific = true,
            const bool         across     = true) const;

      /**
       * @copydoc PETScWrappers::VectorBase::all_zero()
       *
       * @note This function overloads the one in the base class to make this
       * a collective operation.
       */
      bool
      all_zero() const;

    protected:
      /**
       * Create a vector of length @p n. For this class, we create a parallel
       * vector. @p n denotes the total size of the vector to be created. @p
       * locally_owned_size denotes how many of these elements shall be stored
       * locally.
       */
      virtual void
      create_vector(const size_type n, const size_type locally_owned_size);



      /**
       * Create a vector of global length @p n, local size @p
       * locally_owned_size and with the specified ghost indices. Note that
       * you need to call update_ghost_values() before accessing those.
       */
      virtual void
      create_vector(const size_type n,
                    const size_type locally_owned_size,
                    const IndexSet &ghostnodes);


    private:
      /**
       * Copy of the communicator object to be used for this parallel vector.
       */
      MPI_Comm communicator;

      /**
       * Make the reference class a friend.
       */
      friend class internal::VectorReference;
    };
  } // namespace MPI


  // ------------------ template and inline functions -------------


#    ifndef DOXYGEN
  namespace internal
  {
    inline VectorReference::VectorReference(const MPI::Vector &vector,
                                            const size_type    index)
      : vector(vector)
      , index(index)
    {}


    inline const VectorReference &
    VectorReference::operator=(const VectorReference &r) const
    {
      // as explained in the class
      // documentation, this is not the copy
      // operator. so simply pass on to the
      // "correct" assignment operator
      *this = static_cast<PetscScalar>(r);

      return *this;
    }



    inline VectorReference &
    VectorReference::operator=(const VectorReference &r)
    {
      // as explained in the class
      // documentation, this is not the copy
      // operator. so simply pass on to the
      // "correct" assignment operator
      *this = static_cast<PetscScalar>(r);

      return *this;
    }



    inline const VectorReference &
    VectorReference::operator=(const PetscScalar &value) const
    {
      Assert((vector.last_action == VectorOperation::insert) ||
               (vector.last_action == VectorOperation::unknown),
             ExcWrongMode(VectorOperation::insert, vector.last_action));

      Assert(!vector.has_ghost_elements(), ExcGhostsPresent());

      const PetscInt petsc_i = index;

      const PetscErrorCode ierr =
        VecSetValues(vector, 1, &petsc_i, &value, INSERT_VALUES);
      AssertThrow(ierr == 0, ExcPETScError(ierr));

      vector.last_action = VectorOperation::insert;

      return *this;
    }



    inline const VectorReference &
    VectorReference::operator+=(const PetscScalar &value) const
    {
      Assert((vector.last_action == VectorOperation::add) ||
               (vector.last_action == VectorOperation::unknown),
             ExcWrongMode(VectorOperation::add, vector.last_action));

      Assert(!vector.has_ghost_elements(), ExcGhostsPresent());

      vector.last_action = VectorOperation::add;

      // we have to do above actions in any
      // case to be consistent with the MPI
      // communication model (see the
      // comments in the documentation of
      // PETScWrappers::MPI::Vector), but we
      // can save some work if the addend is
      // zero
      if (value == PetscScalar())
        return *this;

      // use the PETSc function to add something
      const PetscInt       petsc_i = index;
      const PetscErrorCode ierr =
        VecSetValues(vector, 1, &petsc_i, &value, ADD_VALUES);
      AssertThrow(ierr == 0, ExcPETScError(ierr));


      return *this;
    }



    inline const VectorReference &
    VectorReference::operator-=(const PetscScalar &value) const
    {
      Assert((vector.last_action == VectorOperation::add) ||
               (vector.last_action == VectorOperation::unknown),
             ExcWrongMode(VectorOperation::add, vector.last_action));

      Assert(!vector.has_ghost_elements(), ExcGhostsPresent());

      vector.last_action = VectorOperation::add;

      // we have to do above actions in any
      // case to be consistent with the MPI
      // communication model (see the
      // comments in the documentation of
      // PETScWrappers::MPI::Vector), but we
      // can save some work if the addend is
      // zero
      if (value == PetscScalar())
        return *this;

      // use the PETSc function to
      // add something
      const PetscInt       petsc_i     = index;
      const PetscScalar    subtractand = -value;
      const PetscErrorCode ierr =
        VecSetValues(vector, 1, &petsc_i, &subtractand, ADD_VALUES);
      AssertThrow(ierr == 0, ExcPETScError(ierr));

      return *this;
    }



    inline const VectorReference &
    VectorReference::operator*=(const PetscScalar &value) const
    {
      Assert((vector.last_action == VectorOperation::insert) ||
               (vector.last_action == VectorOperation::unknown),
             ExcWrongMode(VectorOperation::insert, vector.last_action));

      Assert(!vector.has_ghost_elements(), ExcGhostsPresent());

      vector.last_action = VectorOperation::insert;

      // we have to do above actions in any
      // case to be consistent with the MPI
      // communication model (see the
      // comments in the documentation of
      // PETScWrappers::MPI::Vector), but we
      // can save some work if the factor is
      // one
      if (value == 1.)
        return *this;

      const PetscInt    petsc_i   = index;
      const PetscScalar new_value = static_cast<PetscScalar>(*this) * value;

      const PetscErrorCode ierr =
        VecSetValues(vector, 1, &petsc_i, &new_value, INSERT_VALUES);
      AssertThrow(ierr == 0, ExcPETScError(ierr));

      return *this;
    }



    inline const VectorReference &
    VectorReference::operator/=(const PetscScalar &value) const
    {
      Assert((vector.last_action == VectorOperation::insert) ||
               (vector.last_action == VectorOperation::unknown),
             ExcWrongMode(VectorOperation::insert, vector.last_action));

      Assert(!vector.has_ghost_elements(), ExcGhostsPresent());

      vector.last_action = VectorOperation::insert;

      // we have to do above actions in any
      // case to be consistent with the MPI
      // communication model (see the
      // comments in the documentation of
      // PETScWrappers::MPI::Vector), but we
      // can save some work if the factor is
      // one
      if (value == 1.)
        return *this;

      const PetscInt    petsc_i   = index;
      const PetscScalar new_value = static_cast<PetscScalar>(*this) / value;

      const PetscErrorCode ierr =
        VecSetValues(vector, 1, &petsc_i, &new_value, INSERT_VALUES);
      AssertThrow(ierr == 0, ExcPETScError(ierr));

      return *this;
    }



    inline PetscReal
    VectorReference::real() const
    {
#      ifndef PETSC_USE_COMPLEX
      return static_cast<PetscScalar>(*this);
#      else
      return PetscRealPart(static_cast<PetscScalar>(*this));
#      endif
    }



    inline PetscReal
    VectorReference::imag() const
    {
#      ifndef PETSC_USE_COMPLEX
      return PetscReal(0);
#      else
      return PetscImaginaryPart(static_cast<PetscScalar>(*this));
#      endif
    }
  } // namespace internal

  namespace MPI
  {
    /**
     * Global function @p swap which overloads the default implementation of
     * the C++ standard library which uses a temporary object. The function
     * simply exchanges the data of the two vectors.
     *
     * @relatesalso PETScWrappers::MPI::Vector
     */
    inline void
    swap(Vector &u, Vector &v)
    {
      u.swap(v);
    }

    template <typename number>
    Vector::Vector(const MPI_Comm &              communicator,
                   const dealii::Vector<number> &v,
                   const size_type               locally_owned_size)
      : communicator(communicator)
    {
      Vector::create_vector(v.size(), locally_owned_size);

      *this = v;
    }



    inline Vector &
    Vector::operator=(const PetscScalar s)
    {
      VectorBase::operator=(s);

      return *this;
    }



    template <typename number>
    inline Vector &
    Vector::operator=(const dealii::Vector<number> &v)
    {
      Assert(size() == v.size(), ExcDimensionMismatch(size(), v.size()));

      // FIXME: the following isn't necessarily fast, but this is due to
      // the fact that PETSc doesn't offer an inlined access operator.
      //
      // if someone wants to contribute some code: to make this code
      // faster, one could either first convert all values to PetscScalar,
      // and then set them all at once using VecSetValues. This has the
      // drawback that it could take quite some memory, if the vector is
      // large, and it would in addition allocate memory on the heap, which
      // is expensive. an alternative would be to split the vector into
      // chunks of, say, 128 elements, convert a chunk at a time and set it
      // in the output vector using VecSetValues. since 128 elements is
      // small enough, this could easily be allocated on the stack (as a
      // local variable) which would make the whole thing much more
      // efficient.
      //
      // a second way to make things faster is for the special case that
      // number==PetscScalar. we could then declare a specialization of
      // this template, and omit the conversion. the problem with this is
      // that the best we can do is to use VecSetValues, but this isn't
      // very efficient either: it wants to see an array of indices, which
      // in this case a) again takes up a whole lot of memory on the heap,
      // and b) is totally dumb since its content would simply be the
      // sequence 0,1,2,3,...,n. the best of all worlds would probably be a
      // function in Petsc that would take a pointer to an array of
      // PetscScalar values and simply copy n elements verbatim into the
      // vector...
      for (size_type i = 0; i < v.size(); ++i)
        (*this)(i) = v(i);

      compress(::dealii::VectorOperation::insert);

      return *this;
    }



    inline internal::VectorReference
    Vector::operator()(const size_type index)
    {
      return internal::VectorReference(*this, index);
    }



    inline internal::VectorReference
    Vector::operator[](const size_type index)
    {
      return operator()(index);
    }



    inline PetscScalar
    Vector::operator()(const size_type index) const
    {
      return static_cast<PetscScalar>(internal::VectorReference(*this, index));
    }



    inline PetscScalar
    Vector::operator[](const size_type index) const
    {
      return operator()(index);
    }



    inline const MPI_Comm &
    Vector::get_mpi_communicator() const
    {
      return communicator;
    }

#    endif // DOXYGEN
  }        // namespace MPI
} // namespace PETScWrappers

namespace internal
{
  namespace LinearOperatorImplementation
  {
    template <typename>
    class ReinitHelper;

    /**
     * A helper class used internally in linear_operator.h. Specialization for
     * PETScWrappers::MPI::Vector.
     */
    template <>
    class ReinitHelper<PETScWrappers::MPI::Vector>
    {
    public:
      template <typename Matrix>
      static void
      reinit_range_vector(const Matrix &              matrix,
                          PETScWrappers::MPI::Vector &v,
                          bool /*omit_zeroing_entries*/)
      {
        v.reinit(matrix.locally_owned_range_indices(),
                 matrix.get_mpi_communicator());
      }

      template <typename Matrix>
      static void
      reinit_domain_vector(const Matrix &              matrix,
                           PETScWrappers::MPI::Vector &v,
                           bool /*omit_zeroing_entries*/)
      {
        v.reinit(matrix.locally_owned_domain_indices(),
                 matrix.get_mpi_communicator());
      }
    };

  } // namespace LinearOperatorImplementation
} /* namespace internal */

/** @} */


/**
 * Declare dealii::PETScWrappers::MPI::Vector as distributed vector.
 */
template <>
struct is_serial_vector<PETScWrappers::MPI::Vector> : std::false_type
{};


DEAL_II_NAMESPACE_CLOSE

#  endif // DEAL_II_WITH_PETSC

#endif
/*------------------------- petsc_vector.h -------------------------*/
