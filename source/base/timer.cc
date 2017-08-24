// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>

#if defined(DEAL_II_HAVE_SYS_TIME_H) && defined(DEAL_II_HAVE_SYS_RESOURCE_H)
#  include <sys/time.h>
#  include <sys/resource.h>
#endif

#ifdef DEAL_II_MSVC
#  include <windows.h>
#endif



DEAL_II_NAMESPACE_OPEN

namespace internal
{
  namespace Timer
  {
    namespace
    {
      /**
       * Type trait for checking whether or not a type is a std::chrono::duration.
       */
      template <typename T>
      struct is_duration : std::false_type {};

      /**
       * Specialization to get the right truth value.
       */
      template <typename Rep, typename Period>
      struct is_duration<std::chrono::duration<Rep, Period>> : std::true_type {};

      /**
       * Convert a double precision number with units of seconds into a
       * specified duration type T. Only valid when T is a
       * std::chrono::duration type.
       */
      template <typename T>
      T
      from_seconds(const double time)
      {
        static_assert(is_duration<T>::value,
                      "The template type should be a duration type.");
        return T(std::lround(T::period::den*(time/T::period::num)));
      }

      /**
       * Convert a given duration into a double precision number with units of
       * seconds.
       */
      template <typename Rep, typename Period>
      double
      to_seconds(const std::chrono::duration<Rep, Period> duration)
      {
        return Period::num*double(duration.count())/Period::den;
      }

      /**
       * Return the amount of CPU time that the current process has used so
       * far. Unfortunately, this requires platform-specific calls, so this
       * function returns 0 on platforms that are neither windows nor POSIX.
       */
      template <typename T>
      T
      get_current_cpu_time()
      {
        static_assert(is_duration<T>::value,
                      "The template type should be a duration type.");
        double system_cpu_duration = 0.0;
#ifdef DEAL_II_MSVC
        FILETIME cpuTime, sysTime, createTime, exitTime;
        if (GetProcessTimes(GetCurrentProcess(), &createTime,
                            &exitTime, &sysTime, &cpuTime))
          {
            system_cpu_duration = (double)
                                  (((unsigned long long)cpuTime.dwHighDateTime << 32)
                                   | cpuTime.dwLowDateTime) / 1e6;
          }
        // keep the zero value if GetProcessTimes didn't work
#elif defined(DEAL_II_HAVE_SYS_RESOURCE_H)
        rusage usage;
        getrusage (RUSAGE_SELF, &usage);
        system_cpu_duration = usage.ru_utime.tv_sec + 1.e-6 * usage.ru_utime.tv_usec;
#else
#  warning "Unsupported platform. Porting not finished."
#endif
        return from_seconds<T>(system_cpu_duration);
      }
    }
  }
}

Timer::Timer()
  :
  Timer(MPI_COMM_SELF, /*sync_wall_time=*/false)
{}



Timer::Timer(MPI_Comm mpi_communicator,
             const bool sync_wall_time_)
  :
  current_lap_starting_cpu_time (0.),
  current_lap_starting_wall_time (0.),
  accumulated_cpu_time (0.),
  accumulated_wall_time (0.),
  last_lap_time (numbers::signaling_nan<double>()),
  last_lap_cpu_time (numbers::signaling_nan<double>()),
  running (false),
  mpi_communicator (mpi_communicator),
  sync_wall_time(sync_wall_time_)
{
  last_lap_data.sum = last_lap_data.min = last_lap_data.max = last_lap_data.avg = numbers::signaling_nan<double>();
  last_lap_data.min_index = last_lap_data.max_index = numbers::invalid_unsigned_int;
  accumulated_wall_time_data.sum = accumulated_wall_time_data.min = accumulated_wall_time_data.max = accumulated_wall_time_data.avg = 0.;
  accumulated_wall_time_data.min_index = accumulated_wall_time_data.max_index = 0;

  start();
}



#ifdef DEAL_II_MSVC

namespace
{
  namespace windows
  {
    double wall_clock()
    {
      LARGE_INTEGER freq, time;
      QueryPerformanceFrequency(&freq);
      QueryPerformanceCounter(&time);
      return (double) time.QuadPart / freq.QuadPart;
    }
  }
}

#endif


void Timer::start ()
{
  running = true;
#ifdef DEAL_II_WITH_MPI
  if (sync_wall_time)
    {
      const int ierr = MPI_Barrier(mpi_communicator);
      AssertThrowMPI(ierr);
    }
#endif

#if defined(DEAL_II_HAVE_SYS_TIME_H) && defined(DEAL_II_HAVE_SYS_RESOURCE_H)

//TODO: Break this out into a function like the functions in
//namespace windows above
  struct timeval wall_timer;
  gettimeofday(&wall_timer, nullptr);
  current_lap_starting_wall_time = wall_timer.tv_sec + 1.e-6 * wall_timer.tv_usec;

#elif defined(DEAL_II_MSVC)
  current_lap_starting_wall_time = windows::wall_clock();
#else
#  error "Unsupported platform. Porting not finished."
#endif
  // TODO once we convert all private time variables to chrono types we will
  // not need to call to_seconds here
  current_lap_starting_cpu_time = internal::Timer::to_seconds
                                  (internal::Timer::get_current_cpu_time<std::chrono::nanoseconds>());
}



double Timer::stop ()
{
  if (running)
    {
      running = false;

#if defined(DEAL_II_HAVE_SYS_TIME_H) && defined(DEAL_II_HAVE_SYS_RESOURCE_H)
//TODO: Break this out into a function like the functions in
//namespace windows above
      struct timeval wall_timer;
      gettimeofday(&wall_timer, nullptr);
      last_lap_time = wall_timer.tv_sec + 1.e-6 * wall_timer.tv_usec
                      - current_lap_starting_wall_time;
#elif defined(DEAL_II_MSVC)
      last_lap_time = windows::wall_clock() - current_lap_starting_wall_time;
#else
#  error "Unsupported platform. Porting not finished."
#endif
      // TODO once we convert all private time variables to chrono types we will
      // not need to call to_seconds here
      last_lap_cpu_time = internal::Timer::to_seconds
                          (internal::Timer::get_current_cpu_time<std::chrono::nanoseconds>())
                          - current_lap_starting_cpu_time;

      last_lap_data = Utilities::MPI::min_max_avg (last_lap_time,
                                                   mpi_communicator);
      if (sync_wall_time)
        {
          last_lap_time = last_lap_data.max;
        }
      accumulated_wall_time += last_lap_time;
      accumulated_cpu_time += last_lap_cpu_time;
      accumulated_wall_time_data = Utilities::MPI::min_max_avg (accumulated_wall_time,
                                                                mpi_communicator);
    }
  return accumulated_cpu_time;
}



double Timer::cpu_time() const
{
  if (running)
    {
      const double running_time = internal::Timer::to_seconds
                                  (internal::Timer::get_current_cpu_time<std::chrono::nanoseconds>())
                                  - current_lap_starting_cpu_time
                                  + accumulated_cpu_time;
      return Utilities::MPI::sum (running_time, mpi_communicator);
    }
  else
    {
      return Utilities::MPI::sum (accumulated_cpu_time, mpi_communicator);
    }
}



double Timer::last_cpu_time() const
{
  return last_lap_cpu_time;
}



double Timer::get_lap_time() const
{
  return last_lap_time;
}



double Timer::operator() () const
{
  return cpu_time();
}



double Timer::wall_time () const
{
  if (running)
    {
#if defined(DEAL_II_HAVE_SYS_TIME_H) && defined(DEAL_II_HAVE_SYS_RESOURCE_H)
      struct timeval wall_timer;
      gettimeofday(&wall_timer, nullptr);
      return (wall_timer.tv_sec
              + 1.e-6 * wall_timer.tv_usec
              - current_lap_starting_wall_time
              + accumulated_wall_time);
#else
//TODO[BG]: Do something useful here
      return 0;
#endif
    }
  else
    return accumulated_wall_time;
}



double Timer::last_wall_time () const
{
  return last_lap_time;
}



void Timer::reset ()
{
  last_lap_time = numbers::signaling_nan<double>();
  last_lap_cpu_time = numbers::signaling_nan<double>();
  accumulated_cpu_time = 0.;
  accumulated_wall_time = 0.;
  running         = false;
  last_lap_data.sum = last_lap_data.min = last_lap_data.max = last_lap_data.avg = numbers::signaling_nan<double>();
  last_lap_data.min_index = last_lap_data.max_index = numbers::invalid_unsigned_int;
  accumulated_wall_time_data.sum = accumulated_wall_time_data.min = accumulated_wall_time_data.max = accumulated_wall_time_data.avg = 0.;
  accumulated_wall_time_data.min_index = accumulated_wall_time_data.max_index = 0;
}



/* ---------------------------- TimerOutput -------------------------- */

TimerOutput::TimerOutput (std::ostream &stream,
                          const enum OutputFrequency output_frequency,
                          const enum OutputType output_type)
  :
  output_frequency (output_frequency),
  output_type (output_type),
  out_stream (stream, true),
  output_is_enabled (true),
  mpi_communicator (MPI_COMM_SELF)
{}



TimerOutput::TimerOutput (ConditionalOStream &stream,
                          const enum OutputFrequency output_frequency,
                          const enum OutputType output_type)
  :
  output_frequency (output_frequency),
  output_type (output_type),
  out_stream (stream),
  output_is_enabled (true),
  mpi_communicator (MPI_COMM_SELF)
{}



TimerOutput::TimerOutput (MPI_Comm      mpi_communicator,
                          std::ostream &stream,
                          const enum OutputFrequency output_frequency,
                          const enum OutputType output_type)
  :
  output_frequency (output_frequency),
  output_type (output_type),
  out_stream (stream, true),
  output_is_enabled (true),
  mpi_communicator (mpi_communicator)
{}



TimerOutput::TimerOutput (MPI_Comm      mpi_communicator,
                          ConditionalOStream &stream,
                          const enum OutputFrequency output_frequency,
                          const enum OutputType output_type)
  :
  output_frequency (output_frequency),
  output_type (output_type),
  out_stream (stream),
  output_is_enabled (true),
  mpi_communicator (mpi_communicator)
{}



TimerOutput::~TimerOutput()
{
  try
    {
      while (active_sections.size() > 0)
        leave_subsection();
    }
  catch (...)
    {}

  if ( (output_frequency == summary || output_frequency == every_call_and_summary)
       && output_is_enabled == true)
    print_summary();
}



void
TimerOutput::enter_subsection (const std::string &section_name)
{
  Threads::Mutex::ScopedLock lock (mutex);

  Assert (section_name.empty() == false,
          ExcMessage ("Section string is empty."));

  Assert (std::find (active_sections.begin(), active_sections.end(),
                     section_name) == active_sections.end(),
          ExcMessage (std::string("Cannot enter the already active section <")
                      + section_name + ">."));

  if (sections.find (section_name) == sections.end())
    {
      if (mpi_communicator != MPI_COMM_SELF)
        {
          // create a new timer for this section. the second argument
          // will ensure that we have an MPI barrier before starting
          // and stopping a timer, and this ensures that we get the
          // maximum run time for this section over all processors.
          // The mpi_communicator from TimerOutput is passed to the
          // Timer here, so this Timer will collect timing information
          // among all processes inside mpi_communicator.
          sections[section_name].timer = Timer(mpi_communicator, true);
        }


      sections[section_name].total_cpu_time = 0;
      sections[section_name].total_wall_time = 0;
      sections[section_name].n_calls = 0;
    }

  sections[section_name].timer.reset();
  sections[section_name].timer.start();
  sections[section_name].n_calls++;

  active_sections.push_back (section_name);
}



void
TimerOutput::leave_subsection (const std::string &section_name)
{
  Assert (!active_sections.empty(),
          ExcMessage("Cannot exit any section because none has been entered!"));

  Threads::Mutex::ScopedLock lock (mutex);

  if (section_name != "")
    {
      Assert (sections.find (section_name) != sections.end(),
              ExcMessage ("Cannot delete a section that was never created."));
      Assert (std::find (active_sections.begin(), active_sections.end(),
                         section_name) != active_sections.end(),
              ExcMessage ("Cannot delete a section that has not been entered."));
    }

  // if no string is given, exit the last
  // active section.
  const std::string actual_section_name = (section_name == "" ?
                                           active_sections.back () :
                                           section_name);

  sections[actual_section_name].timer.stop();
  sections[actual_section_name].total_wall_time
  += sections[actual_section_name].timer.wall_time();

  // Get cpu time. On MPI systems, if constructed with an mpi_communicator
  // like MPI_COMM_WORLD, then the Timer will sum up the CPU time between
  // processors among the provided mpi_communicator. Therefore, no
  // communication is needed here.
  const double cpu_time = sections[actual_section_name].timer();
  sections[actual_section_name].total_cpu_time += cpu_time;

  // in case we have to print out something, do that here...
  if ((output_frequency == every_call || output_frequency == every_call_and_summary)
      && output_is_enabled == true)
    {
      std::string output_time;
      std::ostringstream cpu;
      cpu << cpu_time << "s";
      std::ostringstream wall;
      wall << sections[actual_section_name].timer.wall_time() << "s";
      if (output_type == cpu_times)
        output_time = ", CPU time: " + cpu.str();
      else if (output_type == wall_times)
        output_time = ", wall time: " + wall.str() + ".";
      else
        output_time = ", CPU/wall time: " + cpu.str() + " / " + wall.str() + ".";

      out_stream << actual_section_name << output_time
                 << std::endl;
    }

  // delete the index from the list of
  // active ones
  active_sections.erase (std::find (active_sections.begin(), active_sections.end(),
                                    actual_section_name));
}



std::map<std::string, double>
TimerOutput::get_summary_data (const OutputData kind) const
{
  std::map<std::string, double> output;
  for (const auto &section : sections)
    {
      switch (kind)
        {
        case TimerOutput::OutputData::total_cpu_time:
          output[section.first] = section.second.total_cpu_time;
          break;
        case TimerOutput::OutputData::total_wall_time:
          output[section.first] = section.second.total_wall_time;
          break;
        case TimerOutput::OutputData::n_calls:
          output[section.first] = section.second.n_calls;
          break;
        default:
          Assert(false, ExcNotImplemented());
        }
    }
  return output;
}



void
TimerOutput::print_summary () const
{
  // we are going to change the
  // precision and width of output
  // below. store the old values so we
  // can restore it later on
  const std::istream::fmtflags old_flags = out_stream.get_stream().flags();
  const std::streamsize    old_precision = out_stream.get_stream().precision ();
  const std::streamsize    old_width     = out_stream.get_stream().width ();

  // in case we want to write CPU times
  if (output_type != wall_times)
    {
      double total_cpu_time = Utilities::MPI::sum(timer_all(), mpi_communicator);

      // check that the sum of all times is
      // less or equal than the total
      // time. otherwise, we might have
      // generated a lot of overhead in this
      // function.
      double check_time = 0.;
      for (std::map<std::string, Section>::const_iterator
           i = sections.begin(); i!=sections.end(); ++i)
        check_time += i->second.total_cpu_time;

      const double time_gap = check_time-total_cpu_time;
      if (time_gap > 0.0)
        total_cpu_time = check_time;

      // generate a nice table
      out_stream << "\n\n"
                 << "+---------------------------------------------+------------"
                 << "+------------+\n"
                 << "| Total CPU time elapsed since start          |";
      out_stream << std::setw(10) << std::setprecision(3) << std::right;
      out_stream << total_cpu_time << "s |            |\n";
      out_stream << "|                                             |            "
                 << "|            |\n";
      out_stream << "| Section                         | no. calls |";
      out_stream << std::setw(10);
      out_stream << std::setprecision(3);
      out_stream << "  CPU time "  << " | % of total |\n";
      out_stream << "+---------------------------------+-----------+------------"
                 << "+------------+";
      for (std::map<std::string, Section>::const_iterator
           i = sections.begin(); i!=sections.end(); ++i)
        {
          std::string name_out = i->first;

          // resize the array so that it is always
          // of the same size
          unsigned int pos_non_space = name_out.find_first_not_of (" ");
          name_out.erase(0, pos_non_space);
          name_out.resize (32, ' ');
          out_stream << std::endl;
          out_stream << "| " << name_out;
          out_stream << "| ";
          out_stream << std::setw(9);
          out_stream << i->second.n_calls << " |";
          out_stream << std::setw(10);
          out_stream << std::setprecision(3);
          out_stream << i->second.total_cpu_time << "s |";
          out_stream << std::setw(10);
          if (total_cpu_time != 0)
            {
              // if run time was less than 0.1%, just print a zero to avoid
              // printing silly things such as "2.45e-6%". otherwise print
              // the actual percentage
              const double fraction = i->second.total_cpu_time/total_cpu_time;
              if (fraction > 0.001)
                {
                  out_stream << std::setprecision(2);
                  out_stream << fraction * 100;
                }
              else
                out_stream << 0.0;

              out_stream << "% |";
            }
          else
            out_stream << 0.0 << "% |";
        }
      out_stream << std::endl
                 << "+---------------------------------+-----------+"
                 << "------------+------------+\n"
                 << std::endl;

      if (time_gap > 0.0)
        out_stream << std::endl
                   << "Note: The sum of counted times is " << time_gap
                   << " seconds larger than the total time.\n"
                   << "(Timer function may have introduced too much overhead, or different\n"
                   << "section timers may have run at the same time.)" << std::endl;
    }

  // in case we want to write out wallclock times
  if (output_type != cpu_times)
    {
      double total_wall_time = timer_all.wall_time();

      // now generate a nice table
      out_stream << "\n\n"
                 << "+---------------------------------------------+------------"
                 << "+------------+\n"
                 << "| Total wallclock time elapsed since start    |";
      out_stream << std::setw(10) << std::setprecision(3) << std::right;
      out_stream << total_wall_time << "s |            |\n";
      out_stream << "|                                             |            "
                 << "|            |\n";
      out_stream << "| Section                         | no. calls |";
      out_stream << std::setw(10);
      out_stream << std::setprecision(3);
      out_stream << "  wall time | % of total |\n";
      out_stream << "+---------------------------------+-----------+------------"
                 << "+------------+";
      for (std::map<std::string, Section>::const_iterator
           i = sections.begin(); i!=sections.end(); ++i)
        {
          std::string name_out = i->first;

          // resize the array so that it is always
          // of the same size
          unsigned int pos_non_space = name_out.find_first_not_of (" ");
          name_out.erase(0, pos_non_space);
          name_out.resize (32, ' ');
          out_stream << std::endl;
          out_stream << "| " << name_out;
          out_stream << "| ";
          out_stream << std::setw(9);
          out_stream << i->second.n_calls << " |";
          out_stream << std::setw(10);
          out_stream << std::setprecision(3);
          out_stream << i->second.total_wall_time << "s |";
          out_stream << std::setw(10);

          if (total_wall_time != 0)
            {
              // if run time was less than 0.1%, just print a zero to avoid
              // printing silly things such as "2.45e-6%". otherwise print
              // the actual percentage
              const double fraction = i->second.total_wall_time/total_wall_time;
              if (fraction > 0.001)
                {
                  out_stream << std::setprecision(2);
                  out_stream << fraction * 100;
                }
              else
                out_stream << 0.0;

              out_stream << "% |";
            }
          else
            out_stream << 0.0 << "% |";
        }
      out_stream << std::endl
                 << "+---------------------------------+-----------+"
                 << "------------+------------+\n"
                 << std::endl;
    }

  // restore previous precision and width
  out_stream.get_stream().precision (old_precision);
  out_stream.get_stream().width (old_width);
  out_stream.get_stream().flags (old_flags);
}



void
TimerOutput::disable_output ()
{
  output_is_enabled = false;
}



void
TimerOutput::enable_output ()
{
  output_is_enabled = true;
}

void
TimerOutput::reset ()
{
  Threads::Mutex::ScopedLock lock (mutex);
  sections.clear();
  active_sections.clear();
  timer_all.restart();
}


DEAL_II_NAMESPACE_CLOSE
