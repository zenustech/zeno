/*
 *  Timer.h
 *  CSim
 *
 *  Created by Fang Da on 10/9/11.
 *  Copyright 2011 Columbia. All rights reserved.
 *
 */

#ifndef TIMER_H__
#define TIMER_H__

#include <chrono>

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#endif
#include <string>
#include <map>


namespace CSim
{
    //
    //  Usage:
    //
    //  Get absolute time:
    //      double t = Timer::time();
    //      double t = Timer::time(Timer::CPU);
    //      double t = Timer::time(Timer::REAL);
    //
    //  Profiling:
    //      Timer timer(Timer::CPU);    // or other types
    //      timer.start();
    //      // code
    //      timer.stop();
    //      double t = timer.lasttime();
    //
    //  Multiple Profiling:
    //      Timer timer(Timer::CPU);
    //      timer.start();
    //      // profiled code 1
    //      timer.stop();
    //      double t1 = timer.lasttime();
    //      // irrelevant code
    //      timer.start();
    //      // profiled code 2
    //      timer.stop();
    //      double t2 = timer.lasttime();
    //      double t1andt2 = timer.total();
    //
    
    

    
    class Timer
    {
    public:
        enum TimerType
        {
            REAL,
            TIMER_TYPE_COUNT
        };
        
    public:    
        Timer(TimerType type = REAL)
        {
            m_type = type;
            m_count = 0;
            m_total = std::chrono::duration<double>::zero();
            m_time = std::chrono::duration<double>::zero();
        }
        
        ~Timer() 
        { }
         
        inline void start()
        {
            m_start = std::chrono::system_clock::now();
            m_count++;
        }
        
        inline void stop()
        {
            m_time = std::chrono::system_clock::now() - m_start;
            m_total += m_time;
        }
        
        inline double lasttime()
        {
            return m_time.count();
        }
        
        inline double total()
        {
            return m_total.count();
        }
        
        inline int count()
        {
            return m_count;
        }
        
        
    protected:
        TimerType m_type;
        
        std::chrono::time_point<std::chrono::system_clock> m_start;
        std::chrono::duration<double> m_time;
        std::chrono::duration<double> m_total;
        unsigned int m_count;
        
    };

    //
    //  Example usage:
    //
    //      while (...)
    //      {
    //          TimerMan::timer("code").start();
    //          // code
    //          TimerMan::timer("code").stop();
    //      }
    //      ...
    //      std::cout << TimerMan::timer("code").total << std::endl;
    //
    class TimerMan
    {
    public:
        static Timer & timer(const std::string & name);
        
        static void setReport(bool r);
        static void report();
        
        TimerMan();
        ~TimerMan();

        static TimerMan * getSingleton();
        
    protected:
        static TimerMan * s_singleton;
        
        bool m_report;
        std::map<std::string, Timer *> m_timers;
        
    };
}

#endif
