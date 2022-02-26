/**
 * @file Timer.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-10
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */


#ifndef _TIMER_H_
#define _TIMER_H_

#include <string>
#include <iostream>
#include <boost/timer/timer.hpp>

  class Timer
  {
  public:

    /// Create timer with logging
    Timer(std::string task) : _task(task){
        std::cout << task;
    }

    /// Destructor
    ~Timer()
    { 
        std::cout <<"Time for "<< _task << " : \n" << _timer.format();
    }

  private:

    // Name of task
    std::string _task;

    // Implementation of timer
    boost::timer::cpu_timer _timer;

};
#endif