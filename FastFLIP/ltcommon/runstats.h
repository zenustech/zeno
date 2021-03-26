/*
 *  runstats.h
 *
 *  Created by tyson on 21/04/11.
 *
 */

// Hold some runtime stats

#ifndef RUNSTATS_H
#define RUNSTATS_H

#include <map>
#include <string>
#include <stdint.h>
#include <vector>

namespace LosTopos {

class RunStats
{
    
public:
    
    RunStats() :
    int_stats(),
    double_stats(),
    per_frame_int_stats(),
    per_frame_double_stats()
    {}
    
    
    typedef std::pair<int, int64_t> PerFrameInt;
    typedef std::pair<int, double> PerFrameDouble;
    
    void set_int( std::string name, int64_t value );
    void add_to_int( std::string name, int64_t increment );
    int64_t get_int( std::string name );
    bool get_int( std::string name, int64_t& value );
    void update_min_int( std::string name, int64_t value );
    void update_max_int( std::string name, int64_t value );
    
    void set_double( std::string name, double value );
    void add_to_double( std::string name, double increment );
    double get_double( std::string name );
    bool get_double( std::string name, double& value );
    void update_min_double( std::string name, double value );
    void update_max_double( std::string name, double value );
    
    void add_per_frame_int( std::string name, int frame, int64_t value );   
    bool get_per_frame_ints( std::string name, std::vector<PerFrameInt>& sequence );
    
    void add_per_frame_double( std::string name, int frame, double value );  
    bool get_per_frame_doubles( std::string name, std::vector<PerFrameDouble>& sequence );
    
    void write_to_file( const char* filename );
    
    void clear();
    
private:
    
    std::map<std::string, int64_t> int_stats;
    std::map<std::string, double> double_stats;
    
    std::map<std::string, std::vector<PerFrameInt> > per_frame_int_stats;
    std::map<std::string, std::vector<PerFrameDouble> > per_frame_double_stats;
    
};

}

#endif
