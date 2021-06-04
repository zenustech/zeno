/*
 *  runstats.cpp
 *
 *  Created by tyson on 21/04/11.
 *
 */

#include <runstats.h>

#include <commonoptions.h>
#include <fstream>
#include <algorithm>
namespace LosTopos {

LosTopos::RunStats g_stats;
    
// ------------------------------------------------------------------

void RunStats::set_int( std::string name, int64_t value )
{
    int_stats[name] = value;
}

void RunStats::add_to_int( std::string name, int64_t increment )
{
    int64_t value = 0;
    bool exists = get_int( name, value );
    if ( !exists )
    {
        value = 0;
    }
    value += increment;
    set_int( name, value );
}

int64_t RunStats::get_int( std::string name )
{
    std::map<std::string, int64_t>::iterator iter = int_stats.find( name );
    if (  iter == int_stats.end() ) { return ~0; }
    return iter->second;
}

bool RunStats::get_int( std::string name, int64_t& value )
{
    std::map<std::string, int64_t>::iterator iter = int_stats.find( name );
    if ( iter == int_stats.end() )
    {
        return false;
    }
    value = iter->second;
    return true;  
}

void RunStats::update_min_int( std::string name, int64_t value )
{
    int64_t current_min;
    bool exists = get_int( name, current_min );
    if ( !exists ) { current_min = value; }
    int_stats[name] = std::min( value, current_min );
}

void RunStats::update_max_int( std::string name, int64_t value )
{
    int64_t current_max;
    bool exists = get_int( name, current_max );
    if ( !exists ) { current_max = value; }
    int_stats[name] = std::max( value, current_max );
}

// ------------------------------------------------------------------

void RunStats::set_double( std::string name, double value )
{
    double_stats[name] = value;
}

void RunStats::add_to_double( std::string name, double increment )
{
    double value = 0;
    bool exists = get_double( name, value );
    if ( !exists )
    {
        value = 0;
    }
    value += increment;
    set_double( name, value );
}

double RunStats::get_double( std::string name )
{
    std::map<std::string, double>::iterator iter = double_stats.find( name );
    if ( iter == double_stats.end() ) { return UNINITIALIZED_DOUBLE; }
    return iter->second;
}

bool RunStats::get_double( std::string name, double& value )
{
    std::map<std::string, double>::iterator iter = double_stats.find( name );
    if ( iter == double_stats.end() )
    {
        return false;
    }
    value = iter->second;
    return true;     
}

void RunStats::update_min_double( std::string name, double value )
{
    double current_min;
    bool exists = get_double( name, current_min );
    if ( !exists ) { current_min = value; }
    double_stats[name] = std::min( value, current_min );
}

void RunStats::update_max_double( std::string name, double value )
{
    double current_max;
    bool exists = get_double( name, current_max );
    if ( !exists ) { current_max = value; }
    double_stats[name] = std::max( value, current_max );
}

// ------------------------------------------------------------------

void RunStats::add_per_frame_int( std::string name, int frame, int64_t value )
{
    std::vector<PerFrameInt>& sequence = per_frame_int_stats[name];
    sequence.push_back( PerFrameInt(frame,value) );
}

bool RunStats::get_per_frame_ints( std::string name, std::vector<PerFrameInt>& sequence )
{
    std::map<std::string, std::vector<PerFrameInt> >::iterator iter = per_frame_int_stats.find( name );
    if ( iter == per_frame_int_stats.end() )
    {
        return false;
    }
    sequence = iter->second;
    return true;  
}

// ------------------------------------------------------------------

void RunStats::add_per_frame_double( std::string name, int frame, double value )
{
    std::vector<PerFrameDouble>& sequence = per_frame_double_stats[name];
    sequence.push_back( PerFrameDouble(frame,value) );
}

bool RunStats::get_per_frame_doubles( std::string name, std::vector<PerFrameDouble>& sequence )
{
    std::map<std::string, std::vector<PerFrameDouble> >::iterator iter = per_frame_double_stats.find( name );
    if ( iter == per_frame_double_stats.end() )
    {
        return false;
    }
    sequence = iter->second;
    return true;  
}

// ------------------------------------------------------------------

void RunStats::write_to_file( const char* filename )
{
    std::ofstream file( filename );
    
    // ----------
    if ( !int_stats.empty() )
    {
        file << "int_stats: " << std::endl << "----------" << std::endl;
        std::map<std::string, int64_t>::iterator int_iterator = int_stats.begin();
        for ( ; int_iterator != int_stats.end(); ++int_iterator )
        {
            file << int_iterator->first << ": " << int_iterator->second << std::endl;
        }   
        file << std::endl;
    }
    
    // ----------
    
    if ( !double_stats.empty() )
    {
        file << "double_stats: " << std::endl << "----------" << std::endl;
        std::map<std::string, double>::iterator double_iterator = double_stats.begin();
        for ( ; double_iterator != double_stats.end(); ++double_iterator )
        {
            file << double_iterator->first << ": " << double_iterator->second << std::endl;
        }
        file << std::endl;
    }
    
    // ----------
    
    if ( !per_frame_int_stats.empty() )
    {
        file << "per_frame_int_stats: " << std::endl << "----------" << std::endl;
        std::map<std::string, std::vector<PerFrameInt> >::iterator pfi_iter = per_frame_int_stats.begin();
        for ( ; pfi_iter != per_frame_int_stats.end(); ++pfi_iter )
        {
            file << pfi_iter->first << ": " << std::endl;
            std::vector<PerFrameInt>& sequence = pfi_iter->second;
            for ( unsigned int i = 0; i < sequence.size(); ++i )
            {
                file << sequence[i].first << " " << sequence[i].second << std::endl;
            }
        }
        file << std::endl;
    }   
    
    // ----------
    
    if ( !per_frame_double_stats.empty() )
    {
        file << "per_frame_double_stats: " << std::endl << "----------" << std::endl;
        std::map<std::string, std::vector<PerFrameDouble> >::iterator pfd_iter = per_frame_double_stats.begin();
        for ( ; pfd_iter != per_frame_double_stats.end(); ++pfd_iter )
        {
            file << pfd_iter->first << ": " << std::endl;
            std::vector<PerFrameDouble>& sequence = pfd_iter->second;
            for ( unsigned int i = 0; i < sequence.size(); ++i )
            {
                file << sequence[i].first << " " << sequence[i].second << std::endl;
            }
        }
        file << std::endl;      
    }
    
}

// ------------------------------------------------------------------

void RunStats::clear()
{
    int_stats.clear();
    double_stats.clear();
    per_frame_int_stats.clear();
    per_frame_double_stats.clear();
}

}


