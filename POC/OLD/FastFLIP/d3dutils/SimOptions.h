//
//  SimOptions.h
//
//  Christopher Batty, Fang Da 2014
//
//

#ifndef __SimOptions__
#define __SimOptions__

#include <iostream>
#include <string>
#include <vector>
#include <map>

class Options
{
public:
    enum Type
    {
        STRING,
        INTEGER,
        DOUBLE,
        BOOLEAN,
        
        TYPE_COUNT
    };
    
public:
    // option data: key, value, value type
    static void addStringOption(const std::string & key, const std::string & default_value);
    static void addIntegerOption(const std::string & key, int defaut_value);
    static void addDoubleOption(const std::string & key, double default_value);
    static void addBooleanOption(const std::string & key, bool default_value);
    
    static bool parseOptionFile(const std::string & file, const std::vector<std::pair<std::string, std::string> > & option_overrides, bool verbose = false);
    static void outputOptionValues(std::ostream & os);
    
    static const std::string & strValue(const std::string & key);
    static int                 intValue(const std::string & key);
    static double              doubleValue(const std::string & key);
    static bool                boolValue(const std::string & key);
    
protected:
    class Option
    {
    public:
        std::string key;
        Type type;
        
        std::string str_value;
        int         int_value;
        double      double_value;
        bool        bool_value;
    };
    
    static std::map<std::string, Option> s_options;
    
};

#endif
