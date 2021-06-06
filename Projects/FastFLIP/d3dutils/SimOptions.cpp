//
//  SimOptions.cpp
//
//  Christopher Batty, Fang Da 2014
//
//
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <sstream>
#include "SimOptions.h"

std::map<std::string, Options::Option> Options::s_options;

void Options::addStringOption(const std::string & key, const std::string & default_value)
{
    assert(s_options.find(key) == s_options.end()); // verify this option doesn't already exit
    
    Option o;
    o.key = key;
    o.type = STRING;
    o.str_value = default_value;
    
    s_options[key] = o;
}

void Options::addIntegerOption(const std::string & key, int default_value)
{
    assert(s_options.find(key) == s_options.end()); // verify this option doesn't already exit
    
    Option o;
    o.key = key;
    o.type = INTEGER;
    o.int_value = default_value;
    
    s_options[key] = o;
}

void Options::addDoubleOption(const std::string & key, double default_value)
{
    assert(s_options.find(key) == s_options.end()); // verify this option doesn't already exit
    
    Option o;
    o.key = key;
    o.type = DOUBLE;
    o.double_value = default_value;
    
    s_options[key] = o;
}

void Options::addBooleanOption(const std::string & key, bool default_value)
{
    assert(s_options.find(key) == s_options.end()); // verify this option doesn't already exit
    
    Option o;
    o.key = key;
    o.type = BOOLEAN;
    o.bool_value = default_value;
    
    s_options[key] = o;
}


bool Options::parseOptionFile(const std::string & file, const std::vector<std::pair<std::string, std::string> > & option_overrides, bool verbose)
{
    // load options from file
    std::ifstream fin(file);
    if (!fin.is_open())
    {
        std::cout << "Unable to open options file " << file << "." << std::endl;
        assert(!"Unable to open options file");
    }
    
    std::string line;
    while (!fin.eof())
    {
        std::getline(fin, line);
        std::stringstream ss(line);
        
        std::string key;
        ss >> key;
        if (key == "#" || key == "" || ss.eof())    // skip comment lines and empty lines
            continue;

        std::map<std::string, Option>::iterator i = s_options.find(key);
        if (i == s_options.end())
        {
            std::cout << "Unrecognized option: " << key << " in option file " << file << "." << std::endl;
            assert(!"Unrecognized option in file");
        }
        
        switch (i->second.type)
        {
            case STRING:
                ss >> i->second.str_value;
                break;
            case INTEGER:
                ss >> i->second.int_value;
                break;
            case DOUBLE:
                ss >> i->second.double_value;
                break;
            case BOOLEAN:
                ss >> i->second.bool_value;
                break;
            default:
                assert(!"Unexpected option type");
                break;
        }
    }
    
    fin.close();
    
    // apply overrides from the command line
    for (size_t i = 0; i < option_overrides.size(); i++)
    {
        std::stringstream ss(option_overrides[i].second);

        std::map<std::string, Option>::iterator oi = s_options.find(option_overrides[i].first);
        if (oi == s_options.end())
        {
            std::cout << "Unrecognized option: " << option_overrides[i].first << " in command line override." << std::endl;
            assert(!"Unrecognized option in command line override");
        }
        
        switch (oi->second.type)
        {
            case STRING:
                ss >> oi->second.str_value;
                break;
            case INTEGER:
                ss >> oi->second.int_value;
                break;
            case DOUBLE:
                ss >> oi->second.double_value;
                break;
            case BOOLEAN:
                ss >> oi->second.bool_value;
                break;
            default:
                assert(!"Unexpected option type");
                break;
        }
    }

    if (verbose)
    {
        for (std::map<std::string, Option>::iterator i = s_options.begin(); i != s_options.end(); i++)
        {
            std::cout << "option " << i->first << " = ";
            switch (i->second.type)
            {
                case STRING:
                    std::cout << i->second.str_value;
                    break;
                case INTEGER:
                    std::cout << i->second.int_value;
                    break;
                case DOUBLE:
                    std::cout << i->second.double_value;
                    break;
                case BOOLEAN:
                    std::cout << i->second.bool_value;
                    break;
                default:
                    assert(!"Unexpected option type");
                    break;
            }
            std::cout << std::endl;
        }
    }
    
    return false;
}

void Options::outputOptionValues(std::ostream & os)
{
    size_t maxlen = 0;
    for (std::map<std::string, Option>::iterator i = s_options.begin(); i != s_options.end(); i++)
        maxlen = std::max(maxlen, i->first.size());
    
    for (std::map<std::string, Option>::iterator i = s_options.begin(); i != s_options.end(); i++)
    {
        os << i->first << std::string(maxlen + 1 - i->first.size(), ' ');
        switch (i->second.type)
        {
            case STRING:
                os << i->second.str_value;
                break;
            case INTEGER:
                os << i->second.int_value;
                break;
            case DOUBLE:
                os << i->second.double_value;
                break;
            case BOOLEAN:
                os << i->second.bool_value;
                break;
            default:
                assert(!"Unexpected option type");
                break;
        }
        os << std::endl;
    }
}

const std::string & Options::strValue(const std::string & key)
{
    assert(s_options.find(key) != s_options.end()); // verify this option exists
    assert(s_options[key].type == STRING);          // verify this option has the correct type
    return s_options[key].str_value;
}

int Options::intValue(const std::string & key)
{
    assert(s_options.find(key) != s_options.end()); // verify this option exists
    assert(s_options[key].type == INTEGER);         // verify this option has the correct type
    return s_options[key].int_value;
}

double Options::doubleValue(const std::string & key)
{
    assert(s_options.find(key) != s_options.end()); // verify this option exists
    assert(s_options[key].type == DOUBLE);          // verify this option has the correct type
    return s_options[key].double_value;
}

bool Options::boolValue(const std::string & key)
{
    assert(s_options.find(key) != s_options.end()); // verify this option exists
    assert(s_options[key].type == BOOLEAN);         // verify this option has the correct type
    return s_options[key].bool_value;
}

