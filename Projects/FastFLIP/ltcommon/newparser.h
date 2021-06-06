#ifndef NEWPARSER_H
#define NEWPARSER_H

#include <iostream>
#include <map>
#include <string>
#include <array1.h>
#include <vec.h>

struct ParseTree
{
    ParseTree() :
    branches(),
    numbers(),
    strings(),
    vectors()
    {}
    
    std::map<std::string, ParseTree> branches;
    std::map<std::string, double> numbers;
    std::map<std::string, std::string> strings;
    std::map<std::string, LosTopos::Array1d> vectors;
    
    const ParseTree* get_branch(const std::string& name) const;
    bool get_number(const std::string& name, double& result) const;
    bool get_int(const std::string& name, int& result) const;
    bool get_string(const std::string& name, std::string& result) const;
    const LosTopos::Array1d* get_vector(const std::string& name) const;
    bool get_vec2d(const std::string& name, LosTopos::Vec2d& v) const;
    bool get_vec3d(const std::string& name, LosTopos::Vec3d& v) const;
    
    bool remove_first_matching_branch( const std::string& name );
    
};

std::ostream& operator<<(std::ostream& out, const ParseTree& tree);

// return true if no errors occur
bool parse_stream(std::istream& input, ParseTree& tree);

#endif

