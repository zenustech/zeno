#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip> 
#include <type_traits> 

template<typename T>
void printVec(const T& contents, int precision = 8)
{
    for(auto x:contents)
        std::cout<<std::fixed <<std::setprecision(precision)<<x<<"\t";
    std::cout<<"\n";
    return;
}

template<typename T>
void printScalarFieldToScreen(const T& field,int precision=8)
{
    std::cout<<"Printing a scalar field with "<<field.size()<<" elements...\n";
    for(const auto& x:field)
        std::cout<<std::fixed <<std::setprecision(precision)<<x<<"\n";
    return;
}

template<typename T>
void printVectorFieldToScreen(const T& field, int precision=8)
{
    std::cout<<"Printing a vector field with "<<field.size()<<" elements...\n";
    for(const auto& x:field)
        printVec(x);
    return;
}


template<typename T>
void printVectorField(std::string fileName, T content,  size_t precision=8)
{
    std::ofstream f;
    f.open(fileName);
    for(const auto& x:content)
    {
        for(const auto& xx:x)
            f<<std::fixed <<std::setprecision(precision)<<xx<<"\t";
        f<<"\n";
    } 
    f.close();
}

template<typename T>
void printScalarField(std::string fileName, T content,  size_t precision=8)
{
    std::ofstream f;
    f.open(fileName);
    for(const auto& x:content)
    {
        f<<std::fixed <<std::setprecision(precision)<<x<<"\n";
    } 
    f.close();
}


#define echo(content) {std::cout<<(#content)<<": "<<content<<std::endl;}
#define echoVec(content) {std::cout<<(#content)<<": ";  printVec(content);}