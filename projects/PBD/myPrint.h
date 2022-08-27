
#pragma once
#include <fstream>
#include <iostream>
#include <string>

template<typename T>
void print(T contents, int maxTimes=20, bool toFile=false, std::string msg="")
{
    static int times = 0;
    times++;
    if (times>maxTimes)
        return;
    
    if(!toFile)
    {
        std::cout<<msg;
        for(auto x:contents)
            std::cout<<x<<"\t";
        std::cout<<"\n";
        return;
    }
    else if(toFile)
    {
        static std::ofstream fout;
        fout.open("debugOutput.txt", std::ios::app);
        fout<<msg;
        for(const auto& x:contents)
            fout<<x<<"\t";
        fout<<"\n";
        fout.close();
        return;
    }
}

template<typename T>
void printToFile(T content, std::string fileName)
{
    std::ofstream f;
    f.open(fileName);
    for(const auto& x:content)
    {
        for(const auto& xx:x)
            f<<xx<<"\t";
        f<<"\n";
    }
    f.close();
}

#define echo(content) {std::cout<<(#content)<<": "<<content<<std::endl;}