
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

#define echo(content) {std::cout<<(#content)<<": "<<content<<std::endl;}