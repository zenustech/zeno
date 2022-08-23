
#pragma once
#include <fstream>
#include <iostream>

template<typename T>
void print(T contents, int maxTimes=20, bool toFile=false)
{
    static int times = 0;
    times++;
    if (times>maxTimes)
        return;
    
    if(!toFile)
    {
        for(auto x:contents)
            std::cout<<x<<"\t";
        std::cout<<"\n";
        return;
    }
    else if(toFile)
    {
        static std::ofstream fout;
        fout.open("debugOutput.txt", std::ios::app);
        for(const auto& x:contents)
            fout<<x<<"\t";
        fout<<"\n";
        fout.close();
        return;
    }
}

#define echo(content) {std::cout<<(#content)<<": "<<content<<std::endl;}