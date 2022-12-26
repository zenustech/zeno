#include <cstdlib>
#include <ctime>  
#include <type_traits>


/**
 * @brief 生成一个随机数(浮点数),默认范围[0,1)
 * 
 * @param min 最小范围
 * @param max 最大范围
 * @return float 
 */
float genRnd(float min=0, float max=1)
{
    return (rand()/float(RAND_MAX)) * (max-min) + min;
}

/**
 * @brief 生成一个随机数(整数)，范围0~RAND_MAX
 * 
 * @param min 最小范围
 * @param max 最大范围
 * @return int 
 */
int genRndInt(int min=0, int max=RAND_MAX)
{
    return (rand() % (max-min))+ min;
}


/**
 * @brief 为给定的vector填充随机数
 * 
 * @tparam T 数组元素的类型
 * @param rnd_list 填充随机数的数组
 */
template<typename T>
void genRndList(std::vector<T> &rnd_list)
{
    if(std::is_same_v<T,int>)
    {
        std::cout<<"Generating "<<rnd_list.size()<<" integer random number...\n";
        for (size_t i = 0; i < rnd_list.size(); i++)
            rnd_list[i]= rand();
    }
    else
    {
        std::cout<<"Generating "<<rnd_list.size()<<" float random number...\n";
        for (size_t i = 0; i < rnd_list.size(); i++)
            rnd_list[i]= rand()/float(RAND_MAX);
    }
}


void genRndIntList(std::vector<int> &rnd_list, int min=0, int max=RAND_MAX)
{
    std::cout << "Generating " << rnd_list.size() << " integer random number...\n";
    for (size_t i = 0; i < rnd_list.size(); i++)
        rnd_list[i] = genRndInt(min,max);
}


void genRndFloatList(std::vector<float> &rnd_list, float min=0, float max=1)
{
    std::cout << "Generating " << rnd_list.size() << " float random number...\n";
    for (size_t i = 0; i < rnd_list.size(); i++)
        rnd_list[i] = genRnd(min,max);
}


/**
 * @brief 重置随机数种子
 * 
 */
void reSeed()
{
    srand((unsigned)time(NULL));
}




/**
 * @brief 用随机数填充一个VectorField
 * 
 * @tparam T field中元素的类型
 * @param rnd_list 要填充的field
 */
template<typename T>
void fillRndVectorField(std::vector<T> &rnd_list)
{
    if(std::is_same_v<T,int>)
    {
        std::cout<<"Filling "<<rnd_list.size()<<" int random number to the scalar field...\n";
        for (size_t i = 0; i < rnd_list.size(); i++)
            for(auto& x: rnd_list[i])
                x= rand();
    }
    else
    {
        std::cout<<"Filling "<<rnd_list.size()<<" float random number to the scalar field...\n";
        for (size_t i = 0; i < rnd_list.size(); i++)
            for(auto& x: rnd_list[i])
                x= rand()/float(RAND_MAX);
    }
}

/**
 * @brief 用随机数填充一个ScalarField
 * 
 * @tparam T field中元素的类型
 * @param rnd_list 要填充的field
 */
template<typename T>
void fillRndScalarField(std::vector<T> &rnd_list)
{
    if(std::is_same_v<T,int>)
    {
        std::cout<<"Filling "<<rnd_list.size()<<" int random number to the scalar field...\n";
        for (size_t i = 0; i < rnd_list.size(); i++)
            rnd_list[i]= rand();
    }

    else
    {
        std::cout<<"Filling "<<rnd_list.size()<<" float random number to the scalar field...\n";
        for (size_t i = 0; i < rnd_list.size(); i++)
            rnd_list[i]= rand()/float(RAND_MAX);
    }
}