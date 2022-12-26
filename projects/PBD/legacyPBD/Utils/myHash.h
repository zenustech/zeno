#include<zeno/zeno.h>
namespace zeno
{

inline int makeHash2(const vec2i & a)
{
    return a[0] ^ 2718281828 ^ a[1] ^ 3141592653;
    
}
inline int makeHash3(const vec3i & a)
{
    return a[0] ^ 2718281828 ^ a[1] ^ 3141592653 ^ a[2] ^1618033989;
}

void mapEdges(const vector<vec2i> & edges, std::unordered_map<int,vec2i> &map1)
{
    for(const auto & e:edges)
    {
        int key=makeHash2(e);
        map1.emplace(key,e);
    }
}

void mapTris(const vector<vec3i> & tris, std::unordered_map<int,vec3i> &map1)
{
    for(const auto & t:tris)
    {
        int key=makeHash3(t);
        map1.emplace(key,t);
    }
}

int findEdge(const vec2i &edge1, const std::unordered_map<int,vec2i> &map1)
{
    auto found = map1.find(makeHash2(edge1));
    if (found != map1.end()) 
        return found->first; //return its hash if found
    else 
        return 0; //return 0 if not found;
}

template<typename T>
int printMap(const std::unordered_map<int,T> &map1)
{
    std::cout<<"\nmap is:"<<'\n';
    for(auto &x: map1)
    {
        std::cout<<x.first<<',';
        std::cout<<x.second<<'\t';
    }
    std::cout<<"\n";
}


void edgesTrisMap(
    const vector<vec2i> & edges,
    const vector<vec3i> & tris,
    std::unordered_map<int,vec2i> &map1, 
    std::unordered_map<int,vec3i> &map2)
{
    for(const auto & t:tris)
    {
        vec2i e1{t[0], t[1]};   //三角面的三条边
        vec2i e2{t[1], t[2]};   
        vec2i e3{t[2], t[0]};

        int key1=makeHash2(e1);
        int key2=makeHash2(e2);
        int key3=makeHash2(e3);
        int val=makeHash3(t);
        map1.emplace(key1,val); //map1: edges to tris with hash
        map1.emplace(key2,val);
        map1.emplace(key3,val);

        map2.emplace(val,key1); //map2: tris to edges with hash
        map2.emplace(val,key2);
        map2.emplace(val,key3);
    }
}


} // namespace zeno