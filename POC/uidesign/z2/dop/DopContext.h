#pragma once


#include <z2/ztd/stdafx.h>
#include <z2/ztd/functional.h>


namespace z2::dop {


struct DopNode;


struct DopDepsgraph {
    std::set<DopNode *> nodes;

    bool insert_node(DopNode *node);
};


//struct DopNode;


//using DopPromise = ztd::promise<std::any>;
//struct DopContext {
    //std::set<std::string> visited;

    //bool contains(std::string const &key) const {
        //return visited.contains(key);
    //}

    //void insert(std::string const &key) {
        //visited.insert(key);
    //}
//};


//struct DopContext {
    //std::vector<DopNode *> tasks;
    //std::set<DopNode *> visited;

    //struct Ticket {
        //DopContext *ctx;
        //DopNode *node;

        //void wait() const;
    //};

    //Ticket enqueue(DopNode *node) {
        //tasks.push_back(node);
        //return {this, node};
    //}
//};


}  // namespace z2::dop
