#ifndef __TREE_H__
#define __TREE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cmath>

enum nodeType {
    UNDEFINE = 0,     //数字
    NUMBER,     //数字
    FOUROPERATIONS,       //+ - * / %
    ZENVAR,
    FUNC,
    PLACEHOLDER,
    UNARY_FUNC,      //sin cos ref ...
    FUNC_ARG
};

enum operatorVals {
    //四则运算 FOUROPERATIONS
    PLUS = 0,
    MINUS,
    MUL,
    DIV,
    FUNC,
    //一元函数 UNARY_FUNC
    SIN,
    SINH,
    COS,
    COSH,
    ABS,
    DEFAULT_OPVAL,
};

struct node {
    enum operatorVals opVal;
    enum nodeType type;
    float value = 0;  //如果是number

    bool isParenthesisNode = false;

    struct node* left = nullptr;
    struct node* right = nullptr;
    struct node* parent = nullptr;
};

char* getOperatorString(operatorVals op);

struct node* newNode(nodeType type, operatorVals op, struct node* left, struct node* right);
struct node* newNumberNode(float value);

void print_syntax_tree(struct node* root, int depth);
void free_syntax_tree(struct node* root);
float calc_syntax_tree(struct node* root);

bool checkparentheses(std::string& exp, int& addleft, int& addright);

void preOrderVec(struct node* root, std::vector<struct node*>& tmplist);
std::string currFuncName(struct node* root, int rightParenthesesAdded);
#endif