#include <zeno/formula/syntax_tree.h>

char* getOperatorString(operatorVals op)
{
    switch (op) {
    case PLUS:
        return "plus";
    case MINUS:
        return "minus";
    case MUL:
        return "mul";
    case DIV:
        return "div";
    case SIN:
        return "sin";
    case SINH:
        return "sinh";
    case COS:
        return "cos";
    case COSH:
        return "cosh";
    case ABS:
        return "abs";
    default:
        return "";
    }
}

struct node* newNode(nodeType type, operatorVals op, struct node* left, struct node* right) {
    struct node* n = (struct node*)malloc(sizeof(struct node));
    if (!n)
    {
        exit(0);
    }
    n->type = type;
    n->opVal = op;
    n->value = 0;
    n->left = left;
    n->right = right;
    n->parent = nullptr;
    n->isParenthesisNode = false;
    if (left) {
        left->parent = n;
    }
    if (right) {
        right->parent = n;
    }
    return n;
}

struct node* newNumberNode(float value) {
    struct node* n = (struct node*)malloc(sizeof(struct node));
    if (!n)
    {
        exit(0);
    }
    n->type = NUMBER;
    n->value = value;
    n->parent = nullptr;
    n->left = nullptr;
    n->right = nullptr;
    return n;
}

void print_syntax_tree(struct node* root, int depth) {
    const auto& printVal = [](struct node* root, char* prefix) {
        if (root->type == NUMBER)
            printf("%s: %f\n", prefix, root->value);
        else
            printf("%s: %s isParen:%d\n", prefix, getOperatorString(root->opVal), root->isParenthesisNode);
    };
    if (root) {
        for (int i = 0; i < depth; ++i) {
            printf("|  ");
        }
        if (root->parent)
        {
            if (root->parent->left && root->parent->left == root)
                printVal(root, "left");
            else if (root->parent->right && root->parent->right == root)
                printVal(root, "right");
        }
        else {
            printVal(root, "root");
        }
        print_syntax_tree(root->left, depth + 1);
        print_syntax_tree(root->right, depth + 1);
    }
}

void free_syntax_tree(struct node* root)
{
    if (root) {
        free_syntax_tree(root->left);
        free_syntax_tree(root->right);
        //if (root->type == NUMBER)
        //    printf("free number: %f\n", root->value);
        //else
        //    printf("free operators: %s\n", getOperatorString(root->opVal));
        free(root);
        root = nullptr;
        return;
    }
}

float calc_syntax_tree(struct node* root)
{
    if (root) {
        if (root->type == NUMBER)
        {
            return root->value;
        }
        float leftRes = calc_syntax_tree(root->left);
        float rightRes = calc_syntax_tree(root->right);
        switch (root->opVal)
        {
        case PLUS:
            return leftRes + rightRes;
        case MINUS:
            return leftRes - rightRes;
        case MUL:
            return leftRes * rightRes;
        case DIV:
            return leftRes / rightRes;
        case SIN:
            return std::sin(leftRes);
        case SINH:
            return std::sinh(leftRes);
        case COS:
            return std::cos(leftRes);
        case COSH:
            return std::cosh(leftRes);
        case ABS:
            return std::abs(leftRes);
        default:
            return 0;
        }
    }
    return 0;
}

bool checkparentheses(std::string& exp, int& addleft, int& addright)
{
    int left = 0;
    int leftNeed = 0;
    for (auto& i : exp)
    {
        if (i == '(')
            left++;
        else if (i == ')')
            left--;
        if (left < 0)
        {
            leftNeed++;
            left = 0;
        }
    }
    if (left != 0 || leftNeed != 0)
    {
        exp = exp + std::string(left, ')');
        exp = std::string(leftNeed, '(') + exp;
        addleft = leftNeed;
        addright = left;
        return false;
    }
    else {
        addleft = 0;
        addright = 0;
        return true;
    }
}

void preOrderVec(struct node* root, std::vector<struct node*>& tmplist)
{
    if (root)
    {
        tmplist.push_back(root);
        preOrderVec(root->left, tmplist);
        preOrderVec(root->right, tmplist);
    }
}

std::string currFuncName(struct node* root, int rightParenthesesAdded)
{
    std::vector<struct node*> preorderVec;
    preOrderVec(root, preorderVec);
    if (preorderVec.size() != 0)
    {
        auto last = preorderVec.back();
        std::vector<struct node*> allParenthesesAncestors;
        while (last)
        {
            if (last->type == UNARY_FUNC || last->isParenthesisNode) {
                allParenthesesAncestors.push_back(last);
            }
            last = last->parent;
        }
        if (allParenthesesAncestors.size() > rightParenthesesAdded - 1)
        {
            auto currentParentheseNode = allParenthesesAncestors[allParenthesesAncestors.size() - rightParenthesesAdded];
            while (currentParentheseNode)
            {
                if (currentParentheseNode->type == UNARY_FUNC) {
                    std::string candidate = getOperatorString(currentParentheseNode->opVal);
                    if (candidate != "") {
                        return candidate;
                    }
                }
                currentParentheseNode = currentParentheseNode->parent;
            }
        }
    }
    return "";
}
