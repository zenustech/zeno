#include "help.h"

std::vector<int> ArrangeCore(std::vector<int> a)
{
    std::vector<int> b;
    b.resize(a.size());
    for (size_t i = 0; i < b.size(); i++)
    {
        b[i] = i;
    }

    for (size_t i = 0; i < a.size(); i++)
    {
        for (size_t j = 0; j < a.size(); j++)
            if (i != j && a[i] == a[j])
            {
                b[i] = j;
                b[j] = i;
            }
    }

    int s = 0;
    for (size_t j = 0; j < a.size(); j++)
    {
        if (b[j] == (int)j)
            s = j;
    }

    std::vector<int> c;
    while (c.size() < a.size() / 2 + 1)
    {
        c.push_back(s);
        int temp;
        if (s % 2 == 0)
            temp = s + 1;
        else
            temp = s - 1;
        s = b[temp];
    }
    for (size_t i = 0; i < c.size(); i++)
        c[i] = a[c[i]];
    // std::cout << std::endl;
    return c;
}

void WriteParticle(std::vector<Eigen::Vector3d> &points, int num)
{
    std::ofstream outfile;
    // std::string str = "E:/output/meshing/test/20/particle.obj";

    std::string buffer = "E:/output/meshing/test/20/particle";
    std::string over = ".obj";
    std::stringstream ss;
    std::string str;

    ss << num;
    ss >> str;
    str += over;
    str = buffer + str;

    outfile.open(str);

    for (size_t i = 0; i < points.size(); i++)
    {
        outfile << "v " << points[i].x() << " " << points[i].y() << " "
                << points[i].z() << std::endl;
        // std::cout<< points[i].x() << " " << points[i].y() << " " << points[i].z() <<
        // std::endl;
    }
    outfile.close();
}

Eigen::Vector3i sort(int x, int y, int z)
{
    int t;
    if (x > y)
    {
        t = x;
        x = y;
        y = t;
    }
    if (y > z)
    {
        t = y;
        y = z;
        z = t;
    }
    if (x > y)
    {
        t = x;
        x = y;
        y = t;
    }
    return Eigen::Vector3i(x, y, z);
}

Eigen::Vector2i sort(int x, int y)
{
    if (x < y)
        return Eigen::Vector2i(x, y);
    else
        return Eigen::Vector2i(y, x);
}