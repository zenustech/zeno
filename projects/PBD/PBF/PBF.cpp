#include <vector>
#include <array>
#include <zeno/zeno.h>
#include <map>
#include <zeno/types/PrimitiveObject.h>
#include "../Utils/myPrint.h"

using namespace zeno;

struct PBF : INode{
//physical params
public:    
    int numSubsteps = 5;
    float dt= 1.0 / 20.0;
    float pRadius = 3.0;
    vec3f bounds{40.0, 40.0, 40.0};
    vec3f g{0, -10.0, 0};

    float mass = 1.0;
    float rho0 = 1.0;
    float h = 1.1;
    float neighborSearchRadius = h * 1.05;

private:
    void preSolve();
    void solve();
    void postSolve();

    void computeLambda();
    void computeDpos();
    void neighborSearch();

//Data preparing
    //data for physical fields
    void readPoints();
    void initData();
    int numParticles;
    std::vector<vec3f> pos;
    std::vector<vec3f> oldPos;
    std::vector<vec3f> vel;
    std::vector<float> lambda;
    std::vector<vec3f> dpos;

    std::shared_ptr<zeno::PrimitiveObject> prim;

    //helpers
    void boundaryHandling(vec3f &p);
    inline vec3i getCellXYZ(const vec3f& p);
    inline int getCellID(const vec3f& p);
    inline int getCellHash(int i, int j, int k);
    inline bool isInBound(const vec3i& cell);
    inline int cellXYZ2ID(const vec3i& xyz);
    inline vec3i cellID2XYZ(int i);
    inline vec3f kernelSpikyGradient(const vec3f& r, float h);
    inline float kernelPoly6(float dist, float h);
    inline float computeScorr(const vec3f& distVec);

    //data for cells
    std::vector<int> numCellXYZ;
    int numCell;
    float dx; //cell size
    float dxInv; 
    vec3i bound;
    void initCellData()
    {
        // dx = 2.51; //default value for test
        // vec3i bound {40,40,40};
        dx = get_input<zeno::NumericObject>("dx")->get<float>();
        bound = get_input<zeno::NumericObject>("bound")->get<vec3i>();

        dxInv = 1.0/dx;
        int numX = int(bound[0] / dx) + 1;
        int numY = int(bound[1] / dx) + 1;
        int numZ = int(bound[2] / dx) + 1;
        numCellXYZ.resize(3);
        numCellXYZ[0] = numX;
        numCellXYZ[1] = numY;
        numCellXYZ[2] = numZ;
        numCell = numX * numY * numZ;
    }
    struct Cell
    {
        int x,y,z;
        std::vector<int> parInCell; 
    };
    std::map<int, Cell>  cell;

    //neighborList
    std::vector<std::vector<int>> neighborList;

public:
    virtual void apply() override{
        prim = get_input<PrimitiveObject>("prim");
        // numSubsteps = get_input<zeno::NumericObject>("numSubsteps")->get<int>();

        static bool firstTime = true;
        if(firstTime == true)
        {
            firstTime = false;
            initCellData();

            // move pos to local
            numParticles = prim->verts.size();
            pos = std::move(prim->verts);
            initData();  
            echo(numParticles);
        }

        preSolve();
        for (size_t i = 0; i < numSubsteps; i++)
            solve(); 
        postSolve();  

        // echoVec(pos[1]);
        //move back
        // prim->verts = std::move(pos);
        prim->verts.resize(pos.size());
        for (size_t i = 0; i < pos.size(); i++)
            prim->verts[i] = pos[i]/10.0;//scale to show

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(PBF, {   
                    {
                        {"PrimitiveObject", "prim"},
                        {"float", "dx", "2.51"},
                        {"vec3i", "bound", "40, 40, 40"},
                        // {"int", "numSubsteps", "5"}
                    },
                    {   {"PrimitiveObject", "outPrim"} },
                    {},
                    {"PBD"},
                });

void PBF::initData()
{     
    //prepare physical field data
    oldPos.resize(numParticles);
    vel.resize(numParticles);

    lambda.resize(numParticles);
    dpos.resize(numParticles);

    //prepare cell data
    // cell.resize(numCell);
    for (size_t i = 0; i < numCell; i++)
    {
        // //to calculate the x y z coord of cell
        vec3i xyz = cellID2XYZ(i);
        int hash = getCellHash(xyz[0],xyz[1],xyz[2]);

        cell[hash].x = xyz[0];
        cell[hash].y = xyz[1];
        cell[hash].z = xyz[2];
        cell[hash].parInCell.reserve(10); //pre-allocate memory to speed up
    }
    echo(cell.size());
    
    //prepare neighbor list 
    neighborList.resize(numParticles);
    for (size_t i = 0; i < numParticles; i++)
        neighborList[i].reserve(10);
}

void PBF::preSolve()
{
    for (int i = 0; i < numParticles; i++)
        oldPos[i] = pos[i];

    //update the pos
    for (int i = 0; i < numParticles; i++)
    {
        vec3f tempVel = vel[i];
        tempVel += g * dt;
        pos[i] += tempVel * dt;
        boundaryHandling(pos[i]);
    }

    neighborSearch();//grid-baed neighborSearch for now
}

//This neighborSearch algorithm uses grid-based searching,
//which is simple but can be improved
void PBF::neighborSearch()
{
    //clear parInCell and neighborList
    for (size_t i = 0; i < numCell; i++)
        cell[i].parInCell.clear();
    for (size_t i = 0; i < numParticles; i++)
        neighborList[i].clear();
    
    //update the parInCell list
    vec3i cellXYZ;
    int cellID;
    for (size_t i = 0; i < numParticles; i++) // i is the particle ID
    {
        cellID = getCellID(pos[i]);
        // int hash = getCellHash(cellXYZ[0], cellXYZ[1], cellXYZ[2]);
        cell[cellID].parInCell.push_back(i);
    }

    //update the neighborList
    for (size_t i = 0; i < numParticles; i++)
    {
        cellXYZ = getCellXYZ(pos[i]);
        for (int off_x = -1; off_x < 2; off_x++)
            for (int off_y = -1; off_y < 2; off_y++)
                for (int off_z = -1; off_z < 2; off_z++)
                {
                    vec3i off{off_x, off_y, off_z};
                    vec3i toCheckXYZ = cellXYZ + off;
                    int toCheck = cellXYZ2ID(toCheckXYZ);
                    if (isInBound(toCheckXYZ))
                    {
                        Cell theCell = cell[toCheck];
                        std::vector<int> parInTheCell = theCell.parInCell;

                        for (int j = 0; j < parInTheCell.size(); j++)
                        {
                            int p = parInTheCell[j];
                            if(p!=i && length(pos[i] - pos[p]) < neighborSearchRadius)
                            {
                                neighborList[i].push_back(p);
                            }
                        }
                    }
                }  
    }
}

//helpers for neighborSearch
inline bool PBF::isInBound(const vec3i& cellXYZ)
{
    return cellXYZ[0] >= 0 && cellXYZ[0] < numCellXYZ[0] &&
           cellXYZ[1] >= 0 && cellXYZ[1] < numCellXYZ[1] &&
           cellXYZ[2] >= 0 && cellXYZ[2] < numCellXYZ[2];
}

inline int PBF::getCellID(const vec3f& p)
{
    vec3i xyz = p*dxInv;
    int numPerRow = numCellXYZ[0];
    int numPerFloor = numCellXYZ[0] * numCellXYZ[1];
    int res = numPerFloor * xyz[2] + numPerRow * xyz[1] + xyz[0];
    return res;
}

inline int PBF::cellXYZ2ID(const vec3i& xyz)
{
    int numPerRow = numCellXYZ[0];
    int numPerFloor = numCellXYZ[0] * numCellXYZ[1];
    int res = numPerFloor * xyz[2] + numPerRow * xyz[1] + xyz[0];
    return res;
}

inline vec3i PBF::cellID2XYZ(int i)
{
    //to calculate the x y z coord of cell
    int numPerRow = numCellXYZ[0];
    int numPerFloor = numCellXYZ[0] * numCellXYZ[1];

    int floor = (i / numPerFloor);
    int row = (i % numPerFloor) / numPerRow;
    int col = (i % numPerFloor) % numPerRow; 
    
    vec3i res{col,row,floor};
    return res;
}

inline vec3i PBF::getCellXYZ(const vec3f& p)
{
    vec3i res{p*dxInv};
    return res;
}

inline int PBF::getCellHash(int i, int j, int k)
{
    int res = ( (73856093 * i) ^ 
                (19349663 * j) ^ 
                (83492791 * k) ) 
                % (2 * numParticles);
    return res;
}

void PBF::boundaryHandling(vec3f & p)
{
    float worldScale = 20.0; //scale from simulation space to real world space.
    // this is to prevent the kernel from being divergent.
    float bmin = pRadius/worldScale;
    vec3f bmax = bounds - pRadius/worldScale;

    for (size_t dim = 0; dim < 3; dim++)
    {
        float r = ((float) rand() / (RAND_MAX));
        if (p[dim] <= bmin)
            p[dim] = bmin + 1e-5 * r;
        else if (p[dim]>= bmax[dim])
            p[dim] = bmax[dim] - 1e-5 * r;
    }
}

void PBF::solve()
{
    computeLambda();

    computeDpos();

    //apply the dpos to the pos
    for (size_t i = 0; i < numParticles; i++)
        pos[i] += dpos[i];
}

void PBF::computeLambda()
{
    lambda.clear();
    lambda.resize(numParticles);
    for (size_t i = 0; i < numParticles; i++)
    {
        vec3f gradI{0.0, 0.0, 0.0};
        float sumSqr = 0.0;
        float densityCons = 0.0;

        for (size_t j = 0; j < neighborList[i].size(); j++)
        {
            int pj = neighborList[i][j];
            vec3f distVec = pos[i] - pos[pj];
            vec3f gradJ = kernelSpikyGradient(distVec, h);
            gradI += gradJ;
            sumSqr += dot(gradJ, gradJ);
            densityCons += kernelPoly6(length(distVec), h);
        }
        densityCons = (mass * densityCons / rho0) - 1.0;

        //compute lambda
        sumSqr += dot(gradI, gradI);
        float lambdaEpsilon = 100.0; // to prevent the singularity
        lambda[i] = (-densityCons) / (sumSqr + lambdaEpsilon);
    }
}

void PBF::computeDpos()
{
    dpos.clear();
    dpos.resize(numParticles);
    for (size_t i = 0; i < numParticles; i++)
    {
        vec3f dposI{0.0, 0.0, 0.0};
        for (size_t j = 0; j < neighborList[i].size(); j++)
        {
            int pj = neighborList[i][j];
            vec3f distVec = pos[i] - pos[pj];

            float sCorr = computeScorr(distVec);
            dposI += (lambda[i] + lambda[pj] + sCorr) * kernelSpikyGradient(distVec, h);
        }
        dposI /= rho0;
        dpos[i] = dposI;
    }
}

//helper for computeDpos()
inline float PBF::computeScorr(const vec3f& distVec)
{
    float coeffDq = 0.3;
    float coeffK = 0.001;

    float x = kernelPoly6(length(distVec), h) / kernelPoly6(coeffDq * h, h);
    x = x * x;
    x = x * x;
    return (-coeffK) * x;
}

//SPH kernel function
inline float PBF::kernelPoly6(float dist, float h)
{
    float coeff = 315.0 / 64.0 / 3.14159265358979323846;
    float res = 0.0;
    if(dist > 0 && dist < h)
    {
        float x = (h * h - dist * dist) / (h * h * h);
        res = coeff * x * x * x;
    }
    return res;
}

//SPH kernel gradient
inline vec3f PBF::kernelSpikyGradient(const vec3f& r, float h)
{
    float coeff = -45.0 / 3.14159265358979323846;
    vec3f res{0.0, 0.0, 0.0};
    float dist = length(r);
    if (dist > 0 && dist < h)
    {
        float x = (h - dist) / (h * h * h);
        float factor = coeff * x * x;
        res = r * factor / dist;
    }
    return res;
}


void PBF::postSolve()
{
    for (size_t i = 0; i < numParticles; i++)
        boundaryHandling(pos[i]);
    for (size_t i = 0; i < numParticles; i++)
        vel[i] = (pos[i] - oldPos[i]) / dt;
}