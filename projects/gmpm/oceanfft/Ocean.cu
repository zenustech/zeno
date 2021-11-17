#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/PrimitiveObject.h>
//-------attached some cuda library for FFT-----------------


//--------------
#include <math.h>
#include <math_constants.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#define MAX_EPSILON 0.10f
#define THRESHOLD   0.15f
#define REFRESH_DELAY     10 //ms



namespace zeno {
//---------------
        //---------------

        int cuda_iDivUp(int a, int b)
        {
            return (a + (b - 1)) / b;
        }
        
        
        // complex math functions
        __device__
        float2 conjugate(float2 arg)
        {
            return make_float2(arg.x, -arg.y);
        }
        
        __device__
        float2 complex_exp(float arg)
        {
            return make_float2(cosf(arg), sinf(arg));
        }
        
        __device__
        float2 complex_add(float2 a, float2 b)
        {
            return make_float2(a.x + b.x, a.y + b.y);
        }
        
        __device__
        float2 complex_mult(float2 ab, float2 cd)
        {
            return make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
        }
        
        __device__
        float2 normalize(float2 k)
        {       
                float2 v = k;
                float len = ::sqrt(v.x*v.x + v.y*v.y)+0.0000001;
                v.x /= len;
                v.y /= len;
                return v;
        }
        
        
        // generate wave heightfield at time t based on initial heightfield and dispersion relationship
        __global__ void generateSpectrumKernel(float2 *h0,
                                               float2 *ht,
                                               float2 *Dx,
                                               float2 *Dz,
                                               unsigned int in_width,
                                               unsigned int out_width,
                                               unsigned int out_height,
                                               float t,
                                               float patchSize)
        {
            unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
            unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
            unsigned int in_index = y*in_width+x;
            unsigned int in_mindex = (out_height - y)*in_width + (out_width - x); // mirrored
            unsigned int out_index = y*out_width+x;
        
            // calculate wave vector
            float2 k;
            k.x = (-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
            k.y = (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);
        
            // calculate dispersion w(k)
            float k_len = sqrtf(k.x*k.x + k.y*k.y);
            float w = sqrtf(9.81f * k_len);
        
            if ((x < out_width) && (y < out_height))
            {
                float2 h0_k = h0[in_index];
                float2 h0_mk = h0[in_mindex];
        
                // output frequency-space complex values
        
                ht[out_index] = complex_add(complex_mult(h0_k, complex_exp(w * t)), complex_mult(conjugate(h0_mk), complex_exp(-w * t)));
        
                float2 nk = normalize(k);
        
                Dx[out_index] = make_float2(ht[out_index].y * nk.x , -ht[out_index].x * nk.x);
                Dz[out_index] = make_float2(ht[out_index].y * nk.y , -ht[out_index].x * nk.y);
        
            }
        }
        
        __global__ void updateDxKernel(float2  *Dx, float2 *Dz, unsigned int width)
        {
            unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
            unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
            unsigned int i = y*width+x;
        
            // cos(pi * (m1 + m2))
            float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;
        
            Dx[i].x = Dx[i].x * sign_correction;
            Dz[i].x = Dz[i].x * sign_correction;
        }
        
        
        // update height map values based on output of FFT
        __global__ void updateHeightmapKernel(float2  *heightMap,
                                              float2 *ht,
                                              unsigned int width)
        {
            unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
            unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
            unsigned int i = y*width+x;
        
            // cos(pi * (m1 + m2))
            float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;
        
            heightMap[i].x = ht[i].x * sign_correction;
        }









struct OceanFFT : zeno::IObject{
    OceanFFT() = default;
    //static constexpr unsigned int spectrumW = meshSize + 4;
    //static constexpr unsigned int spectrumH = meshSize + 1;
    //static constexpr int frameCompare = 4;  
    ~OceanFFT() {
        if (d_h0 != nullptr) {
            cudaFree(d_h0);
            cudaFree(d_ht);
            cudaFree(Dx);
            cudaFree(Dz);
            free(h_h0);
            free(g_hhptr);
            free(g_hDx);
            free(g_hDz);

        }
    }

    // FFT data
    cufftHandle fftPlan;
    float2 *d_h0{nullptr};   // heightfield at time 0
    float2 *h_h0{nullptr};
    float2 *d_ht{nullptr};   // heightfield at time t
    float2 *Dx{nullptr};
    float2 *Dz{nullptr};
    float2 *g_hDx{nullptr};
    float2 *g_hDz{nullptr};
    float2 *g_hhptr{nullptr};


    // simulation parameters
    float L_scale = 1.0;
    float g;              // gravitational constant
    float amplitude;
    float A{1e-4f};              // wave scale factor
    float patchSize;        // patch size
    float windSpeed{100.0f};
    float windDir;
    float dir_depend{0.07f};
    float Kx;
    float Ky;
    float Vdir;
    float V;
    float depth{5000};
    float in_width;
    float out_width;
    float out_height;
    float ocdir;
    float animTime{0.0f};
    float prevTime{0.0f};
    float animationRate{-0.001f};
 
    float *d_heightMap{nullptr};
    unsigned int width;
    unsigned int height;
    bool autoTest;

    float timeScale;
    float timeShift;
    float speed;

    float choppyness;
    int WaveExponent = 8;
    int meshSize;
    int spectrumW;
    int spectrumH;
    int frameCompare;
    int spectrumSize = spectrumW*spectrumH*sizeof(float2);
};


void GenerateSpectrumKernel(float2 *d_h0,
        float2 *d_ht,
        float2 *Dx,
        float2 *Dz,
        unsigned int in_width,
        unsigned int out_width,
        unsigned int out_height,
        float animTime,
        float patchSize)
{
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(out_width, block.x), cuda_iDivUp(out_height, block.y), 1);
    generateSpectrumKernel<<<grid, block>>>(d_h0, d_ht, Dx, Dz, in_width, out_width, out_height, animTime, patchSize);
    cudaDeviceSynchronize();
}
    



void UpdateDxKernel(float2  *Dx,float2 *Dz, unsigned int width, unsigned int height)
{
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    
    updateDxKernel<<<grid, block>>>(Dx, Dz, width);
    cudaDeviceSynchronize();
}


    
    

void UpdateHeightmapKernel(float2  *d_heightMap,
    float2 *d_ht,
    unsigned int width,
    unsigned int height)
{
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);

    updateHeightmapKernel<<<grid, block>>>(d_heightMap, d_ht, width);


    cudaDeviceSynchronize();
}
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//put gaussian into zeno node

float urand()
{
    return rand() / (float)RAND_MAX;
}

// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss()
{
    float u1 = urand();
    float u2 = urand();

    if (u1 < 1e-6f)
    {
        u1 = 1e-6f;
    }

    return sqrtf(-2 * logf(u1)) * cosf(2*CUDART_PI_F * u2);
}

float frequency(float k, float g, float depth)
{
	return sqrt(g * k * tanh(min(k * depth, 2)));
}
float frequencyDerivative(float k, float g, float depth)
{
	float th = tanh(min(k * depth, 20));
	float ch = cosh(k * depth);
	return g * (depth * k / ch / ch + th) / frequency(k, g, depth) / 2;
}
//free function
float phillips(float Kx, float Ky, float Vdir, float V, float A, float dir_depend, float g, float depth)
//float phillips(std::shared_ptr<OceanFFT> cuOceanObj)
{
            float k_squared = Kx*Kx + Ky*Ky;

            if (k_squared == 0.0f)
            {
                return 0.0f;
            }
            float phil = 0;
            float L = V * V / g;
            if(1.0/sqrt(k_squared) <= 15){
                // largest possible wave from constant wind of velocity v
                

                float k_x = Kx / sqrtf(k_squared);
                float k_y = Ky / sqrtf(k_squared);
                float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

                phil= A * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

                // filter out waves moving opposite to wind
                if (w_dot_k < 0.0f)
                {
                    phil *= dir_depend;
                }
            }

            //damp out waves with very small length w << l
            //float w = L/100000;
            //phil *= expf(-k_squared * w * w);
            //float dOmegadk = frequencyDerivative(sqrtf(k_squared), g, depth);
            //phil *= abs(dOmegadk);
            return phil;
}

void generate_h0(std::shared_ptr<OceanFFT> OceanObj)
{
    for (unsigned int y = 0; y <= OceanObj->meshSize; y++)
    {
        for (unsigned int x = 0; x<=OceanObj->meshSize; x++)
        {
            float kx = (-(int)OceanObj->meshSize / 2.0f + x) * (2.0f * CUDART_PI_F / OceanObj->patchSize);
            float ky = (-(int)OceanObj->meshSize / 2.0f + y) * (2.0f * CUDART_PI_F / OceanObj->patchSize);

            float P = sqrtf(phillips(kx, ky, OceanObj->windDir, OceanObj->speed, OceanObj->A, OceanObj->dir_depend,OceanObj->g, OceanObj->depth));
            if (kx == 0.0f && ky == 0.0f)
            {
                P = 0.0f;
            }
            

            //float Er = urand()*2.0f-1.0f;
            //float Ei = urand()*2.0f-1.0f;
            float Er = zeno::gauss();
            float Ei = zeno::gauss();

            float theta = urand() * 2.0 * 3.1415926;
            float h0_re = Er * P * cosf(theta) * CUDART_SQRT_HALF_F;
            float h0_im = Ei * P * sinf(theta) * CUDART_SQRT_HALF_F;

            int i = y*OceanObj->spectrumW+x;
            OceanObj->h_h0[i].x = h0_re;
            OceanObj->h_h0[i].y = h0_im;
        }
    }
}

//here begins zeno stuff
 
struct UpdateDx : zeno::INode {
    virtual void apply() override {
        printf("UpdateDxKernel::apply() called!\n");
        
        auto real_ocean = get_input<OceanFFT>("real_ocean");
        //have some questions need to ask for details.
        UpdateDxKernel(real_ocean->Dx, real_ocean->Dz, real_ocean->width, real_ocean->height);
    }
};
ZENDEFNODE(UpdateDx,
    { /* inputs:  */ { "real_ocean" }, 
      /* outputs: */ { }, 
      /* params: */  { }, 
      /* category: */ {"Ocean",}});


struct GenerateSpectrum : zeno::INode {
    virtual void apply() override {
        printf("GenerateSpectrumKernel::apply() called!\n");

        auto real_ocean = get_input<OceanFFT>("real_ocean");


        GenerateSpectrumKernel(real_ocean->d_h0, real_ocean->d_ht, real_ocean->Dx, 
        real_ocean->Dz, real_ocean->in_width, real_ocean->out_width, 
        real_ocean->out_height, real_ocean->animTime, real_ocean->patchSize);
    }
};
ZENDEFNODE(GenerateSpectrum,
    { /* inputs: */ { "real_ocean" }, 
    /* outputs: */ {},
    /* params: */  {}, 
    /* category: */ { "Ocean",}});


struct UpdateHeightmap : zeno::INode {
    virtual void apply() override {
        printf("UpdateHeightmapKernel::apply() called!\n");

        auto real_ocean = get_input<OceanFFT>("real_ocean");

        UpdateHeightmapKernel(real_ocean->d_ht, real_ocean->d_ht, real_ocean->width, real_ocean->height);
    }
};

ZENDEFNODE(UpdateHeightmap,
        { /* inputs: */ { "real_ocean"}, 
        /* outputs: */  { }, 
        /* params: */ { }, 
        /* category: */ {"Ocean",}});



struct MakeCuOcean : zeno::INode {
    void apply() override
    {

        auto cuOceanObj = std::make_shared<OceanFFT>();

        

        

        //other parameters
        cuOceanObj->amplitude = get_input<zeno::NumericObject>("amp")->get<float>();
        cuOceanObj->WaveExponent = get_input<zeno::NumericObject>("WaveExponent")->get<int>();
        cuOceanObj->choppyness   = get_input<zeno::NumericObject>("chop")->get<float>();

        //meshSize = 2^waveExponent
        cuOceanObj->meshSize  = 1<<cuOceanObj->WaveExponent;

        //spectrumH/W = meshSize
        cuOceanObj->spectrumH   = cuOceanObj->meshSize + 1;
        cuOceanObj->spectrumW   = cuOceanObj->meshSize + 4;
        
        cuOceanObj->spectrumSize = cuOceanObj->spectrumH * cuOceanObj->spectrumW;
        cuOceanObj->L_scale = get_input<zeno::NumericObject>("patchSize")->get<float>()/100.0;
        //gravity=
        cuOceanObj->g   = get_input<zeno::NumericObject>("gravity")->get<float>() / cuOceanObj->L_scale;
        //patchSize = 
        cuOceanObj->patchSize  = 100;
        //dir = 
        cuOceanObj->windDir  = get_input<zeno::NumericObject>("windDir")->get<float>()/360.0*2.0*CUDART_PI_F;
        //timeScale
        //timeshift
        cuOceanObj->timeScale  = get_input<zeno::NumericObject>("timeScale")->get<float>();
        cuOceanObj->timeShift  = get_input<zeno::NumericObject>("timeshift")->get<float>();
        //speed = 
        cuOceanObj->speed  = get_input<zeno::NumericObject>("speed")->get<float>() / cuOceanObj->L_scale;
        cuOceanObj->depth = get_input<zeno::NumericObject>("depth")->get<float>();
        cuOceanObj->A *= cuOceanObj->amplitude;



        // create FFT plan
        cufftPlan2d(&(cuOceanObj->fftPlan), cuOceanObj->meshSize, cuOceanObj->meshSize, CUFFT_C2C);
        

        cudaMalloc((void**)&(cuOceanObj->d_h0), sizeof(float2)*cuOceanObj->spectrumSize);
        cudaMalloc((void**)&(cuOceanObj->d_ht), sizeof(float2)*cuOceanObj->meshSize*cuOceanObj->meshSize);

        cuOceanObj->g_hhptr = (float2*)malloc(sizeof(float2) *cuOceanObj->meshSize*cuOceanObj->meshSize);

        cudaMalloc((void**)&(cuOceanObj->Dx), sizeof(float2) *cuOceanObj->meshSize*cuOceanObj->meshSize);
        cudaMalloc((void**)&(cuOceanObj->Dz), sizeof(float2) *cuOceanObj->meshSize*cuOceanObj->meshSize);
      

        cuOceanObj->g_hDz = (float2*)malloc(sizeof(float2) * cuOceanObj->meshSize*cuOceanObj->meshSize);
        cuOceanObj->g_hDx = (float2*)malloc(sizeof(float2) * cuOceanObj->meshSize*cuOceanObj->meshSize);
        cuOceanObj->h_h0 = (float2*)malloc(sizeof(float2)  * cuOceanObj->spectrumSize);

        generate_h0(cuOceanObj);
        //cpu to gpu
        cudaMemcpy(cuOceanObj->d_h0, cuOceanObj->h_h0, sizeof(float2)*cuOceanObj->spectrumSize, cudaMemcpyHostToDevice);
 
        set_output("gpuOcean", cuOceanObj);
    }
};

ZENDEFNODE(MakeCuOcean,
        { /* inputs:  */ { {"int", "WaveExponent", "8"}, {"float", "depth", "5000"}, {"float","chop", "0.5"}, {"float", "gravity", "9.81"}, {"float", "windDir", "0"}, {"float","timeScale","1.0"}, {"float", "patchSize", "100.0"}, {"float", "speed", "100.0"}, {"float", "timeshift", "0.0"}, {"float", "amp", "1.0"}}, 
          /* outputs: */ { "gpuOcean", }, 
          /* params: */  {  }, 
          /* category: */ {"Ocean",}});


//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------

inline float lerp(float const &a, float const &b, float const &x)
{
    return (1-x)*a + x * b;
}
inline float bilerp(float const &v00, float const &v01, float const &v10, float const &v11, float const &x, float const &y)
{
    return lerp(lerp(v00, v01, x), lerp(v10, v11, x), y);
}
struct OceanCompute : zeno::INode {
    void apply() override{

    // move to ocean 
    //CUDA Implementation
    // generate wave spectrum in frequency domain
    // execute inverse FFT to convert to spatial domain
    //CUDA TEST generate wave spectrum in frequency domain
    // execute inverse FFT to convert to spatial domain
    // update heightmap values
    //-----------------------------------------------------------------------------------
    auto ingrid = get_input<zeno::PrimitiveObject>("grid");
    auto t = get_input<zeno::NumericObject>("time")->get<float>();

    auto CalOcean = get_input<OceanFFT>("ocean_FFT");

    
    //--------------------------------------------------------------------------------------------------
    GenerateSpectrumKernel(CalOcean->d_h0, CalOcean->d_ht, CalOcean->Dx, CalOcean->Dz, CalOcean->spectrumW, CalOcean->meshSize, CalOcean->meshSize, CalOcean->timeShift + CalOcean->timeScale*t, CalOcean->patchSize);
    cufftExecC2C(CalOcean->fftPlan, CalOcean->d_ht, CalOcean->d_ht, CUFFT_INVERSE);
    cufftExecC2C(CalOcean->fftPlan, CalOcean->Dx, CalOcean->Dx, CUFFT_INVERSE);
    cufftExecC2C(CalOcean->fftPlan, CalOcean->Dz, CalOcean->Dz, CUFFT_INVERSE);
   

    UpdateHeightmapKernel(CalOcean->d_ht, CalOcean->d_ht, CalOcean->meshSize, CalOcean->meshSize);
    //choppy
    UpdateDxKernel(CalOcean->Dx, CalOcean->Dz, CalOcean->meshSize, CalOcean->meshSize);


    cudaMemcpy(CalOcean->g_hhptr, CalOcean->d_ht, sizeof(float2)*CalOcean->meshSize * CalOcean->meshSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(CalOcean->g_hDx, CalOcean->Dx, sizeof(float2)*CalOcean->meshSize * CalOcean->meshSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(CalOcean->g_hDz, CalOcean->Dz, sizeof(float2)*CalOcean->meshSize * CalOcean->meshSize, cudaMemcpyDeviceToHost);

    auto grid = std::make_shared<zeno::PrimitiveObject>(*ingrid);
    auto &inpos = ingrid->verts;
    auto &pos = grid->attr<vec3f>("pos");
    auto &vel = grid->add_attr<vec3f>("vel");
    grid->resize(ingrid->size());
    float h = CalOcean->L_scale * (float)CalOcean->patchSize / (float)(CalOcean->meshSize);
    float L = CalOcean->L_scale * (float)CalOcean->patchSize;
#pragma omp parallel for
    for(size_t i = 0; i<pos.size(); i++)
    {
        float u = pos[i][0]+0.5*L, v = pos[i][2]+0.5*L;
        float uu = std::fmod(std::fmod(u, L) + L, L);
        float vv = std::fmod(std::fmod(v, L) + L, L);
        uu = uu/h;
        vv = vv/h;
        int tu = (int)uu;
        int tv = (int)vv;
        float cx = uu - tu;
        float cy = vv - tv;
        int i00 = tv * CalOcean->meshSize + tu;
        int i01 = tv * CalOcean->meshSize + (tu + 1)%CalOcean->meshSize;
        int i10 = ((tv + 1)%(CalOcean->meshSize)) * CalOcean->meshSize + tu;
        int i11 = ((tv + 1)%(CalOcean->meshSize)) * CalOcean->meshSize + (tu + 1)%(CalOcean->meshSize);
        float h00 = CalOcean->g_hhptr[i00].x;
        float h01 = CalOcean->g_hhptr[i01].x;
        float h10 = CalOcean->g_hhptr[i10].x;
        float h11 = CalOcean->g_hhptr[i11].x;
        float Dx00 = CalOcean->g_hDx[i00].x;
        float Dx01 = CalOcean->g_hDx[i01].x;
        float Dx10 = CalOcean->g_hDx[i10].x;
        float Dx11 = CalOcean->g_hDx[i11].x;
        float Dz00 = CalOcean->g_hDz[i00].x;
        float Dz01 = CalOcean->g_hDz[i01].x;
        float Dz10 = CalOcean->g_hDz[i10].x;
        float Dz11 = CalOcean->g_hDz[i11].x;
        float h = bilerp(h00, h01, h10, h11, cx, cy);
        float Dx = bilerp(Dx00, Dx01, Dx10, Dx11, cx, cy);
        float Dz = bilerp(Dz00, Dz01, Dz10, Dz11, cx, cy);
        vel[i] = CalOcean->L_scale * zeno::vec3f(-CalOcean->choppyness*Dx, h, -CalOcean->choppyness*Dz);
        pos[i] = inpos[i] + vel[i];
    }

//    auto grid = std::make_shared<zeno::PrimitiveObject>();
//    //here goes the zeno part
//    auto &pos = grid->add_attr<vec3f>("pos");
//    auto &vel = grid->add_attr<vec3f>("vel");
//    auto &index = grid->add_attr<vec3i>("index");
//    grid->resize(CalOcean->meshSize * CalOcean->meshSize);
//    int k = 0;
//    float h = (float)CalOcean->patchSize / (float)(CalOcean->meshSize - 1);
//    for(int j=0; j<CalOcean->meshSize;j++)
//    {
//        for(int i=0;i<CalOcean->meshSize;i++)
//        {
//            pos.push_back(CalOcean->L_scale * zeno::vec3f(i, 0, j)*h);
//            vel.push_back(CalOcean->L_scale * zeno::vec3f(-CalOcean->choppyness*CalOcean->g_hDx[k].x, CalOcean->g_hhptr[k].x, -CalOcean->choppyness*CalOcean->g_hDz[k].x));
//            index.push_back(zeno::vec3i(i,0,j));
//            k++;
//        }
//    }


    
    //grid->userData.get("h") = std::make_shared<NumericObject>(CalOcean->L_scale * h);
    //grid->userData.get("transform") = std::make_shared<NumericObject>(0.5f * zeno::vec3f(CalOcean->L_scale * CalOcean->patchSize, 0, CalOcean->L_scale * CalOcean->patchSize));
    set_output("OceanData", grid);
    }
};


ZENDEFNODE(OceanCompute,
        { /* inputs:  */ {"grid", "time","ocean_FFT", }, 
          /* outputs: */ { "OceanData", }, 
          /* params: */  {  }, 
          /* category: */ {"Ocean",}});






}






