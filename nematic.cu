// #include "msgpack.hpp"
#include <vector>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <ostream>
#include <chrono>
#include <ctime>
#include "/home/aaveg/cuPSS/inc/cupss.h"
#include <chrono>

#ifdef WITHCUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define NX 128
#define NY 128



void noSlip(evolver *, float2 *, int, int, int);
__global__ void v_zero(float2 *vs, int sx, int sy, int sz);

void phiBound(evolver *, float2 *, int, int, int);
__global__ void phi_grav(float2 *vs, int sx, int sy, int sz);

int main() {

    int gpu = 1;
    float dx = 1.0f;
    float dy = 1.0f;
    float dt = 0.05f;
    // int nn = 1000000000;
    // int total_steps = 1000000000;
    int steps_per_l = 20;
    float a2 = -1.0f;
    float a4 = 1.0f;
    float kQ = 4.0f;
    float lambda = 1.0f;
    // float gamma = 20.0f;
    float eta = 1.0f;
    float fric = 0.01f; //0.01f;
    float beta = -1.0f;
    float output_every_n_steps = 50000000;

    // float angle_radians = M_PI / 3.0; 
    // float nx_ = cos(angle_radians);
    // float ny_ = sin(angle_radians);

    evolver system(gpu, NX, NY, dx, dy, dt, output_every_n_steps );

    system.createField("Qxx", true);
    system.createField("Qxy", true);
    system.createField("vx", false);
    system.createField("vy", false);
    system.createField("phi", true);
    system.createField("alpha", false);

    system.createField("iqxQxx", false);
    system.createField("iqyQxx", false);
    system.createField("iqxQxy", false);
    system.createField("iqyQxy", false);
    system.createField("sigxx", false);
    system.createField("sigxy", false);
    system.createField("sigyy", false);
    system.createField("w", false);
    system.createField("Q2", false);

    system.createField("iqxphi", false);
    system.createField("iqyphi", false);
    system.addEquation("iqxphi = iqx*phi");
    system.addEquation("iqyphi = iqy*phi");

    system.addParameter("a2", a2);
    system.addParameter("a4", a4);
    system.addParameter("kQ", kQ);
    system.addParameter("lambda", lambda);
    // system.addParameter("gamma", gamma);
    system.addParameter("eta", eta);
    system.addParameter("fric", fric);
    system.addParameter("beta", beta);
    system.addParameter("ka",  40.0);

    system.addParameter("a", -1.0);
    system.addParameter("b",  1.0);
    system.addParameter("k",  40.0);

    system.fieldsMap["vx"]->hasCB = true;
    system.fieldsMap["vx"]->callback = noSlip;
    system.fieldsMap["vy"]->hasCB = true;
    system.fieldsMap["vy"]->callback = noSlip;
    // system.fieldsMap["phi"]->hasCB = true;
    // system.fieldsMap["phi"]->callback = phiBound;

    system.addEquation("dt Qxx + (kQ*q^2)*Qxx = -a2*phi*Qxx -a4*Q2*Qxx - vx*iqxQxx - vy*iqyQxx + lambda*iqx*vx - 2*Qxy*w");  // updated -a2*phi*Qxx and moved to rhs
    system.addEquation("dt Qxy + (kQ*q^2)*Qxy = -a2*phi*Qxy -a4*Q2*Qxy - vx*iqxQxy - vy*iqyQxy + 0.5*lambda*iqx*vy + 0.5*lambda*iqy*vx + 2*Qxx*w"); // updated -a2*phi*Qxy and moved to rhs
    system.addEquation("iqxQxx = iqx*Qxx");
    system.addEquation("iqxQxy = iqx*Qxy");
    system.addEquation("iqyQxx = iqy*Qxx");
    system.addEquation("iqyQxy = iqy*Qxy");

    system.addEquation("sigxx = beta * alpha * 0.5*(1+phi) * Qxx  -  ka*0.5*(iqxphi^2-iqyphi^2)  "); // added phi  and f_phi
    system.addEquation("sigxy = beta * alpha * 0.5*(1+phi) * Qxy  -  ka*iqxphi*iqyphi"); // added phi  and f_phi
    system.addEquation("sigyy = - beta * alpha * 0.5*(1+phi) * Qxx  +  ka*0.5*(iqxphi^2-iqyphi^2)"); // added phi  and f_phi   
    system.addEquation("w = 0.5*iqx*vy - 0.5*iqy*vx");
    system.addEquation("Q2 = Qxx^2 + Qxy^2");

    // gravitational terms
    system.addParameter("gGrav", 0.01);
    system.createField("fgy", false);
    // system.createField("ident", false);
    system.addEquation("fgy = - (0.5*(phi+1))*gGrav");


    // stokes flow solution 
    // system.addEquation("vx * (fric + eta*q^2) = -(iqx * iqy^2 * 1/q^2)*sigxx + (iqy + iqx^2*iqy*1/q^2 + iqx^2*iqy*1/q^2) * sigxy + (iqx*iqy^2*1/q^2) * sigyy");
    system.addEquation("vx * (fric + eta*q^2) = -(iqx * iqy^2 * 1/q^2)*sigxx + ( -iqy^3*1/q^2 + iqx^2*iqy*1/q^2) * sigxy + (iqx*iqy^2*1/q^2) * sigyy + fgy*iqx*iqy*1/q^2");
    // system.addEquation("vy * (fric + eta*q^2) = (iqx + iqx*iqy^2*1/q^2 + iqx*iqy^2*1/q^2) * sigxy + (iqx^2*iqy*1/q^2) * sigxx  + (iqy + iqy^3*1/q^2) * sigyy - fgy*iqx*iqx*i/q^2");
    system.addEquation("vy * (fric + eta*q^2) = (-iqx^3*1/q^2 + iqx*iqy^2*1/q^2) * sigxy + (iqx^2*iqy*1/q^2) * sigxx  - (iqx^2*iqy*1/q^2) * sigyy - fgy*iqx*iqx*1/q^2");
    
    // Cahn Hilliard 
    system.addEquation("dt phi + q^2*(a + k*q^2)*phi= -vx*iqxphi - vy*iqyphi - b*q^2*phi^3"); // added convection term -(vx*iqx + vy*iqy)*phi
    // system.initializeUniformNoise("phi", 0.01);

    // system.printInformation();
 
    // std::srand(12);
    std::cout <<  "begin\n";
    std::cout.flush();

    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            std::cin >> system.fields[0]->real_array[index].x   ;//(nx_*nx_ - 0.5) + 0.01f * (float)(rand()%200-100);
        }
    }
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            std::cin >> system.fields[1]->real_array[index].x   ;//nx_*ny_ + 0.01f * (float)(rand()%200-100);
        }
    }
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            std::cin >> system.fields[2]->real_array[index].x   ;
        }
    }
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            std::cin >> system.fields[3]->real_array[index].x   ;
        }
    }
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            std::cin >> system.fields[4]->real_array[index].x   ;
        }
    }

            // if(i < 0.5*NX){
            //     system.fields[14]->real_array[index].x = -1;
            // }
            // else{
            //     system.fields[14]->real_array[index].x = 1;
            // }
            
    system.prepareProblem();


    // for (int i = 0; i < system.fields.size(); i++)
    // {
    //     system.fields[i]->outputToFile = false;
    // }

    // system.setOutputField("Q2", true);
    // system.setOutputField("alpha", true);

    // system.setOutputField("vx", true);
    // system.setOutputField("vy", true);

    

    float2 alpha_temp[NX*NY];
    float2 Qxx_temp[NX*NY];
    float2 Qxy_temp[NX*NY];
    float2 vx_temp[NX*NY];
    float2 vy_temp[NX*NY];
    float2 phi_temp[NX*NY];
    for (int i = 0; i < NX*NY; i++)
    {        
        alpha_temp[i].x = 0.0f;
        alpha_temp[i].y = 0.0f;
    }


    // float data_to_send[4*NX*NY]; 
    
    int frame_count = 0;
    // auto start = std::chrono::system_clock::now();

    // int client_socket = accept(server_socket, nullptr, nullptr);


    while (true) {

        // ##################################
        //      RECIEVE AND COPY DATA TO GPU 
        // ##################################
        
        for (int ny = 0; ny < NY; ny++){
            for (int nx = 0; nx < NX; nx++){
                int index = ny * NX + nx;
                std::cin >> alpha_temp[index].x;
            }
        } 

        cudaMemcpy(system.fields[5]->real_array_d, alpha_temp, NX*NY*sizeof(float2), cudaMemcpyHostToDevice);
        system.fields[5]->toComp();

        // ###########################
        //         RUN SOLVER
        // ###########################
        for (int tt = 0; tt < steps_per_l; tt++)
        {
            system.advanceTime();
            // std::cout << "taken one time step" << std::endl;
            frame_count++;
        }

        // ###########################
        //         COPY TO RAM
        // ###########################
        cudaMemcpy(Qxx_temp, system.fields[0]->real_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);
        cudaMemcpy(Qxy_temp, system.fields[1]->real_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);
        cudaMemcpy(vx_temp, system.fields[2]->real_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);
        cudaMemcpy(vy_temp, system.fields[3]->real_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);
        cudaMemcpy(phi_temp, system.fields[4]->real_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);

        for (int ny = 0; ny < NY; ny++){
            for (int nx = 0; nx < NX; nx++){
                int index = ny * NX + nx;
                // std::cout << system.fields[0]->real_array[index].x << " ";
                std::cout << Qxx_temp[index].x << "&";
                std::cout << Qxy_temp[index].x << "&";
                std::cout << vx_temp[index].x << "&";
                std::cout << vy_temp[index].x << "&";
                std::cout << phi_temp[index].x << " ";
            }
        } 
            std::cout <<  "\n";
            std::cout.flush();

      
    }

    return 0;
}



void noSlip(evolver *sys, float2 *vs, int sx, int sy, int sz)
{
    v_zero<<<sys->blocks, sys->threads_per_block>>>(vs, sx, sy, sz);
}
__global__ void v_zero(float2 *vs, int sx, int sy, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (index < sx*sy*sz )
    {
        if (j < sy/16 || j > sy - sy/16)
            vs[index].x = 0.0f;
    }
}


void phiBound(evolver *sys, float2 *phig, int sx, int sy, int sz)
{
    phi_grav<<<sys->blocks, sys->threads_per_block>>>(phig, sx, sy, sz);
}
__global__ void phi_grav(float2 *phig, int sx, int sy, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (index < sx*sy*sz)
        {
            if (j < sy/16)
            {
                phig[index].x = 1.0f;
            }

            if (j > sy - sy/16)
            {
                phig[index].x = -1.0f;
            }
        }
}