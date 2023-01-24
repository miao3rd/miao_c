
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fftw3.h"
#include "pms_cuda.h"
#include "pms_spline_cuda.h"


#define NO_DIMENSION 1

#define S_IFFT_PROOF 0
#define OUTPUT_S 0

#define LENGTH 10


/***************CPU mode operation*************************/
#define CPU_MODE 1

#define Nx 256
#define Ny 256
#define Nf 128


double complex_mutl_realrst(fftw_complex in1, fftw_complex in2)
{
	return in1[0] * in2[0] - in1[1] * in2[1];
}

double complex_mutl_imgrst(fftw_complex in1, fftw_complex in2)
{
	return in1[0] * in2[1] + in1[1] * in2[0];
}

__global__ void pms_phase_compensation_kernel(double* in, double* K, double* Kz,double R0)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int off_set_address = id * 2;
	double S_ftxy_real;
	double S_ftxy_img;
	double test_real;
	double test_img;

	S_ftxy_real = *(in + off_set_address);
	S_ftxy_img = *(in + off_set_address+1);
	test_real = cos((-(*(K + id)) + (*(Kz + id))) * R0);
	test_img = sin((-(*(K + id)) + (*(Kz + id))) * R0);

	*(in + off_set_address) = test_real * S_ftxy_real - test_img * S_ftxy_img;
	*(in + off_set_address + 1) = test_real * S_ftxy_img + test_img * S_ftxy_real;
}

void pms_phase_compensation(_fcpms* in, _dpms* K, _dpms* Kz, double R0)
{
	int total_thread;
	total_thread = in->length;
	int nBlock;
	nBlock = total_thread / N_THREAD + 1;
	_dpms dev_in;
	_dpms dev_k;
	_dpms dev_kz;

	pms_creat_memory_cuda_with_paramete_cuda(&dev_in, in->length * 2);
	pms_creat_memory_cuda_with_paramete_cuda(&dev_k, K->length);
	pms_creat_memory_cuda_with_paramete_cuda(&dev_kz, Kz->length);

	cudaMemcpy(dev_in.pointer, (double*)(in->pointer), in->length * 2 * sizeof(double), cudaMemcpyHostToDevice);
	pms_memcpy_cuda(&dev_k, K, cudaMemcpyHostToDevice);
	pms_memcpy_cuda(&dev_kz, Kz, cudaMemcpyHostToDevice);


	pms_phase_compensation_kernel << <nBlock, N_THREAD >> > (dev_in.pointer, dev_k.pointer, dev_kz.pointer, R0);

	cudaMemcpy((double*)(in->pointer),dev_in.pointer,  in->length * 2 * sizeof(double), cudaMemcpyDeviceToHost);

}



int main()
{
	/************Define the loop parameters**************/
	int index_x;
	int index_y;
	int index_f;
	int index_l;
	int index;
	double temp;

	/************Set scene parameters***********/
	const double c = 3e8;		// Speed of light in m/s
	const double f0 = 75e9;		// Radar transmit signal center frequency in Hz
	const double B = 35e9;		// Bandwidth in Hz
	const double fs = 110e9;
	const double pi = 3.14159265358979;
	double lambda = c / (f0 + 0.5 * B);

	/**************Horizontal direction*************/
	double th = 20 * pi / 180;
	double Kxmax = 4 * pi * fs / c * sin(th / 2);
	double x_step = pi / Kxmax;
	double deltaX = x_step;
	double lambda2 = lambda / x_step;
	double Lx = Nx * x_step;
	double R0 = Lx / 2 / tan(th / 2);
	_dpms TRX_pos = { NULL,1,Nx,NO_DIMENSION,NO_DIMENSION };
	pms_creat_memory(&TRX_pos);
	for (index_x = 0; index_x < Nx; index_x++)
	{
		*(TRX_pos.pointer + index_x) = (-(double)(Nx - 1) / 2 + index_x) * deltaX;
	}

	/****************Vertical direction***************/
	double deltaY = deltaX;
	_dpms TRY_pos = { NULL,1,Ny,NO_DIMENSION,NO_DIMENSION };
	pms_creat_memory(&TRY_pos);
	for (index_y = 0; index_y < Ny; index_y++)
	{
		*(TRY_pos.pointer + index_y) = (-(double)(Nx - 1) / 2 + index_y) * deltaY;
	}
	double Ly = ((__int64)Ny - 1) * deltaY;
	double delta_f = B / (Nf - 1);		//Transceiver frequency sampling interval in Hz
	double rhoz = c / 2 / B;			//Distance dimensional theoretical resolution
	double TRZ = -R0;					//The transceiver is located at a distance from the dimensional position in m.

	/*************Calculate theoretical resolution************/
	double rhoy = deltaX;
	double rhox = deltaY;

	/**************Calculating the wave number domain***************/
	_dpms fvec = { NULL,1,Nf,NO_DIMENSION,NO_DIMENSION };
	_dpms kvec = { NULL,1,Nf,NO_DIMENSION,NO_DIMENSION };
	//	fvec.pointer = (double*)malloc(fvec.length * sizeof(double));
	pms_creat_memory(&fvec);
	pms_creat_memory(&kvec);
	for (temp = f0, index_f = 0; index_f <= Nf; temp += delta_f, index_f++)
	{
		*(fvec.pointer + index_f) = temp;
	}
	for (index_f = 0; index_f < Nf; index_f++)
	{
		*(kvec.pointer + index_f) = (*(fvec.pointer + index_f)) * 2 * pi / c;
	}

	_dpms Kz = { NULL,3,Nx,Ny,Nf };
	pms_creat_memory(&Kz);

	_dpms Kx = { NULL,2,Nx,Ny,NO_DIMENSION };
	pms_creat_memory(&Kx);

	_dpms Ky = { NULL,2,Nx,Ny,NO_DIMENSION };
	pms_creat_memory(&Ky);

	double kx_max = Kxmax;
	_dpms kx = { NULL,1,Nx,NO_DIMENSION,NO_DIMENSION };
	pms_creat_memory(&kx);
	linspace(-kx_max, kx_max, Nx, kx.pointer);

	double ky_max = pi / deltaY;
	_dpms ky = { NULL,1,Nx,NO_DIMENSION,NO_DIMENSION };
	pms_creat_memory(&ky);
	linspace(-ky_max, ky_max, Ny, ky.pointer);

	_dpms ones_Ny = { NULL,1,Ny,NO_DIMENSION,NO_DIMENSION };
	pms_creat_memory(&ones_Ny);
	std::fill(ones_Ny.pointer, ones_Ny.pointer + Ny, 1);
	matmul(kx.pointer, ones_Ny.pointer, Kx.pointer, kx.length, 1, ones_Ny.length);
	pms_free_memory(&ones_Ny);

	_dpms ones_Nx = { NULL,1,Nx,NO_DIMENSION,NO_DIMENSION };
	pms_creat_memory(&ones_Nx);
	std::fill(ones_Nx.pointer, ones_Nx.pointer + Nx, 1);
	matmul(ones_Nx.pointer, ky.pointer, Ky.pointer, ones_Ny.length, 1, kx.length);
	pms_free_memory(&ones_Nx);

	_dpms kz = { NULL,2,Nx,Ny,NO_DIMENSION };
	pms_creat_memory(&kz);

	double k;
	double k_temp;

	for (index_f = 0; index_f < Nf; index_f++)
	{
		k = *(kvec.pointer + index_f);
		k_temp = pow(2 * k, 2);
		for (index_x = 0; index_x < Nx; index_x++)
			for (index_y = 0; index_y < Ny; index_y++)
			{
				*(kz.pointer + pms_offset_address_2d(index_x, index_y, &kz)) = sqrt(k_temp - pow(*(Kx.pointer + pms_offset_address_2d(index_x, index_y, &Kx)), 2) - pow(*(Ky.pointer + pms_offset_address_2d(index_x, index_y, &Ky)), 2));
			}
		for (index_x = 0; index_x < Nx; index_x++)
			for (index_y = 0; index_y < Ny; index_y++)
			{
				*(Kz.pointer + pms_offset_address_3d(index_x, index_y, index_f, &Kz)) = *(kz.pointer + pms_offset_address_2d(index_x, index_y, &kz));
			}
	}

	_dpms k1 = { NULL,1,kvec.length,NO_DIMENSION,NO_DIMENSION };
	pms_creat_memory(&k1);

	for (index_f = 0; index_f < Nf; index_f++)
	{
		*(k1.pointer + index_f) = *(kvec.pointer + index_f) * 2;
	}

	_dpms K = { NULL,3,Nx,Ny,Nf };
	pms_creat_memory(&K);

	for (index_x = 0; index_x < Nx; index_x++)
		for (index_y = 0; index_y < Ny; index_y++)
		{
			memcpy(K.pointer + pms_offset_address_3d(index_x, index_y, 0, &K), k1.pointer, k1.length * sizeof(double));
		}
	_dpms kx2 = { NULL,2,Nx,Nf,NO_DIMENSION };
	pms_creat_memory(&kx2);

	_dpms kz2 = { NULL,2,Nx,Nf,NO_DIMENSION };
	pms_creat_memory(&kz2);

	for (index_f = 0; index_f < Nf; index_f++)
	{
		for (index_x = 0; index_x < Nx; index_x++)
			for (index_y = 0; index_y < Ny; index_y++)
			{
				*(kx2.pointer + pms_offset_address_3d(index_x, index_y, index_f, &kx2)) = *(kx.pointer + index_x);
			}
	}
	_dpms kkk = { NULL,3,Nx,Nf,Ny };
	pms_creat_memory(&kkk);
	//	_dpms kz2 = { NULL,2,Nx,Nf,NO_DIMENSION };
	//	pms_creat_memory(&kz2);

	permute3(Kz.pointer, kkk.pointer, Kz.number_of_x, Kz.number_of_y, Kz.number_of_z, 1, 3, 2);
	index_y = 0;
	for (index_x = 0; index_x < Nx; index_x++)
		for (index_f = 0; index_f < Nf; index_f++)
		{
			*(kz2.pointer + pms_offset_address_2d(index_x, index_f, &kz2)) = *(kkk.pointer + pms_offset_address_3d(index_x, index_f, index_y, &kkk));
		}
	pms_free_memory(&kkk);

	/**************Projection.c.v*******************/
	/*
	_fpms z_map = { NULL,1,Kz.length,NO_DIMENSION,NO_DIMENSION };
	pms_creat_memory(&z_map);

	for (index_l = 0; index_l < Kz.length; index_l++)
	{
		*(z_map.pointer + index_l) = *(Kz.pointer + index_l);
	}
//	memcpy(z_map.pointer, Kz.pointer, Kz.length*sizeof(double));
	std::sort(z_map.pointer, z_map.pointer + z_map.length);
	z_map.length=std::unique(z_map.pointer, z_map.pointer + z_map.length)-z_map.pointer;
	*/

	/**************Simulation point location*******************/
	int number_of_ptar_x = 34;
	int number_of_ptar_y = 4;
	_dpms Ptar = { NULL,2,number_of_ptar_x,number_of_ptar_y,NO_DIMENSION };
	pms_creat_memory(&Ptar);
	double ptar_data[] = { 0,0,0.5,1, 0.05,0,0.5,1, 0.1,0,0.5,1, 0.15,0,0.5,1, 0.2,0,0.5,1, 0.25,0,0.5,1, 0.3,0,0.5,1, 0.35,0,0.5,1, 0.4,0,0.5,1,
			-0.05,0,0.5,1, -0.1,0,0.5,1, -0.15,0,0.5,1, -0.2,0,0.5,1, -0.25,0,0.5,1, -0.3,0,0.5,1, -0.35,0,0.5,1, -0.4,0,0.5,1,
		  0.4,-0.4,0.5,1, 0.4,-0.35,0.5,1, 0.4,-0.3,0.5,1, 0.4,-0.25,0.5,1, 0.4,-0.2,0.5,1, 0.4,-0.15,0.5,1, 0.4,-0.1,0.5,1, 0.4,-0.05,0.5,1, 0.4,0,0.5,1,
		  0.4,0.4,0.5,1, 0.4,0.35,0.5,1, 0.4,0.3,0.5,1, 0.4,0.25,0.5,1, 0.4,0.2,0.5,1, 0.4,0.15,0.5,1, 0.4,0.1,0.5,1, 0.4,0.05,0.5,1 };

	int Object_num = Ptar.number_of_x;

	_dpms dual_Ptar_matrix = { NULL,2,number_of_ptar_y,4,NO_DIMENSION };
	double dual_Ptar_matrix_data[] = { rhox * Nx / 2,0,0,0, 0,rhoy * Ny / 2,0,0, 0,0,rhoz * Nf / 2,0, 0,0,0,1 };
	dual_Ptar_matrix.pointer = dual_Ptar_matrix_data;
	matmul(ptar_data, dual_Ptar_matrix.pointer, Ptar.pointer, Ptar.number_of_x, Ptar.number_of_y, dual_Ptar_matrix.number_of_y);


	_fcpms S = { NULL,3,Nx,Ny,Nf };
	pms_creat_memory(&S);
	_fcpms s_f = { NULL,1,Nf,NO_DIMENSION,NO_DIMENSION };
	pms_creat_memory(&s_f);

	int index_s;
	int index_Ptar;

	/***************Construct the simulated echo signal******************/
#if CPU_MODE
	double xn;
	double yn;
	double zn;
	double R;
	double sigma;

	for (index_y = 0; index_y < Ny; index_y++)
		for (index_x = 0; index_x < Nx; index_x++)
		{
			for (index_f = 0; index_f < Nf; index_f++)
			{
				(*(s_f.pointer + index_f))[0] = 0;
				(*(s_f.pointer + index_f))[1] = 0;
			}
			for (index_Ptar = 0; index_Ptar < Object_num; index_Ptar++)
			{
				xn = *(Ptar.pointer + pms_offset_address_2d(index_Ptar, 0, &Ptar));
				yn = *(Ptar.pointer + pms_offset_address_2d(index_Ptar, 1, &Ptar));
				zn = *(Ptar.pointer + pms_offset_address_2d(index_Ptar, 2, &Ptar));
				R = sqrt((pow(xn - *(TRX_pos.pointer + index_x), 2)) + (pow(yn - *(TRY_pos.pointer + index_y), 2)) + (pow(zn - TRZ, 2)));
				sigma = *(Ptar.pointer + pms_offset_address_2d(index_Ptar, 3, &Ptar));
				for (index_f = 0; index_f < Nf; index_f++)
				{
					(*(s_f.pointer + index_f))[0] += sigma * (cos(2 * kvec.pointer[index_f] * (R - R0)));
					(*(s_f.pointer + index_f))[1] += sigma * (sin(2 * kvec.pointer[index_f] * (R0 - R)));
				}
			}
			memcpy(S.pointer + pms_offset_address_3d(index_x, index_y, 0, &S), s_f.pointer, s_f.length * sizeof(fftw_complex));
		}
	//output_data(&S, "D:/TEMP/S.bin");

#else


#endif

	/**************Two-dimensional Fourier transform**************/

	fftw_plan plan;
	_fcpms S_ftxy = { NULL,3,Nx,Ny,Nf };
	pms_creat_memory(&S_ftxy);

#if S_IFFT_PROOF
	_fcpms S_i = { NULL,3,Nx,Ny,Nf };
	pms_creat_memory(&S_i);
#endif
	pms_permute_self(&S, 3, 2, 1);

	fftshift3(S.pointer, S.number_of_x, S.number_of_y, S.number_of_z);
	pms_copy_info(&S_ftxy, &S);
	for (index_x = 0; index_x < S.number_of_x; index_x++)
	{
		plan = fftw_plan_dft_2d(S.number_of_y, S.number_of_z, S.pointer + pms_offset_address_3d(index_x, 0, 0, &S), S_ftxy.pointer + pms_offset_address_3d(index_x, 0, 0, &S), FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);

	}
#if S_IFFT_PROOF
	for (index_x = 0; index_x < S.number_of_x; index_x++)
	{
		plan = fftw_plan_dft_2d(S.number_of_y, S.number_of_z, S_ftxy.pointer + pms_offset_address_3d(index_x, 0, 0, &S), S_i.pointer + pms_offset_address_3d(index_x, 0, 0, &S), FFTW_BACKWARD, FFTW_ESTIMATE);
		fftw_execute(plan);

	}
	double sum = 0;
	for (index_x = 0; index_x < S_ftxy.length; index_x++)
	{
		(*(S_i.pointer + index_x))[0] /= (S.number_of_y * S.number_of_z);
		(*(S_i.pointer + index_x))[1] /= (S.number_of_y * S.number_of_z);
		sum += ((*S_i.pointer + index_x)[0] - (*S.pointer + index_x)[0]);
		sum += ((*S_i.pointer + index_x)[1] - (*S.pointer + index_x)[1]);
	}

#endif
	//	pms_permute_self(&S_ftxy, 1,3,2);
	pms_permute_self(&S_ftxy, 3, 2, 1);
	fftshift3(S_ftxy.pointer, S_ftxy.number_of_x, S_ftxy.number_of_y, S_ftxy.number_of_z);
	//pms_permute_self(&S_ftxy, 3, 2, 1);

#if OUTPUT_S
	output_data(&S_ftxy, "Sft.csv");
#endif

	//Phase compensation
	_fcpms temp_S_ftxy;
	pms_copy_info(&temp_S_ftxy, &S_ftxy);
	pms_creat_memory(&temp_S_ftxy);
	pms_copy(&temp_S_ftxy, &S_ftxy);

	double test_real;
	double test_img;
	double S_ftxy_real;
	double S_ftxy_img;

	//pms_phase_compensation(&S_ftxy,&K,&Kz,R0);
	
	for (index_x = 0; index_x < S_ftxy.number_of_x; index_x++)
	{
		for (index_y = 0; index_y < S_ftxy.number_of_y; index_y++)
		{
			for (index_f = 0; index_f < S_ftxy.number_of_z; index_f++)
			{
				S_ftxy_real = (*(temp_S_ftxy.pointer + pms_offset_address_3d(index_x, index_y, index_f, &temp_S_ftxy)))[0];
				S_ftxy_img = (*(temp_S_ftxy.pointer + pms_offset_address_3d(index_x, index_y, index_f, &temp_S_ftxy)))[1];
				test_real = cos((-(*(K.pointer + pms_offset_address_3d(index_x, index_y, index_f, &K))) + (*(Kz.pointer + pms_offset_address_3d(index_x, index_y, index_f, &Kz)))) * R0);
				test_img = sin((-(*(K.pointer + pms_offset_address_3d(index_x, index_y, index_f, &K))) + (*(Kz.pointer + pms_offset_address_3d(index_x, index_y, index_f, &Kz)))) * R0);

				(*(S_ftxy.pointer + pms_offset_address_3d(index_x, index_y, index_f, &S_ftxy)))[0] = test_real * S_ftxy_real - test_img * S_ftxy_img;
				(*(S_ftxy.pointer + pms_offset_address_3d(index_x, index_y, index_f, &S_ftxy)))[1] = test_real * S_ftxy_img + test_img * S_ftxy_real;


			}
		}
	}


	//Set interpolation parameters
	_dpms a;
	int ax;
	int ay;
	double max_a;
	double Kzm;
	double KzM;
	double Kzmid;
	int Nf2;


	pms_creat_memory_with_paramete(&a, Nx, Ny);
	pms_permute_self(&Kz, 3, 2, 1);
	memcpy(a.pointer, Kz.pointer + pms_offset_address_3d(Nf - 1, 0, 0, &Kz), Kz.number_of_y * Kz.number_of_z * sizeof(double));
	pms_permute_self(&Kz, 3, 2, 1);
	max_a = *a.pointer;
	for (index_x = 0; index_x < Kz.number_of_y; index_x++)
	{
		for (index_y = 0; index_y < Kz.number_of_z; index_y++)
		{
			if (max_a < *(a.pointer + pms_offset_address_2d(index_x, index_y, &a)))
			{
				max_a = *(a.pointer + pms_offset_address_2d(index_x, index_y, &a));
				ax = index_x;
				ay = index_y;
			}
		}
	}

	Kzm = *Kz.pointer;
	KzM = *Kz.pointer;

	for (index = 0; index < Kz.length; index++)
	{
		if (Kzm > * (Kz.pointer + index))
		{
			Kzm = *(Kz.pointer + index);
		}
		if (KzM < *(Kz.pointer + index))
		{
			KzM = *(Kz.pointer + index);
		}
	}
	Kzm = abs(Kzm);
	Kzmid = *(Kz.pointer + pms_offset_address_3d(ax, ay, 0, &Kz));
	Nf2 = (int)(Nf * (KzM - Kzm) / (KzM - Kzmid)) + 1;

	//	pms_free_memory(&kz2);
	pms_creat_memory_with_paramete(&kz2, Nf2);
	linspace(Kzm, KzM, Nf2, kz2.pointer);

	_dpms Kz2;
	_dpms Kx2;
	_dpms Ky2;
	_dpms K2;

	pms_creat_memory_with_paramete(&Kz2, Nx, Ny, Nf2);
	pms_creat_memory_with_paramete(&Kx2, Nx, Ny, Nf2);
	pms_creat_memory_with_paramete(&Ky2, Nx, Ny, Nf2);
	pms_creat_memory_with_paramete(&K2, Nx, Ny, Nf2);

	for (index_x = 0; index_x < Nx; index_x++)
	{
		for (index_y = 0; index_y < Ny; index_y++)
		{
			int Kz2_address_offset = pms_offset_address_3d(index_x, index_y, 0, &Kz2);
			memcpy(Kz2.pointer + Kz2_address_offset, kz2.pointer, kz2.length * sizeof(double));
		}
	}
	pms_repmat(&Ky, &Ky2, 1, 1, Nf2);
	pms_repmat(&Kx, &Kx2, 1, 1, Nf2);
	for (index = 0; index < Kx2.length; index++)
	{
		*(K2.pointer + index) = sqrt(pow(*(Kz2.pointer + index), 2) + pow(*(Kx2.pointer + index), 2) + pow(*(Ky2.pointer + index), 2));
	}


	_fcpms S2;
	_dpms K3;
	_dpms K4;
	_dpms S4;
	_fcpms S6;
	_dpms K3_uni;
	_dpms K4_uni;

	pms_creat_memory_with_paramete(&S2, Ny, Nx, Nf2);
	pms_creat_memory_with_paramete(&K3, Nx, Nf);
	pms_creat_memory_with_paramete(&K4, Nx, Nf2);
	pms_creat_memory_with_paramete(&S4, Nx, Nf);
	pms_creat_memory_with_paramete(&S6, Nx, Nf2);
	pms_creat_memory_with_paramete(&K3_uni, Nx, Nf);
	pms_creat_memory_with_paramete(&K4_uni, Nx, Nf2);

	double dk1;
	dk1 = ((*(K.pointer + pms_offset_address_3d(0, 0, Nf - 1, &K))) - (*(K.pointer + pms_offset_address_3d(0, 0, 0, &K)))) / (Nf - 1);

	double k111 = *(K.pointer);

	pms_permute_self(&K, 2, 1, 3);
	pms_permute_self(&K2, 2, 1, 3);
	pms_permute_self(&S_ftxy, 2,1,3);

	_dpmscl z3;
	pms_creat_memory_with_paramete(&z3, Nf2);
	_dpmscl z4;
	pms_creat_memory_with_paramete(&z4, Nf2);

	_dpmscl xs;

	pms_creat_memory_with_paramete(&xs, Ny * Nx * Nf2);
	pms_clear_dpmscl(&xs);

	_dpmscl xs_length;

	pms_creat_memory_with_paramete(&xs_length, Ny * Nx);
	pms_clear_dpmscl(&xs_length);

	double K3_uni_Nf;
	double K3_uni_1;
	int zz3;
	int zz4;

	_ipms zz4_record;
	_ipms zz3_record;

	pms_creat_memory_with_paramete(&zz3_record, Ny, Nx);
	pms_creat_memory_with_paramete(&zz4_record, Ny, Nx);

	for (index_y = 0; index_y < Ny; index_y++)
	{
		memcpy(K3.pointer, K.pointer + pms_offset_address_3d(index_y, 0, 0, &K), K3.length * sizeof(double));
		memcpy(K4.pointer, K2.pointer + pms_offset_address_3d(index_y, 0, 0, &K2), K4.length * sizeof(double));


		for (index = 0; index < K3.length; index++)
		{
			*(K3_uni.pointer + index) = ((*(K3.pointer + index) - k111) / dk1) + 1;
		}
		for (index = 0; index < K4.length; index++)
		{
			*(K4_uni.pointer + index) = ((*(K4.pointer + index) - k111) / dk1) + 1;
			if (abs(1 - (*(K4_uni.pointer + index))) < 0.00000001)
			{
				*(K4_uni.pointer + index) = 1;
			}
		}

		for (index_x = 0; index_x < Nx; index_x++)
		{
			//[x3,z3]=find(K4_uni(j,:)>K3_uni(j,Nf));
			pms_clear_dpmscl(&z3);
			K3_uni_Nf = (*(K3_uni.pointer + pms_offset_address_2d(index_x, Nf - 1, &K3_uni)));
			for (index = 0; index < K4_uni.number_of_y; index++)
			{
				if ((*(K4_uni.pointer + pms_offset_address_2d(index_x, index, &K4_uni))) > K3_uni_Nf)
				{
					pms_add_element_array_dpmscl(&z3, index);
				}
			}

			if (z3.enable_length == 0)
			{
				pms_add_element_array_dpmscl(&z3, (double)Nf2);
			}
			zz3 = pms_find_min_allover(&z3);
			*(zz3_record.pointer + pms_offset_address_2d(index_y, index_x, &zz3_record)) = zz3;
			//[x4,z4]=find(K4_uni(j,:)<K3_uni(j,1));
			pms_clear_dpmscl(&z4);
			K3_uni_1 = (*(K3_uni.pointer + pms_offset_address_2d(index_x, 0, &K3_uni)));
			for (index = 0; index < K4_uni.number_of_y; index++)
			{
				if ((*(K4_uni.pointer + pms_offset_address_2d(index_x, index, &K4_uni))) < K3_uni_1)
				{
					pms_add_element_array_dpmscl(&z4, index);
				}
			}

			if (z4.enable_length == 0)
			{
				pms_add_element_array_dpmscl(&z4, -1);
			}
			zz4 = pms_find_max_allover(&z4);
			*(zz4_record.pointer + pms_offset_address_2d(index_y, index_x, &zz4_record)) = zz4;
			pms_add_array_dpmscl(&xs, &K4_uni, zz4 + 1 + K4_uni.number_of_y * index_x, zz3 - 1 + K4_uni.number_of_y * index_x);
			pms_add_element_array_dpmscl(&xs_length, zz3 - zz4 - 1);

		}

	}

	_dpms in_x;
	_dpms S_ftxy_real_part;
	_dpms S_ftxy_img_part;

	_dpms ys;

	_dpms xs2;
	_dpms ys2;

	//output_data(&S_ftxy, "D:/temp/S_ftxy.bin");

	pms_creat_memory_with_paramete(&in_x, Nf);
	pms_copy_info(&S_ftxy_real_part, &S_ftxy);
	pms_copy_info(&S_ftxy_img_part, &S_ftxy);

	pms_creat_memory(&S_ftxy_real_part);
	pms_creat_memory(&S_ftxy_img_part);
	pms_copy_fc_real(&S_ftxy_real_part, &S_ftxy);
	pms_copy_fc_imagine(&S_ftxy_img_part, &S_ftxy);

	pms_creat_memory_with_paramete(&ys, xs.length);
	pms_creat_memory_with_paramete(&xs2, xs.length);
	pms_creat_memory_with_paramete(&ys2, xs.length);

	for (index = 0; index < Nf; index++)
	{
		*(in_x.pointer + index) = *(K3_uni.pointer + index);
	}





	cudaError_t cudaStatus;

	_dpms dev_input_x;
	_dpms dev_input_y;
	_dpms dev_output_x;
	_dpms dev_output_y;

	cudaStatus = pms_creat_memory_with_paramete_cuda(&dev_input_x, LENGTH);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = pms_creat_memory_with_paramete_cuda(&dev_input_y, LENGTH);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	cudaStatus = pms_creat_memory_with_paramete_cuda(&dev_output_x, LENGTH * 2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	cudaStatus = pms_creat_memory_with_paramete_cuda(&dev_output_y, LENGTH * 2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	spline_plan splineplan;
	splineplan = creat_spline_plan_inout(&in_x, &S_ftxy_real_part, &xs, &ys);
	set_spline_plan_mode(splineplan, NATURAL_BOUNDARY, RUN_ON_GPU, MODE_SINX_MINY_MOUTX_MOUTY, &xs_length);
	execute_spline_plan(splineplan);

	/*spline_plan splineplan2;
	splineplan2 = creat_spline_plan_inout(&in_x, &S_ftxy_img_part, &xs, &ys2);
	set_spline_plan_mode(splineplan2, NATURAL_BOUNDARY, RUN_ON_GPU, MODE_SINX_MINY_MOUTX_MOUTY, &xs_length);*/
	creat_spline_plan_change_inout(splineplan,&in_x, &S_ftxy_img_part, &xs, &ys2);
	execute_spline_plan(splineplan);

	/*********************Data stitching********************/
	_fcpms ys_complex;
	int ys_offset=0;
	pms_creat_memory_with_paramete(&ys_complex, ys.length);
	
	
	//The real and imaginary parts are combined into a complex matrix
	for (index = 0; index < ys.length; index++)
	{
		(*(ys_complex.pointer + index))[0] = *(ys.pointer + index);
		(*(ys_complex.pointer + index))[1] = *(ys2.pointer + index);
	}
	//output_data(&ys_complex, "D:/TEMP/ys_complex.bin");
	
	for (index_y = 0; index_y < Ny; index_y++)
	{
		memset(S6.pointer, 0, S6.length*sizeof(fftw_complex));
		for (index_x = 0; index_x < Nx; index_x++)
		{
			zz4 = *(zz4_record.pointer + pms_offset_address_2d(index_y, index_x, &zz4_record)) + 1;
			memcpy(S6.pointer+ index_x*Nf2+zz4,
				ys_complex.pointer+ ys_offset,
				(*(xs_length.pointer+(index_y*Nx+index_x)))*sizeof(fftw_complex));
			ys_offset += *(xs_length.pointer + (index_y * Nx + index_x));
		}
		memcpy(S2.pointer + pms_offset_address_3d(index_y, 0, 0, &S2), S6.pointer, S6.length * sizeof(fftw_complex));

	}
	pms_permute_self(&S2, 2,1,3);
	//output_data(&S2, "D:/TEMP/S2.bin");
	//pms_permute_self(&S2, 3,2,1);
	//output_data(&S2,"D:/TEMP/S2.bin");
	//output_data(&zz3_record, "D:/TEMP/zz3.bin");
	//output_data(&zz4_record, "D:/TEMP/zz4.bin");

	/**************************************************************/
	_fcpms S_iftxyz;
	pms_copy_info(&S_iftxyz, &S2);
	pms_creat_memory(&S_iftxyz);
	fftshift3(S2.pointer, S2.number_of_x, S2.number_of_y, S2.number_of_z);

	fftw_plan i_plan;
	i_plan = fftw_plan_dft_3d(S2.number_of_x, S2.number_of_y, S2.number_of_z, S2.pointer, S_iftxyz.pointer, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(i_plan);

	for (index = 0; index < S_iftxyz.length; index++)
	{
		(*(S_iftxyz.pointer + index))[0] /= S_iftxyz.length;
		(*(S_iftxyz.pointer + index))[1] /= S_iftxyz.length;
	}

	fftshift3(S_iftxyz.pointer, S_iftxyz.number_of_x, S_iftxyz.number_of_y, S_iftxyz.number_of_z);
	output_data(&S_iftxyz, "D:/S_iftxyz.bin");
	return 0;
}

