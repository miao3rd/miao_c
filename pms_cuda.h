/*
*PMS为pointer manage struct简称
*pms结构体专为矩阵设计
*记录矩阵指针，长度，维度数以及各维度的长度
*配套了矩阵乘法，维度转换等函数
*可直接对pms结构体操作
*避免了手动操作各矩阵维度信息
*/
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/*******************************
头文件版本号：0.2a
相比CPU版本，增加了在显存中申请空间的函数
以及内存和
********************************/


#ifndef PMS_H_CUDA
#define PMS_H_CUDA

#define PMS_CUDA_MAJOR_VERSION 0
#define PMS_CUDA_MINOR_VERSION 2

/********************
宏函数功能：偏移地址计算

*********************/
#define OFFSET_ADDRESS_2D(x,y,yn) (x*yn+y)
#define OFFSET_ADDRESS_3D(x,y,z,yn,zn) (x*yn*zn+y*zn+z)

#define OAC_2 OFFSET_ADDRESS_2D
#define OAC_3 OFFSET_ADDRESS_3D

#define NO_DIMENSION 1

/********************
宏定义功能：错误类型

*********************/
#define ERROR_MATRIX_MULTIPLICATION_DIMENSION_MISMATCH -127
#define ERROR_PMS_DIMENSION_INVALID -126

/**************
矩阵指针管理结构体定义
*************/
struct int_pointer_manage_struct
{
	int* pointer;
	int dimension;
	int number_of_x;
	int number_of_y;
	int number_of_z;
	int length = number_of_x * number_of_y * number_of_z;
};


struct double_pointer_manage_struct
{
	double* pointer;
	int dimension;
	int number_of_x;
	int number_of_y;
	int number_of_z;
	int length = number_of_x * number_of_y * number_of_z;
};

struct float_pointer_manage_struct
{
	float* pointer;
	int dimension;
	int number_of_x;
	int number_of_y;
	int number_of_z;
	int length = number_of_x * number_of_y * number_of_z;
};



#ifdef FFTW3_H
struct fftw_complex_pointer_manage_struct
{
	fftw_complex* pointer;
	int dimension;
	int number_of_x;
	int number_of_y;
	int number_of_z;
	int length = number_of_x * number_of_y * number_of_z;
};
#endif

typedef struct int_pointer_manage_struct _ipms;
typedef struct double_pointer_manage_struct _dpms;
typedef struct float_pointer_manage_struct _fpms;

#ifdef FFTW3_H
typedef struct fftw_complex_pointer_manage_struct _fcpms;
#endif

/**********************
函数申明
**********************/
cudaError_t pms_creat_memory_cuda(_dpms* dpms);
cudaError_t pms_creat_memory_cuda(_fpms* fpms);



/***************************************
函数pms_set_parameter用于设置pms结构体参数
调用格式为
pms_set_parameter(&pms,x);
pms_set_parameter(&pms,x,y);
pms_set_parameter(&pms,x,y,z);
有几个参数就写几个
*****************************************/
void pms_set_parameter(_ipms* ipms, int number_of_x)
{
	ipms->dimension = 1;
	ipms->number_of_x = number_of_x;
	ipms->number_of_y = NO_DIMENSION;
	ipms->number_of_z = NO_DIMENSION;
	ipms->length = number_of_x;
}

void pms_set_parameter(_ipms* ipms, int number_of_x, int number_of_y)
{
	ipms->dimension = 2;
	ipms->number_of_x = number_of_x;
	ipms->number_of_y = number_of_y;
	ipms->number_of_z = NO_DIMENSION;
	ipms->length = number_of_x * number_of_y;
}

void pms_set_parameter(_ipms* ipms, int number_of_x, int number_of_y, int number_of_z)
{
	ipms->dimension = 3;
	ipms->number_of_x = number_of_x;
	ipms->number_of_y = number_of_y;
	ipms->number_of_z = number_of_z;
	ipms->length = number_of_x * number_of_y * number_of_z;
}

void pms_set_parameter(_fpms* fpms, int number_of_x)
{
	fpms->dimension = 1;
	fpms->number_of_x = number_of_x;
	fpms->number_of_y = NO_DIMENSION;
	fpms->number_of_z = NO_DIMENSION;
	fpms->length = number_of_x;
}

void pms_set_parameter(_fpms* fpms, int number_of_x, int number_of_y)
{
	fpms->dimension = 2;
	fpms->number_of_x = number_of_x;
	fpms->number_of_y = number_of_y;
	fpms->number_of_z = NO_DIMENSION;
	fpms->length = number_of_x * number_of_y;
}

void pms_set_parameter(_fpms* fpms, int number_of_x, int number_of_y, int number_of_z)
{
	fpms->dimension = 3;
	fpms->number_of_x = number_of_x;
	fpms->number_of_y = number_of_y;
	fpms->number_of_z = number_of_z;
	fpms->length = number_of_x * number_of_y * number_of_z;
}

void pms_set_parameter(_dpms* dpms, int number_of_x)
{
	dpms->dimension = 1;
	dpms->number_of_x = number_of_x;
	dpms->number_of_y = NO_DIMENSION;
	dpms->number_of_z = NO_DIMENSION;
	dpms->length = number_of_x;
}

void pms_set_parameter(_dpms* dpms, int number_of_x, int number_of_y)
{
	dpms->dimension = 2;
	dpms->number_of_x = number_of_x;
	dpms->number_of_y = number_of_y;
	dpms->number_of_z = NO_DIMENSION;
	dpms->length = number_of_x * number_of_y;
}

void pms_set_parameter(_dpms* dpms,int number_of_x,int number_of_y,int number_of_z)
{
	dpms->dimension = 3;
	dpms->number_of_x = number_of_x;
	dpms->number_of_y = number_of_y;
	dpms->number_of_z = number_of_z;
	dpms->length = number_of_x * number_of_y * number_of_z;
}

void pms_creat_memory_cuda_with_paramete_cuda(_fpms* fpms, int number_of_x)
{
	pms_set_parameter(fpms, number_of_x);
	pms_creat_memory_cuda(fpms);
}

void pms_creat_memory_cuda_with_paramete_cuda(_fpms* fpms, int number_of_x, int number_of_y)
{
	pms_set_parameter(fpms, number_of_x, number_of_y);
	pms_creat_memory_cuda(fpms);
}

void pms_creat_memory_cuda_with_paramete_cuda(_fpms* fpms, int number_of_x, int number_of_y, int number_of_z)
{
	pms_set_parameter(fpms, number_of_x, number_of_y, number_of_z);
	pms_creat_memory_cuda(fpms);
}


void pms_creat_memory_cuda_with_paramete_cuda(_dpms* dpms, int number_of_x)
{
	pms_set_parameter(dpms, number_of_x);
	pms_creat_memory_cuda(dpms);
}

void pms_creat_memory_cuda_with_paramete_cuda(_dpms* dpms, int number_of_x, int number_of_y)
{
	pms_set_parameter(dpms, number_of_x, number_of_y);
	pms_creat_memory_cuda(dpms);
}

void pms_creat_memory_cuda_with_paramete_cuda(_dpms* dpms, int number_of_x, int number_of_y, int number_of_z)
{
	pms_set_parameter(dpms, number_of_x, number_of_y, number_of_z);
	pms_creat_memory_cuda(dpms);
}


#ifdef FFTW3_H
void pms_set_parameter(_fcpms* fcpms, int number_of_x)
{
	fcpms->dimension = 1;
	fcpms->number_of_x = number_of_x;
	fcpms->number_of_y = NO_DIMENSION;
	fcpms->number_of_z = NO_DIMENSION;
	fcpms->length = number_of_x;
}

void pms_set_parameter(_fcpms* fcpms, int number_of_x, int number_of_y)
{
	fcpms->dimension = 2;
	fcpms->number_of_x = number_of_x;
	fcpms->number_of_y = number_of_y;
	fcpms->number_of_z = NO_DIMENSION;
	fcpms->length = number_of_x * number_of_y;
}

void pms_set_parameter(_fcpms* fcpms, int number_of_x, int number_of_y, int number_of_z)
{
	fcpms->dimension = 3;
	fcpms->number_of_x = number_of_x;
	fcpms->number_of_y = number_of_y;
	fcpms->number_of_z = number_of_z;
	fcpms->length = number_of_x * number_of_y * number_of_z;
}
#endif

/***************************************
函数pms_creat_memory用于申请内存
调用格式为
pms_creat_memory(&pms);
请确保pms内已经设置好参数
*****************************************/
void pms_creat_memory(_dpms* dpms)
{
	dpms->pointer = (double*)malloc(dpms->length * sizeof(double));
}

cudaError_t pms_creat_memory_cuda(_dpms* dpms)
{
	return cudaMalloc((void**)&(dpms->pointer), dpms->length * sizeof(double));
}

void pms_creat_memory(_fpms* fpms)
{
	fpms->pointer = (float*)malloc(fpms->length * sizeof(float));
}

cudaError_t pms_creat_memory_cuda(_fpms* fpms)
{
	return cudaMalloc((void**)&fpms->pointer, fpms->length * sizeof(float));
}

void pms_creat_memory(_ipms* ipms)
{
	ipms->pointer = (int*)malloc(ipms->length * sizeof(int));
}

cudaError_t pms_creat_memory_cuda(_ipms* ipms)
{
	return cudaMalloc((void**)&ipms->pointer, ipms->length * sizeof(int));
}


#ifdef FFTW3_H
void pms_creat_memory(_fcpms* fcpms)
{
	fcpms->pointer = (fftw_complex*)malloc(fcpms->length * sizeof(fftw_complex));
}
#endif
/***************************************
函数pms_creat_memory_with_paramete可以同时完成分配参数和申请内存
调用格式为
pms_creat_memory_with_paramete(&pms,x);
pms_creat_memory_with_paramete(&pms,x,y);
pms_creat_memory_with_paramete(&pms,x,y,z);
请确保pms内已经设置好参数
*****************************************/
void pms_creat_memory_with_paramete(_fpms* fpms, int number_of_x)
{
	pms_set_parameter(fpms, number_of_x);
	pms_creat_memory(fpms);
}

void pms_creat_memory_with_paramete(_fpms* fpms, int number_of_x, int number_of_y)
{
	pms_set_parameter(fpms, number_of_x, number_of_y);
	pms_creat_memory(fpms);
}

void pms_creat_memory_with_paramete(_fpms* fpms, int number_of_x, int number_of_y, int number_of_z)
{
	pms_set_parameter(fpms, number_of_x, number_of_y, number_of_z);
	pms_creat_memory(fpms);
}

void pms_creat_memory_with_paramete(_ipms* ipms, int number_of_x)
{
	pms_set_parameter(ipms, number_of_x);
	pms_creat_memory(ipms);
}

void pms_creat_memory_with_paramete(_ipms* ipms, int number_of_x, int number_of_y)
{
	pms_set_parameter(ipms, number_of_x, number_of_y);
	pms_creat_memory(ipms);
}

void pms_creat_memory_with_paramete(_ipms* ipms, int number_of_x, int number_of_y, int number_of_z)
{
	pms_set_parameter(ipms, number_of_x, number_of_y, number_of_z);
	pms_creat_memory(ipms);
}


void pms_creat_memory_with_paramete(_dpms* dpms, int number_of_x)
{
	pms_set_parameter(dpms, number_of_x);
	pms_creat_memory(dpms);
}

void pms_creat_memory_with_paramete(_dpms* dpms, int number_of_x, int number_of_y)
{
	pms_set_parameter(dpms, number_of_x, number_of_y);
	pms_creat_memory(dpms);
}

void pms_creat_memory_with_paramete(_dpms* dpms, int number_of_x, int number_of_y, int number_of_z)
{
	pms_set_parameter(dpms, number_of_x, number_of_y, number_of_z);
	pms_creat_memory(dpms);
}

void pms_creat_memory_with_paramete_cuda(_fpms* fpms, int number_of_x)
{
	pms_set_parameter(fpms, number_of_x);
	pms_creat_memory_cuda(fpms);
}

void pms_creat_memory_with_paramete_cuda(_fpms* fpms, int number_of_x, int number_of_y)
{
	pms_set_parameter(fpms, number_of_x, number_of_y);
	pms_creat_memory_cuda(fpms);
}

void pms_creat_memory_with_paramete_cuda(_fpms* fpms, int number_of_x, int number_of_y, int number_of_z)
{
	pms_set_parameter(fpms, number_of_x, number_of_y, number_of_z);
	pms_creat_memory_cuda(fpms);
}


cudaError_t  pms_creat_memory_with_paramete_cuda(_dpms* dpms, int number_of_x)
{
	pms_set_parameter(dpms, number_of_x);
	return pms_creat_memory_cuda(dpms);
}

void pms_creat_memory_with_paramete_cuda(_dpms* dpms, int number_of_x, int number_of_y)
{
	pms_set_parameter(dpms, number_of_x, number_of_y);
	pms_creat_memory_cuda(dpms);
}

void pms_creat_memory_with_paramete_cuda(_dpms* dpms, int number_of_x, int number_of_y, int number_of_z)
{
	pms_set_parameter(dpms, number_of_x, number_of_y, number_of_z);
	pms_creat_memory_cuda(dpms);
}


#ifdef FFTW3_H

void pms_creat_memory_with_paramete(_fcpms* fcpms, int number_of_x)
{
	pms_set_parameter(fcpms, number_of_x);
	pms_creat_memory(fcpms);
}

void pms_creat_memory_with_paramete(_fcpms* fcpms, int number_of_x, int number_of_y)
{
	pms_set_parameter(fcpms, number_of_x, number_of_y);
	pms_creat_memory(fcpms);
}

void pms_creat_memory_with_paramete(_fcpms* fcpms, int number_of_x, int number_of_y, int number_of_z)
{
	pms_set_parameter(fcpms, number_of_x, number_of_y, number_of_z);
	pms_creat_memory(fcpms);
}

#endif


void pms_free_memory(_dpms* dpms)
{
	if (dpms->pointer != NULL)
	{
		free(dpms->pointer);
	}
}

void pms_free_memory(_fpms* fpms)
{
	free(fpms->pointer);
}


cudaError_t pms_free_memory_cuda(_dpms* dpms)
{
	return cudaFree(dpms->pointer);
}

cudaError_t  pms_free_memory_cuda(_fpms* fpms)
{
	return cudaFree(fpms->pointer);
}

int inline pms_offset_address_2d(int x, int y, _dpms* dpms)
{
	return (x * (dpms->number_of_y) + y);
}

int inline pms_offset_address_2d(int x, int y, _ipms* ipms)
{
	return (x * (ipms->number_of_y) + y);
}


#ifdef FFTW3_H

int inline pms_offset_address_2d(int x, int y, _fcpms* fcpms)
{
	return (x * (fcpms->number_of_y) + y);
}

void pms_free_memory(_fcpms* fcpms)
{
	free(fcpms->pointer);
}
#endif

int inline pms_offset_address_3d(int x, int y, int z, _dpms* dpms)
{
	return (x * (dpms->number_of_y) * (dpms->number_of_z) + y * (dpms->number_of_z) + z);
}

#ifdef FFTW3_H
int inline pms_offset_address_3d(int x, int y, int z, _fcpms* dpms)
{
	return (x * (dpms->number_of_y) * (dpms->number_of_z) + y * (dpms->number_of_z) + z);
}
#endif

int inline numel(_dpms* dpms)
{
	return dpms->length;
}


#ifdef FFTW3_H
void transposition(fftw_complex* a, fftw_complex* b, int line, int column)
{
	int i;
	int j;
	for (i = 0; i < column; i++)
		for (j = 0; j < line; j++)
			memcpy(*(b + i * (__int64)line + j), *(a + j * (__int64)column + i), sizeof(fftw_complex));
}
#endif

/***************************
对应MATLAB中的linspace函数
start起始值
end结束值
num取点个数
out为数组指针
order为顺序（正序或倒序）
*****************************/
void linspace(double start, double end, int num, double* out, int order = 0)
{
	int i;
	double step = (end - start) / (num - 1);
	if (order)
	{
		start = end;
		step = -step;
	}
	out[0] = start;
	for (i = 1; i < num; i++)
		out[i] = out[i - 1] + step;
}


/********************************
matmul矩阵乘法

*********************************/
void matmul(double* a, double* b, double* c, int n, int m, int t)		//输入参数：c=a*b，m-行数，n-列数
{
	int i, j, k;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < t; j++)
		{
			*(c + i * (__int64)t + j) = 0;
			for (k = 0; k < m; k++)
				(*(c + i * (__int64)t + j)) += (*(a + i * (__int64)m + k)) * (*(b + k * (__int64)t + j));
		}
	}
}

void pms_matmul(_dpms* out, _dpms* in1, _dpms* in2)
{
	if (in1->number_of_y == in2->number_of_x)
	{
		matmul(in1->pointer, in2->pointer, out->pointer, in1->number_of_x, in2->number_of_x, in2->number_of_y);
	}
	else
	{
		printf("矩阵乘法有误");
		exit(ERROR_MATRIX_MULTIPLICATION_DIMENSION_MISMATCH);
	}
}

void matmul(int* a, int* b, int* c, int n, int m, int t)		//输入参数：c=a*b，m-行数，n-列数
{
	int i, j, k;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < t; j++)
		{
			*(c + i * (__int64)t + j) = 0;
			for (k = 0; k < m; k++)
				(*(c + i * (__int64)t + j)) += (*(a + i * (__int64)m + k)) * (*(b + k * (__int64)t + j));
			//printf("%lf ", *(c + i * t + j));
		}
		//printf("\n");
	}
}

void permute3(double* in, double* out, int number_of_x, __int64 number_of_y, __int64 number_of_z, int new_dimension_of_x, int new_dimension_of_y, int new_dimension_of_z)
{
	int i;
	int j;
	int k;
	int transport_matrix[3][3] = { 0 };
	int in_vector_order[3] = { 0,0,0 };
	int in_vector_number[3] = { number_of_x, number_of_y, number_of_z };
	int out_vector_order[3] = { 0,0,0 };
	int out_vector_number[3] = { 0,0,0 };
	transport_matrix[0][new_dimension_of_x - 1] = 1;
	transport_matrix[1][new_dimension_of_y - 1] = 1;
	transport_matrix[2][new_dimension_of_z - 1] = 1;
	matmul(*transport_matrix, in_vector_number, out_vector_number, 3, 3, 1);
	for (i = 0; i < number_of_x; i++)
		for (j = 0; j < number_of_y; j++)
			for (k = 0; k < number_of_z; k++)
			{
				int in_vector_order[3] = { i,j,k };
				matmul(*transport_matrix, in_vector_order, out_vector_order, 3, 3, 1);
				*(out + out_vector_order[0] * out_vector_number[1] * out_vector_number[2] + out_vector_order[1] * out_vector_number[2] + out_vector_order[2]) = *(in + i * number_of_y * number_of_z + j * number_of_z + k);
			}
}

/*
void permute3(double* in, double* out, int number_of_x, __int64 number_of_y, __int64 number_of_z, int new_dimension_of_x, int new_dimension_of_y, int new_dimension_of_z)
{
	int index1;
	int index2;
	int index3;

	switch (new_dimension_of_x)
	{
	case(2):
		if (new_dimension_of_y == 1)
		{

			for (index1 = 0; index1 < number_of_x; index1++)
			{
				for (index2 = 0; index2 < number_of_y; index2++)
				{
					for (index3 = 0; index3 < number_of_z; index3++)
					{
						*(out + index2 * number_of_x * number_of_z + index1 * number_of_z + index3) = *(in + index1 * number_of_y * number_of_z + index2 * number_of_z + index3);
					}
				}
			}
		}
		else
		{
			for (index1 = 0; index1 < number_of_x; index1++)
			{
				for (index2 = 0; index2 < number_of_y; index2++)
				{
					for (index3 = 0; index3 < number_of_z; index3++)
					{
						*(out + index3 * number_of_x * number_of_y + index1 * number_of_y + index2) = *(in + index1 * number_of_y * number_of_z + index2 * number_of_z + index3);
					}
				}
			}
		}
		break;

	case(3):
		if (new_dimension_of_y == 1)
		{

			for (index1 = 0; index1 < number_of_x; index1++)
			{
				for (index2 = 0; index2 < number_of_y; index2++)
				{
					for (index3 = 0; index3 < number_of_z; index3++)
					{
						*(out + index3 * number_of_x * number_of_y + index1 * number_of_y + index2) = *(in + index1 * number_of_y * number_of_z + index2 * number_of_z + index3);
					}
				}
			}
		}
		else
		{
			for (index1 = 0; index1 < number_of_x; index1++)
			{
				for (index2 = 0; index2 < number_of_y; index2++)
				{
					for (index3 = 0; index3 < number_of_z; index3++)
					{
						*(out + index3 * number_of_y * number_of_x + index2 * number_of_x + index1) = *(in + index1 * number_of_y * number_of_z + index2 * number_of_z + index3);
					}
				}
			}
		}
		break;


	}
}*/

void pms_copy_info(_dpms* pms_dst, _dpms* pms_rsc)
{
	pms_dst->length = pms_rsc->length;
	pms_dst->dimension = pms_rsc->dimension;
	pms_dst->number_of_x = pms_rsc->number_of_x;
	pms_dst->number_of_y = pms_rsc->number_of_y;
	pms_dst->number_of_z = pms_rsc->number_of_z;
}

#ifdef FFTW3_H
void pms_copy_info(_dpms* pms_dst, _fcpms* pms_rsc)
{
	pms_dst->length = pms_rsc->length;
	pms_dst->dimension = pms_rsc->dimension;
	pms_dst->number_of_x = pms_rsc->number_of_x;
	pms_dst->number_of_y = pms_rsc->number_of_y;
	pms_dst->number_of_z = pms_rsc->number_of_z;
}

void pms_copy_fc_real(_dpms* pms_dst, _fcpms* pms_rsc)
{
	int index;
	for (index = 0; index < pms_rsc->length; index++)
	{
		*(pms_dst->pointer + index) = (*(pms_rsc->pointer + index))[0];
	}
}


void pms_copy_fc_imagine(_dpms* pms_dst, _fcpms* pms_rsc)
{
	int index;
	for (index = 0; index < pms_rsc->length; index++)
	{
		*(pms_dst->pointer + index) = (*(pms_rsc->pointer + index))[1];
	}
}

#endif // FFTW3_H

void pms_copy(_dpms* pms_dst, _dpms* pms_rsc)
{
	if (pms_rsc->length == pms_dst->length)		//如果两者长度相等，则
	{
		if (pms_dst->pointer != NULL)
		{
			memcpy(pms_dst->pointer, pms_rsc->pointer, pms_rsc->length * sizeof(double));
		}
		else
		{
			pms_dst->length = pms_rsc->length;
			pms_dst->pointer = (double*)malloc(pms_rsc->length * sizeof(double));
		}
	}
	else
	{
		if (pms_dst->pointer != NULL)
		{
			pms_free_memory(pms_dst);
			pms_dst->pointer = (double*)malloc(pms_rsc->length * sizeof(double));
		}
		else
		{
			pms_dst->pointer = (double*)malloc(pms_rsc->length * sizeof(double));
		}
	}
	pms_copy_info(pms_dst, pms_rsc);
}


void pms_permute(_dpms* dpms_in, _dpms* dpms_out, int new_dimension_of_x, int new_dimension_of_y, int new_dimension_of_z)
{
	permute3(dpms_in->pointer, dpms_out->pointer, dpms_in->number_of_x, dpms_in->number_of_y, dpms_in->number_of_z, new_dimension_of_x, new_dimension_of_y, new_dimension_of_z);
	dpms_out->number_of_x = dpms_in->number_of_x * (new_dimension_of_x == 1) + dpms_in->number_of_y * (new_dimension_of_y == 1) + dpms_in->number_of_z * (new_dimension_of_z == 1);
	dpms_out->number_of_y = dpms_in->number_of_x * (new_dimension_of_x == 2) + dpms_in->number_of_y * (new_dimension_of_y == 2) + dpms_in->number_of_z * (new_dimension_of_z == 2);
	dpms_out->number_of_z = dpms_in->number_of_x * (new_dimension_of_x == 3) + dpms_in->number_of_y * (new_dimension_of_y == 3) + dpms_in->number_of_z * (new_dimension_of_z == 3);
}


void pms_permute_self(_dpms* dpms_in, int new_dimension_of_x, int new_dimension_of_y, int new_dimension_of_z)
{
	_dpms dpms_out;
	pms_copy_info(&dpms_out, dpms_in);
	pms_creat_memory(&dpms_out);
	pms_permute(dpms_in, &dpms_out, new_dimension_of_x, new_dimension_of_y, new_dimension_of_z);
	pms_copy(dpms_in, &dpms_out);
	pms_free_memory(&dpms_out);
}



#ifdef FFTW3_H

void permute3(fftw_complex* in, fftw_complex* out, int number_of_x, __int64 number_of_y, __int64 number_of_z, int new_dimension_of_x, int new_dimension_of_y, int new_dimension_of_z)
{
	int i;
	int j;
	int k;
	int transport_matrix[3][3] = { 0 };
	int in_vector_order[3] = { 0,0,0 };
	int in_vector_number[3] = { number_of_x, number_of_y, number_of_z };
	int out_vector_order[3] = { 0,0,0 };
	int out_vector_number[3] = { 0,0,0 };
	transport_matrix[0][new_dimension_of_x - 1] = 1;
	transport_matrix[1][new_dimension_of_y - 1] = 1;
	transport_matrix[2][new_dimension_of_z - 1] = 1;
	matmul(*transport_matrix, in_vector_number, out_vector_number, 3, 3, 1);
	//cout << out_vector_number[0] << endl << out_vector_number[1] << endl << out_vector_number[2] << endl;
	for (i = 0; i < number_of_x; i++)
		for (j = 0; j < number_of_y; j++)
			for (k = 0; k < number_of_z; k++)
			{
				int in_vector_order[3] = { i,j,k };
				matmul(*transport_matrix, in_vector_order, out_vector_order, 3, 3, 1);
				//cout << out_vector_order[0] << endl << out_vector_order[1] << endl << out_vector_order[2] << endl;
				memcpy(*(out + out_vector_order[0] * out_vector_number[1] * out_vector_number[2] + out_vector_order[1] * out_vector_number[2] + out_vector_order[2]), *(in + i * number_of_y * number_of_z + j * number_of_z + k), sizeof(fftw_complex));
			}
}
/*
void permute3(fftw_complex* in, fftw_complex* out, int number_of_x, __int64 number_of_y, __int64 number_of_z, int new_dimension_of_x, int new_dimension_of_y, int new_dimension_of_z)
{
	int index1;
	int index2;
	int index3;

	switch (new_dimension_of_x)
	{
	case(2):
		if (new_dimension_of_y == 1)
		{

			for (index1 = 0; index1 < number_of_x; index1++)
			{
				for (index2 = 0; index2 < number_of_y; index2++)
				{
					for (index3 = 0; index3 < number_of_z; index3++)
					{
						memcpy((out + index2 * number_of_x * number_of_z + index1 * number_of_z + index3),(in + index1 * number_of_y * number_of_z + index2 * number_of_z + index3),sizeof(fftw_complex));
					}
				}
			}
		}
		else
		{
			for (index1 = 0; index1 < number_of_x; index1++)
			{
				for (index2 = 0; index2 < number_of_y; index2++)
				{
					for (index3 = 0; index3 < number_of_z; index3++)
					{
						memcpy((out + index3 * number_of_x * number_of_y + index1 * number_of_y + index2),(in + index1 * number_of_y * number_of_z + index2 * number_of_z + index3), sizeof(fftw_complex));
					}
				}
			}
		}
		break;

	case(3):
		if (new_dimension_of_y == 1)
		{

			for (index1 = 0; index1 < number_of_x; index1++)
			{
				for (index2 = 0; index2 < number_of_y; index2++)
				{
					for (index3 = 0; index3 < number_of_z; index3++)
					{
						memcpy((out + index3 * number_of_x * number_of_y + index1 * number_of_y + index2) ,(in + index1 * number_of_y * number_of_z + index2 * number_of_z + index3), sizeof(fftw_complex));
					}
				}
			}
		}
		else
		{
			for (index1 = 0; index1 < number_of_x; index1++)
			{
				for (index2 = 0; index2 < number_of_y; index2++)
				{
					for (index3 = 0; index3 < number_of_z; index3++)
					{
						memcpy((out + index3 * number_of_y * number_of_x + index2 * number_of_x + index1) ,(in + index1 * number_of_y * number_of_z + index2 * number_of_z + index3), sizeof(fftw_complex));
					}
				}
			}
		}
		break;


	}
}
*/

void fftshift_core(fftw_complex* in, int const length, fftw_complex* temp)
{
	if (length % 2)
	{
		memcpy(*temp, *in, sizeof(fftw_complex) * (length / 2 + 1));
		memcpy(*in, *(in + length / 2 + 1), sizeof(fftw_complex) * length / 2);
		memcpy(*(in + length / 2), *temp, sizeof(fftw_complex) * (length / 2 + 1));
	}
	else
	{
		memcpy(*temp, *(in + length / 2), sizeof(fftw_complex) * length / 2);
		memcpy(*(in + length / 2), *in, sizeof(fftw_complex) * length / 2);
		memcpy(*in, *temp, sizeof(fftw_complex) * length / 2);
	}
}

void fftshift(fftw_complex* in, int const length)
{
	fftw_complex* temp = (fftw_complex*)malloc(length * sizeof(fftw_complex));
	fftshift_core(in, length, temp);
	free(temp);
}

void fftshift2(fftw_complex* in, int const line, int const column)
{
	int i;
	int j;

	fftw_complex* temp1 = (fftw_complex*)malloc(column * sizeof(fftw_complex));
	for (i = 0; i < line; i++)
	{
		fftshift_core((in + i * column), column, temp1);
	}
	free(temp1);
	fftw_complex* temp = (fftw_complex*)malloc((line / 2 + 1) * column * sizeof(fftw_complex));
	if (line % 2 == 1)
	{
		//奇数需分割矩阵
		memcpy(temp, in, (line + 1) / 2 * column * sizeof(fftw_complex));
		memcpy(in, in + (line + 1) / 2 * column, (line - 1) / 2 * column * sizeof(fftw_complex));
		memcpy(in + (line - 1) / 2 * column, temp, (line + 1) * column * sizeof(fftw_complex) / 2);
	}
	else
	{
		//偶数则对半交换
		memcpy(temp, in, line * column * sizeof(fftw_complex) / 2);
		memcpy(in, in + line * column / 2, line * column * sizeof(fftw_complex) / 2);
		memcpy(in + line * column / 2, temp, line * column * sizeof(fftw_complex) / 2);
	}

	free(temp);
}

void fftshift3(fftw_complex* in, int const number_of_x, int const number_of_y, int const number_of_z)
{
	int i;
	int j;
	int k;
	for (i = 0; i < number_of_x; i++)
	{
		fftshift2((in + i * number_of_y * number_of_z), number_of_y, number_of_z);
	}
	fftw_complex* temp;
	temp = (fftw_complex*)malloc((number_of_x + 1) / 2 * number_of_y * number_of_z * sizeof(fftw_complex));
	if (number_of_x % 2 == 1)
	{
		memcpy(temp, in, (number_of_x + 1) / 2 * number_of_y * number_of_z * sizeof(fftw_complex));
		memcpy(in, in + (number_of_x + 1) / 2 * number_of_y * number_of_z, (number_of_x - 1) / 2 * number_of_y * number_of_z * sizeof(fftw_complex));
		memcpy(in + (number_of_x - 1) / 2 * number_of_y * number_of_z, temp, (number_of_x + 1) / 2 * number_of_y * number_of_z * sizeof(fftw_complex));
	}
	else
	{
		memcpy(temp, in, number_of_x * number_of_y * number_of_z * sizeof(fftw_complex) / 2);
		memcpy(in, in + number_of_x * number_of_y * number_of_z / 2, number_of_x * number_of_y * number_of_z * sizeof(fftw_complex) / 2);
		memcpy(in + number_of_x * number_of_y * number_of_z / 2, temp, number_of_x * number_of_y * number_of_z * sizeof(fftw_complex) / 2);
	}
	free(temp);
}

void pms_fftshift(_fcpms* in)
{
	if (in->dimension == 1)
	{
		fftshift(in->pointer, in->number_of_x);
	}
	else
		if (in->dimension == 2)
		{
			fftshift2(in->pointer, in->number_of_x,in->number_of_y);
		}
		else
			if (in->dimension == 3)
			{
				fftshift3(in->pointer, in->number_of_x, in->number_of_y,in->number_of_z);
			}
			else
			{
				printf("Fftshist dimension error. Only 1-3 dimension can be fftshift.\n");
			}
}

void pms_copy_info(_fcpms* pms_dst, _fcpms* pms_rsc)
{
	pms_dst->length = pms_rsc->length;
	pms_dst->dimension = pms_rsc->dimension;
	pms_dst->number_of_x = pms_rsc->number_of_x;
	pms_dst->number_of_y = pms_rsc->number_of_y;
	pms_dst->number_of_z = pms_rsc->number_of_z;
}



void pms_copy(_fcpms* pms_dst, _fcpms* pms_rsc)
{
	if (pms_rsc->length == pms_dst->length)
	{
		if (pms_dst->pointer != NULL)
		{
			memcpy(pms_dst->pointer, pms_rsc->pointer, pms_rsc->length * sizeof(fftw_complex));
		}
		else
		{
			pms_dst->length = pms_rsc->length;
			pms_dst->pointer = (fftw_complex*)malloc(pms_rsc->length * sizeof(fftw_complex));
		}
	}
	else
	{
		if (pms_dst->pointer != NULL)
		{
			pms_free_memory(pms_dst);
			pms_dst->pointer = (fftw_complex*)malloc(pms_rsc->length * sizeof(fftw_complex));
		}
		else
		{
			pms_dst->pointer = (fftw_complex*)malloc(pms_rsc->length * sizeof(fftw_complex));
		}
	}
	pms_copy_info(pms_dst, pms_rsc);
}
#endif

#ifdef FFTW3_H
void pms_permute(_fcpms* fcpms_in, _fcpms* fcpms_out, int new_dimension_of_x, int new_dimension_of_y, int new_dimension_of_z)
{
	permute3(fcpms_in->pointer, fcpms_out->pointer, fcpms_in->number_of_x, fcpms_in->number_of_y, fcpms_in->number_of_z, new_dimension_of_x, new_dimension_of_y, new_dimension_of_z);
	fcpms_out->number_of_x = fcpms_in->number_of_x * (new_dimension_of_x == 1) + fcpms_in->number_of_y * (new_dimension_of_y == 1) + fcpms_in->number_of_z * (new_dimension_of_z == 1);
	fcpms_out->number_of_y = fcpms_in->number_of_x * (new_dimension_of_x == 2) + fcpms_in->number_of_y * (new_dimension_of_y == 2) + fcpms_in->number_of_z * (new_dimension_of_z == 2);
	fcpms_out->number_of_z = fcpms_in->number_of_x * (new_dimension_of_x == 3) + fcpms_in->number_of_y * (new_dimension_of_y == 3) + fcpms_in->number_of_z * (new_dimension_of_z == 3);
}


void pms_permute_self(_fcpms* fcpms_in, int new_dimension_of_x, int new_dimension_of_y, int new_dimension_of_z)
{
	_fcpms fcpms_out;
	pms_copy_info(&fcpms_out, fcpms_in);
	pms_creat_memory(&fcpms_out);
	pms_permute(fcpms_in, &fcpms_out, new_dimension_of_x, new_dimension_of_y, new_dimension_of_z);
	pms_copy(fcpms_in, &fcpms_out);
	pms_free_memory(&fcpms_out);
}


void repmat3(double* in, double* out, int number_of_x, int number_of_y, int number_of_z, int multiple_of_x, int multiple_of_y, int multiple_of_z)
{
	int i;
	int j;
	int k;
	for (i = 0; i < number_of_x * multiple_of_x; i++)
		for (j = 0; j < number_of_y * multiple_of_y; j++)
			for (k = 0; k < number_of_z * multiple_of_z; k++)
			{
				*(out + i * number_of_y * multiple_of_y * number_of_z * multiple_of_z + j * number_of_z * multiple_of_z + k) = *(in + (i % number_of_x) * number_of_y * number_of_z + (j % number_of_y) * number_of_z + (k % number_of_z));
			}
}

void pms_repmat(_dpms* in, _dpms* out, int multiple_of_x, int multiple_of_y, int multiple_of_z)
{
	repmat3(in->pointer, out->pointer, in->number_of_x, in->number_of_y, in->number_of_z, multiple_of_x, multiple_of_y, multiple_of_z);
}
/*
void output_data(_fcpms* fcpms, char* file_name)
{
	FILE* fp;
	int index;
	fp = fopen(file_name, "w");
	for (index = 0; index < fcpms->length; index++)
	{
		fprintf(fp, "%lf", (*(fcpms->pointer + index))[0]);
		if ((*(fcpms->pointer + index))[1] >= 0)
			fprintf(fp, "+");
		fprintf(fp, "%lfi;\n", (*(fcpms->pointer + index))[1]);
	}
	fclose(fp);
}




void output_data(_dpms* dpms, char* file_name)
{
	FILE* fp;
	int index;
	fp = fopen(file_name, "w");
	for (index = 0; index < dpms->length; index++)
	{
		fprintf(fp, "%lf", (*(dpms->pointer + index)));
	}
	fclose(fp);
}
*/

void output_data(_ipms* ipms, char* file_name)
{
	FILE* fp;
	int index;
	fp = fopen(file_name, "wb");
	fwrite(ipms->pointer, ipms->length * sizeof(int), 1, fp);
	fclose(fp);
}
void output_data(_dpms* dpms, char* file_name)
{
	FILE* fp;
	int index;
	fp = fopen(file_name, "wb");
	fwrite(dpms->pointer, dpms->length * sizeof(double), 1, fp);
	fclose(fp);
}

void output_data(_fcpms* fcpms, char* file_name)
{
	FILE* fp;
	int index;
	fp = fopen(file_name, "wb");
	fwrite(*(fcpms->pointer), fcpms->length * sizeof(fftw_complex), 1, fp);
	fclose(fp);
}

#endif	//!FFTW3_H

cudaError_t pms_memcpy_cuda(_dpms* dpms_dst,_dpms* dpms_src, enum cudaMemcpyKind kind)
{
	return cudaMemcpy(dpms_dst->pointer, dpms_src->pointer, dpms_src->length * sizeof(double), kind);
}

cudaError_t pms_memcpy_cuda(_fpms* fpms_dst,_fpms* fpms_src, enum cudaMemcpyKind kind)
{
	return cudaMemcpy(fpms_dst->pointer, fpms_src->pointer, fpms_src->length * sizeof(float), kind);
}



struct double_pointer_manage_struct_changing_length :double_pointer_manage_struct
{
	int enable_length_x;
	int enable_length_y;
	int enable_length_z;
	int enable_length;

};

typedef double_pointer_manage_struct_changing_length _dpmscl;

void pmscl_enable_area_init(_dpmscl* dpms)
{
	switch (dpms->dimension)
	{
	case 1:
		dpms->enable_length_x = 0;
		dpms->enable_length_y = NO_DIMENSION;
		dpms->enable_length_z = NO_DIMENSION;
		break;
	case 2:
		dpms->enable_length_x = 0;
		dpms->enable_length_y = 0;
		dpms->enable_length_z = NO_DIMENSION;
		break;
	case 3:
		dpms->enable_length_x = 0;
		dpms->enable_length_y = 0;
		dpms->enable_length_z = 0;
		break;
	default:
		printf("pms维度无效");
		exit(ERROR_PMS_DIMENSION_INVALID);

	}
	dpms->enable_length = dpms->enable_length_x * dpms->enable_length_y * dpms->enable_length_z;
	
}

void pms_add_element_array_dpmscl(_dpmscl* dpmscl, double element)
{
	*(dpmscl->pointer + dpmscl->enable_length) = element;
	dpmscl->enable_length++;
	dpmscl->enable_length_x++;
}


void pms_add_array_dpmscl(_dpmscl* dpmscl, _dpms* added,int start, int end)
{
	if (end > start)
	{
		memcpy((dpmscl->pointer + dpmscl->enable_length), (added->pointer + start), (end - start+1)*sizeof(double));
		dpmscl->enable_length += (end - start+1);
		dpmscl->enable_length_x += (end - start+1);
	}
	else
	{
		printf("pms_add_array_dpmscl起始位置高于结束位置");
	}
}

void pms_clear_dpmscl(_dpmscl* dpmscl)
{
	dpmscl->enable_length = 0;
	pmscl_enable_area_init(dpmscl);
}

void pms_creat_memory(_dpmscl* dpms)
{
	dpms->pointer = (double*)malloc(dpms->length * sizeof(double));
}

void pms_creat_memory_with_paramete(_dpmscl* dpms, int number_of_x)
{
	pms_set_parameter(dpms, number_of_x);
	pms_creat_memory(dpms);
}

void pms_creat_memory_with_paramete(_dpmscl* dpms, int number_of_x, int number_of_y)
{
	pms_set_parameter(dpms, number_of_x, number_of_y);
	pms_creat_memory(dpms);
}

void pms_creat_memory_with_paramete(_dpmscl* dpms, int number_of_x, int number_of_y, int number_of_z)
{
	pms_set_parameter(dpms, number_of_x, number_of_y, number_of_z);
	pms_creat_memory(dpms);
}

double pms_find_min_allover(_dpms* dpms)
{
	double min;
	int index;
	min = (*(dpms->pointer));
	for (index = 0; index < dpms->length; index++)
	{
		if (min > (*(dpms->pointer + index)))
		{
			min = (*(dpms->pointer + index));
		}
	}
	return min;
}


double pms_find_min_allover(_dpmscl* dpms)
{
	double min;
	int index;
	min = (*(dpms->pointer));
	for (index = 0; index < dpms->enable_length; index++)
	{
		if (min > (*(dpms->pointer + index)))
		{
			min = (*(dpms->pointer + index));
		}
	}
	return min;
}

int pms_find_min_postion_allover(_dpms* dpms)
{
	int min_index;
	int index;
	min_index = 0;
	for (index = 0; index < dpms->length; index++)
	{
		if ((*(dpms->pointer + min_index)) > (*(dpms->pointer + index)))
		{
			min_index = index;
		}
	}
	return min_index;
}


double pms_find_max_allover(_dpms* dpms)
{
	double max;
	int index;
	max = (*(dpms->pointer));
	for (index = 0; index < dpms->length; index++)
	{
		if (max < (*(dpms->pointer + index)))
		{
			max = (*(dpms->pointer + index));
		}
	}
	return max;
}


double pms_find_max_allover(_dpmscl* dpms)
{
	double max;
	int index;
	max = (*(dpms->pointer));
	for (index = 0; index < dpms->enable_length; index++)
	{
		if (max < (*(dpms->pointer + index)))
		{
			max = (*(dpms->pointer + index));
		}
	}
	return max;
}


int pms_find_max_postion_allover(_dpms* dpms)
{
	int max_index;
	int index;
	max_index = 0;
	for (index = 0; index < dpms->length; index++)
	{
		if ((*(dpms->pointer + max_index)) < (*(dpms->pointer + index)))
		{
			max_index = index;
		}
	}
	return max_index;
}


#endif // !PMS_H