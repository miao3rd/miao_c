#ifndef PMS_SPLINE_H_CUDA
#define PMS_SPLINE_H_CUDA

#include "pms_cuda.h"

#define NATURAL_BOUNDARY 1          //Natural boundary condition
#define NOT_A_KNOT_BOUNDARY 3       //Non-kinked boundary conditions (Procedure not included)

#define RUN_ON_CPU 1                
#define RUN_ON_GPU 2                

#define MAX_LENGTH 256             //Length of L, U array


#define MODE_SINX_MINY_SOUTX_MOUTY 1        //Mode definition, single input x multiple output y single input x multiple output y.
#define MODE_SINX_MINY_MOUTX_MOUTY 2        //Mode definition, single input x multiple output y multiple input x multiple output y.

#define N_THREAD 32

__constant__ double dev_con_L[MAX_LENGTH];
__constant__ double dev_con_U[MAX_LENGTH];
__constant__ double dev_con_h[MAX_LENGTH];

//handle
struct pms_spline_plan
{
    int origin_length;
    int new_length;

    int boundary_condition;
    int run_place;
    int inout_mode;
    int plan_memory_is_creat = 0;

    /*
    *Pointer in structure with dev_ prefix is GPU pointer
    *but the structure is in memory
    */
    _dpms* in_x;
    _dpms* in_y;
    _dpms* out_x;
    _dpms* out_y;
    _dpms* out_x_length;
    _dpms* out_x_offset_address;

    _dpms* dev_in_x;
    _dpms* dev_in_y;
    _dpms* dev_out_x;
    _dpms* dev_out_y;
    _dpms* dev_out_x_length;
    _dpms* dev_out_x_offset_address;

    /*Here the H,M,Y matrices are
    *H*M=Y
    *H is the coefficient matrix (tridiagonal), which is obtained from the difference of x.
    *The three diagonals are divided into H1, H2, and H3, from top right to bottom left.
    *M is the matrix to be solved.
    *Y is the matrix obtained from in_y.
    */

    _dpms* Dx;
    _dpms* dev_Dx;
    _dpms* H1;
    _dpms* H2;
    _dpms* H3;
    _dpms* M;
    _dpms* Y;
    _dpms* dev_Y;

    /*
    *LU decomposition of the resulting matrix
    *They are obtained by the catch-up method
    */
    _dpms* dev_L;
    _dpms* dev_U;

    _dpms* L;
    _dpms* U;

    _dpms* dev_Um;
    _dpms* dev_m;
    _dpms* dev_H3;


    _dpms* con_a;
    _dpms* con_b;
    _dpms* con_c;
    _dpms* con_d;

    _dpms* dev_coe_a;
    _dpms* dev_coe_b;
    _dpms* dev_coe_c;
    _dpms* dev_coe_d;
};

typedef struct pms_spline_plan* spline_plan;

//Handle parameter setting function
spline_plan creat_spline_plan_inout(_dpms* in_x, _dpms* in_y, _dpms* out_x, _dpms* out_y)
{
    spline_plan plan = (spline_plan)malloc(sizeof(pms_spline_plan));

    plan->in_x = in_x;
    plan->in_y = in_y;
    plan->out_x = out_x;
    plan->out_y = out_y;
    plan->plan_memory_is_creat = 0;
    return plan;
}

void creat_spline_plan_change_inout(spline_plan plan,_dpms* in_x, _dpms* in_y, _dpms* out_x, _dpms* out_y)
{
    //spline_plan plan = (spline_plan)malloc(sizeof(pms_spline_plan));

    plan->in_x = in_x;
    plan->in_y = in_y;
    plan->out_x = out_x;
    plan->out_y = out_y;
    
}

void set_spline_plan_mode(spline_plan plan, int boundary_condition, int run_place, int inout_mode, _dpms* out_x_length = NULL)
{
    plan->boundary_condition = boundary_condition;
    plan->run_place = run_place;
    plan->inout_mode = inout_mode;
    if (inout_mode == MODE_SINX_MINY_MOUTX_MOUTY)
    {
        plan->out_x_length = out_x_length;
    }
}


/**************************************
Function Name: execute_spline_plan_calc_H
Function: Calculate H matrix (tridiagonal matrix) in CPU
    Relates only to the input x
    Compute only once if x is constant

***************************************/
void execute_spline_plan_calc_H(_dpms* input_x, _dpms* step_length, _dpms* coefficient_matrix_diag_a, _dpms* coefficient_matrix_diag_b, _dpms* coefficient_matrix_diag_c)
{
    int index;
    for (index = 0; index < step_length->length; index++)
    {
        step_length->pointer[index] = input_x->pointer[index + 1] - input_x->pointer[index];
    }
    for (index = 1; index < step_length->length - 1; index++)
    {
        coefficient_matrix_diag_a->pointer[index] = step_length->pointer[index - 1];
        coefficient_matrix_diag_b->pointer[index] = (step_length->pointer[index - 1] + step_length->pointer[index]) * 2;
        coefficient_matrix_diag_c->pointer[index] = step_length->pointer[index];

    }
}

/**************************************
Function name: execute_spline_plan_calc_Y_kernel
Function: Calculate Y matrix in GPU

***************************************/
__global__ void execute_spline_plan_calc_Y_kernel(double* input_y_pointer, double* result_matrix_pointer, \
    double* step_length_pointer, int length, int total_thread)
{
    int index;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int off_set_address = id * length;

    if (id < total_thread)
    {
        for (index = 1 + off_set_address; index < length - 1 + off_set_address; index++)
        {
            result_matrix_pointer[index] = ((input_y_pointer[index + 1] - input_y_pointer[index]) / dev_con_h[index % length]\
                - (input_y_pointer[index] - input_y_pointer[index - 1]) / dev_con_h[index % length - 1]) * 6;

        }
    }

}

/**************************************
Function name: execute_spline_plan_calc_Y_gpu
Function: send y-matrix instruction to GPU and allocate resources

***************************************/
void execute_spline_plan_calc_Y_gpu(_dpms* input_y, _dpms* step_length, _dpms* result_matrix)
{
    int total_thread;
    int nblock;
    total_thread = input_y->length / step_length->length;
    nblock = total_thread / N_THREAD + 1;

    execute_spline_plan_calc_Y_kernel << <1,nblock, N_THREAD >> > (input_y->pointer, result_matrix->pointer, step_length->pointer, step_length->length, total_thread);
}

/**************************************
Function name: execute_spline_plan_set_H_boundary
Function: Set the boundary according to the boundary conditions

***************************************/
void execute_spline_plan_set_H_boundary(_dpms* step_length, _dpms* coefficient_matrix_diag_a, _dpms* coefficient_matrix_diag_b, _dpms* coefficient_matrix_diag_c, int boundary_conditions)
{
    switch (boundary_conditions)
    {
    case NATURAL_BOUNDARY:
        coefficient_matrix_diag_b->pointer[0] = 1;
        coefficient_matrix_diag_c->pointer[0] = 0;
        coefficient_matrix_diag_b->pointer[coefficient_matrix_diag_b->length - 1] = 0;
        coefficient_matrix_diag_a->pointer[coefficient_matrix_diag_a->length - 1] = 1;
    }
}

/**************************************
Function name: execute_spline_plan_calc_LU
Function: Calculate the L and U matrices in the CPU
    L and U matrices are unchanged when the input x is unchanged
    L and U matrices can be reused after calculating once

***************************************/
void execute_spline_plan_calc_LU(_dpms* coefficient_matrix_diag_a, _dpms* coefficient_matrix_diag_b, _dpms* coefficient_matrix_diag_c, \
    _dpms* coefficient_matrix_L, _dpms* coefficient_matrix_U)
{
    int index;
    int x_length = coefficient_matrix_diag_a->length - 1;
    coefficient_matrix_U->pointer[0] = coefficient_matrix_diag_b->pointer[0];
    coefficient_matrix_L->pointer[0] = coefficient_matrix_diag_a->pointer[x_length] / coefficient_matrix_U->pointer[x_length - 1];
    for (index = 1; index < x_length; index++)
    {
        coefficient_matrix_L->pointer[index] = coefficient_matrix_diag_a->pointer[index] / coefficient_matrix_U->pointer[index - 1];
        coefficient_matrix_U->pointer[index] = coefficient_matrix_diag_b->pointer[index]\
            - coefficient_matrix_diag_c->pointer[index - 1] * coefficient_matrix_L->pointer[index];
    }
}

/**************************************
Function name: execute_spline_plan_calc_Um_kernel
Function: Kernel function to calculate the product of two matrices U and m in GPU

***************************************/
__global__ void execute_spline_plan_calc_Um_kernel(double* coefficient_matrix_L_pointer, double* coefficient_matrix_Um_pointer, \
    double* result_matrix_pointer, int length, int total_thread)
{
    int index;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int off_set_address = id * length;

    if (id < total_thread)
    {
        coefficient_matrix_Um_pointer[off_set_address] = result_matrix_pointer[off_set_address];
        for (index = 1 + off_set_address; index < length - 1 + off_set_address; index++)
        {
            /*
            coefficient_matrix_Um_pointer[index] = result_matrix_pointer[index]\
                - coefficient_matrix_L_pointer[index] * coefficient_matrix_Um_pointer[index - 1];
            */
            coefficient_matrix_Um_pointer[index] = result_matrix_pointer[index]\
                - dev_con_L[index % length] * coefficient_matrix_Um_pointer[index - 1];
        }

    }




}

/**************************************
Function name: execute_spline_plan_calc_Um_kernel
Function: Kernel function to calculate the product of two matrices U and m in GPU

***************************************/
void execute_spline_plan_calc_Um_gpu(_dpms* coefficient_matrix_L, _dpms* coefficient_matrix_dev_L, _dpms* coefficient_matrix_Um, _dpms* result_matrix)
{


    int total_thread;
    int nblock;
    total_thread = result_matrix->length / coefficient_matrix_L->length;
    nblock = total_thread / N_THREAD + 1;

    execute_spline_plan_calc_Um_kernel << <nblock, N_THREAD >> > (coefficient_matrix_dev_L->pointer, coefficient_matrix_Um->pointer, \
        result_matrix->pointer, coefficient_matrix_L->length, total_thread);

}

/**************************************
Function name: execute_spline_plan_calc_m_kernel
Function: Calculate m matrix in GPU.

***************************************/
__global__ void execute_spline_plan_calc_m_kernel(double* coefficient_matrix_x_pointer, double* coefficient_matrix_Ux_pointer, \
    double* coefficient_matrix_diag_c_pointer, int length, int total_thread)
{
    int index;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int off_set_address = id * length;
    if (id < total_thread)
    {
        coefficient_matrix_x_pointer[length - 1 + off_set_address] = coefficient_matrix_Ux_pointer[length - 1 + off_set_address] / dev_con_U[length - 1];
        for (index = length - 2 + off_set_address; index >= off_set_address; index--)
        {
            coefficient_matrix_x_pointer[index] = (coefficient_matrix_Ux_pointer[index]\
                - coefficient_matrix_diag_c_pointer[index % length] * coefficient_matrix_x_pointer[index + 1])\
                / dev_con_U[index % length];
        }
    }

}

/**************************************
Function name: execute_spline_plan_calc_m_kernel
Function: Calculate m matrix in GPU.

***************************************/
void execute_spline_plan_calc_m_gpu(_dpms* coefficient_matrix_x, _dpms* coefficient_matrix_Um, \
    _dpms* coefficient_matrix_U, _dpms* coefficient_matrix_diag_c)
{

    int total_thread;
    int nblock;
    int length = coefficient_matrix_diag_c->length;
    total_thread = coefficient_matrix_Um->length / length;     //分配调用核心数的thread
    nblock = total_thread / N_THREAD + 1;


    //    cudaMemcpyToSymbol(dev_con_L,coefficient_matrix_L->pointer, coefficient_matrix_L->length*sizeof(double));
    execute_spline_plan_calc_m_kernel << <nblock, N_THREAD >> > (coefficient_matrix_x->pointer, coefficient_matrix_Um->pointer, \
        coefficient_matrix_diag_c->pointer, length, total_thread);
}

/**************************************
Function name: execute_spline_plan_calc_coefficient_abcd_kernel
Function: Calculate the four coefficient matrices a, b, c, d in GPU.

***************************************/
__global__ void execute_spline_plan_calc_coefficient_abcd_kernel(double* input_y_pointer, \
    double* coefficient_matrix_m_pointer, double* step_length_pointer, double* spline_coefficient_a_pointer, \
    double* spline_coefficient_b_pointer, double* spline_coefficient_c_pointer, double* spline_coefficient_d_pointer, int length, int total_thread)
{
    int index;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int off_set_address = id * length;
    double yi, yi2;
    double mi, mi2;
    double hi;
    if (id < total_thread)
    {
        for (index = off_set_address; index < length + off_set_address; index++)
        {
            yi = *(input_y_pointer + index);
            yi2 = *(input_y_pointer + index + 1);
            mi = *(coefficient_matrix_m_pointer + index);
            mi2 = *(coefficient_matrix_m_pointer + index + 1);
            hi = dev_con_h[index % length];
            //*(spline_coefficient_a_pointer + index) = yi;
            *(spline_coefficient_b_pointer + index) = (yi2 - yi) / hi - hi * mi / 2 - hi * (mi2 - mi) / 6;
            *(spline_coefficient_c_pointer + index) = mi / 2;
            *(spline_coefficient_d_pointer + index) = (mi2 - mi) / (6 * hi);

        }
    }


}

/**************************************
Function name: execute_spline_plan_calc_coefficient_abcd_kernel
Function: Calculate the four coefficient matrices a, b, c, d in GPU.

***************************************/
void execute_spline_plan_calc_coefficient_abcd_gpu(_dpms* dev_input_y, _dpms* dev_m, _dpms* dev_step_length, \
    _dpms* dev_coe_a, _dpms* dev_coe_b, _dpms* dev_coe_c, _dpms* dev_coe_d)
{
    int length = dev_step_length->length;


    int total_thread;
    int nblock;
    total_thread = dev_input_y->length / length;
    nblock = total_thread / N_THREAD + 1;


    //    cudaMemcpyToSymbol(dev_con_L,coefficient_matrix_L->pointer, coefficient_matrix_L->length*sizeof(double));
    execute_spline_plan_calc_coefficient_abcd_kernel << <nblock, N_THREAD >> > (dev_input_y->pointer, dev_m->pointer, dev_step_length->pointer, \
        dev_coe_a->pointer, dev_coe_b->pointer, dev_coe_c->pointer, dev_coe_d->pointer, length, total_thread);
}
/**************************************
Function name: execute_spline_plan_calc_coefficient_abcd_gpu
Function: CPU function that sends instructions to GPU to calculate the coefficient a, b, c, d.

***************************************/
__global__ void execute_spline_plan_calc_outputy_0101_kernel(double* spline_coefficient_a_pointer, double* spline_coefficient_b_pointer, \
    double* spline_coefficient_c_pointer, double* spline_coefficient_d_pointer, double* input_x_pointer, double* output_x_pointer, \
    double* output_y_pointer, int length, int new_length)
{
    int origin_count = 0;
    int new_count = 0;
    //    int offset_address = threadIdx.x * length;
    //    int new_offset_address = threadIdx.x * new_length;
    int index_o = (threadIdx.x + blockIdx.x * blockDim.x) * length;
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * new_length;
    __shared__ double diff_io;
    for (; origin_count < length; index_o++, origin_count++)
    {
        while (output_x_pointer[index % new_length] <= input_x_pointer[index_o % length] && new_count <= new_length)
        {
            diff_io = output_x_pointer[index % new_length] - input_x_pointer[index_o % length];
            output_y_pointer[index] = spline_coefficient_a_pointer[index_o] + \
                spline_coefficient_b_pointer[index_o] * diff_io + \
                spline_coefficient_c_pointer[index_o] * pow(diff_io, 2) + \
                spline_coefficient_d_pointer[index_o] * pow(diff_io, 3);
            index++;
            new_count++;
        }

    }
}

__global__ void execute_spline_plan_calc_outputy_0111_kernel(double* spline_coefficient_a_pointer, double* spline_coefficient_b_pointer, \
    double* spline_coefficient_c_pointer, double* spline_coefficient_d_pointer, double* input_x_pointer, double* output_x_pointer, \
    double* output_y_pointer, double* output_x_offset, int length,int total_thread)
{
    int origin_count = 0;
    int id = (threadIdx.x + blockIdx.x * blockDim.x);
    //    int offset_address = threadIdx.x * length;
    //    int new_offset_address = threadIdx.x * new_length;
    int index_o = id * length;
    int index = output_x_offset[id];
    int index_max = output_x_offset[id + 1];
    double diff_io;
    if (id < total_thread)
    {
        for (; origin_count < length; index_o++, origin_count++)
        {
            while (output_x_pointer[index] <= input_x_pointer[index_o % length + 1] && index < index_max)
            {
                diff_io = output_x_pointer[index] - input_x_pointer[index_o % length];
                output_y_pointer[index] = spline_coefficient_a_pointer[index_o] + \
                    spline_coefficient_b_pointer[index_o] * diff_io + \
                    spline_coefficient_c_pointer[index_o] * pow(diff_io, 2) + \
                    spline_coefficient_d_pointer[index_o] * pow(diff_io, 3);
                index++;
            }
        }
    }
    
}

void execute_spline_plan_calc_outputy_gpu_0101(_dpms* dev_coe_a, _dpms* dev_coe_b, _dpms* dev_coe_c, _dpms* dev_coe_d, \
    _dpms* dev_input_x, _dpms* dev_output_x, _dpms* dev_output_y)
{
    int nblock = 1;
    int ntread = dev_coe_a->length / dev_input_x->length;
    int nBlock;
    nBlock = ntread / N_THREAD +1;
    ntread = 256;
    execute_spline_plan_calc_outputy_0101_kernel << <nBlock, ntread >> > (dev_coe_a->pointer, dev_coe_b->pointer, dev_coe_c->pointer, dev_coe_d->pointer, \
        dev_input_x->pointer, dev_output_x->pointer, dev_output_y->pointer, dev_input_x->length, dev_output_x->length);

}

void execute_spline_plan_calc_outputy_gpu_0111(_dpms* dev_coe_a, _dpms* dev_coe_b, _dpms* dev_coe_c, _dpms* dev_coe_d, \
    _dpms* dev_input_x, _dpms* dev_output_x, _dpms* dev_output_y, _dpms* dev_output_x_offset)
{
    int total_thread;
    total_thread = dev_output_x_offset->length;
    int nBlock;
    nBlock = total_thread / N_THREAD +1;
    execute_spline_plan_calc_outputy_0111_kernel << <nBlock, N_THREAD >> > (dev_coe_a->pointer, dev_coe_b->pointer, dev_coe_c->pointer, dev_coe_d->pointer, \
        dev_input_x->pointer, dev_output_x->pointer, dev_output_y->pointer, dev_output_x_offset->pointer, dev_input_x->length, total_thread);

}

/*
cudaError_t spline_plan_LU_memcpy_(_dpms* L, _dpms* U, _dpms* dev_L, _dpms* dev_U)
{
    __shared__ double dev_Ln;
}
*/

/**************************************
Function name: execute_spline_plan_memory
Function: The space of each pms in the created plan

***************************************/
void execute_spline_plan_memory(spline_plan plan)
{
    cudaSetDevice(0);
    int length;
    length = plan->in_x->length;
    int mutl_length;
    mutl_length = plan->in_y->length;
    cudaError_t cudaStatue;

    if (plan->plan_memory_is_creat == 0)
    {
        plan->plan_memory_is_creat = 1;
    }
    else
    {
        return;
    }

    _dpms* Dx = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_Dx = (_dpms*)malloc(sizeof(_dpms));
    _dpms* H1 = (_dpms*)malloc(sizeof(_dpms));
    _dpms* H2 = (_dpms*)malloc(sizeof(_dpms));
    _dpms* H3 = (_dpms*)malloc(sizeof(_dpms));
    _dpms* M = (_dpms*)malloc(sizeof(_dpms));
    _dpms* Y = (_dpms*)malloc(sizeof(_dpms));

    _dpms* dev_in_x = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_in_y = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_out_x = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_out_y = (_dpms*)malloc(sizeof(_dpms));
    _dpms* out_x_offset_address;
    _dpms* dev_out_x_length;
    _dpms* dev_out_x_offset_address;

    if (plan->inout_mode == MODE_SINX_MINY_MOUTX_MOUTY)
    {

        out_x_offset_address = (_dpms*)malloc(sizeof(_dpms));
        dev_out_x_length = (_dpms*)malloc(sizeof(_dpms));
        dev_out_x_offset_address = (_dpms*)malloc(sizeof(_dpms));
    }

    _dpms* dev_Y = (_dpms*)malloc(sizeof(_dpms));
    _dpms* L = (_dpms*)malloc(sizeof(_dpms));
    _dpms* U = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_L = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_U = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_Um = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_m = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_H3 = (_dpms*)malloc(sizeof(_dpms));
    _dpms* con_a = (_dpms*)malloc(sizeof(_dpms));
    _dpms* con_b = (_dpms*)malloc(sizeof(_dpms));
    _dpms* con_c = (_dpms*)malloc(sizeof(_dpms));
    _dpms* con_d = (_dpms*)malloc(sizeof(_dpms));

    _dpms* dev_coe_a = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_coe_b = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_coe_c = (_dpms*)malloc(sizeof(_dpms));
    _dpms* dev_coe_d = (_dpms*)malloc(sizeof(_dpms));



    pms_creat_memory_with_paramete(H1, length);
    pms_creat_memory_with_paramete(H2, length);
    pms_creat_memory_with_paramete(H3, length);
    pms_creat_memory_with_paramete(M, length);
    pms_creat_memory_with_paramete(Dx, length);
    pms_creat_memory_with_paramete(Y, mutl_length);
    pms_creat_memory_with_paramete(L, length);
    pms_creat_memory_with_paramete(U, length);
    
    if (plan->inout_mode == MODE_SINX_MINY_MOUTX_MOUTY)
    {
        pms_creat_memory_with_paramete(out_x_offset_address, plan->out_x_length->length + 1);
        plan->out_x_offset_address = out_x_offset_address;
    }

    if (plan->run_place == RUN_ON_GPU)
    {
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_in_x, length);
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_out_x, plan->out_x->length);
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_in_y, mutl_length);
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_out_y, plan->out_y->length);
        if (plan->inout_mode == MODE_SINX_MINY_MOUTX_MOUTY)
        {
            cudaStatue = pms_creat_memory_with_paramete_cuda(dev_out_x_length, plan->out_x_length->length);
            cudaStatue = pms_creat_memory_with_paramete_cuda(dev_out_x_offset_address, plan->out_x_length->length + 1);
            plan->dev_out_x_length = dev_out_x_length;
            plan->dev_out_x_offset_address = dev_out_x_offset_address;
        }
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_m, mutl_length);
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_Dx, length);
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_Y, mutl_length);
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_H3, length);
        //cudaStatue = pms_creat_memory_with_paramete_cuda(dev_coe_a, mutl_length);
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_coe_b, mutl_length);
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_coe_c, mutl_length);
        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_coe_d, mutl_length);
        //        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_L, length);
        //        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_U, length);
        pms_set_parameter(dev_L, length);
        dev_L->pointer = dev_con_L;
        pms_set_parameter(dev_U, length);
        dev_U->pointer = dev_con_U;

        cudaStatue = pms_creat_memory_with_paramete_cuda(dev_Um, mutl_length);

        plan->dev_in_x = dev_in_x;
        plan->dev_out_x = dev_out_x;
        plan->dev_in_y = dev_in_y;
        plan->dev_out_y = dev_out_y;
        plan->dev_Dx = dev_Dx;
        plan->dev_Y = dev_Y;
        plan->dev_L = dev_L;
        plan->dev_U = dev_U;
        plan->dev_Um = dev_Um;
        plan->dev_m = dev_m;
        plan->dev_H3 = dev_H3;
        plan->dev_coe_a = dev_coe_a;
        plan->dev_coe_b = dev_coe_b;
        plan->dev_coe_c = dev_coe_c;
        plan->dev_coe_d = dev_coe_d;
    }

    plan->Y = Y;
    plan->H1 = H1;
    plan->H2 = H2;
    plan->H3 = H3;
    plan->M = M;
    plan->Dx = Dx;
    plan->L = L;
    plan->U = U;
}

void execute_spline_plan_calc_input_x_offset_address(_dpms* out_x_offset_address, _dpms* out_x_length)
{
    int index;
    out_x_offset_address->pointer[0] = 0;
    for (index = 0; index < out_x_length->length; index++)
    {
        out_x_offset_address->pointer[index + 1] = out_x_length->pointer[index] + out_x_offset_address->pointer[index];
    }
}

// Execute the function of plan. This is the main program for cubic spline interpolation
void execute_spline_plan(spline_plan plan)
{
    cudaError_t cudaStatus;

    cudaSetDevice(0);
    execute_spline_plan_memory(plan);        //Creates space for each pms structure in handle plan, both the space in the structure itself and the space pointed to by the pointer in the structure.
    execute_spline_plan_calc_H(plan->in_x, plan->Dx, plan->H1, plan->H2, plan->H3);         //Calculate the value of the H matrix (only once)
    execute_spline_plan_set_H_boundary(plan->Dx, plan->H1, plan->H2, plan->H3, plan->boundary_condition);        //Set boundary conditions
    execute_spline_plan_calc_LU(plan->H1, plan->H2, plan->H3, plan->L, plan->U);            //Calculate the value of the L, U matrix
//    spline_plan_LU_memcpy_();
    if (plan->run_place == RUN_ON_GPU)
    {
        //       pms_creat_memory_with_paramete_cuda((plan->dev_Dx), LENGTH);
        cudaStatus = cudaMemcpy((plan->dev_Dx->pointer), (plan->Dx->pointer), (plan->Dx->length) * sizeof(double), cudaMemcpyHostToDevice);

        cudaStatus = cudaMemcpyToSymbol(dev_con_h, plan->Dx->pointer, plan->Dx->length * sizeof(double));
        plan->dev_Dx->pointer = dev_con_h;

        //       cudaStatus=pms_memcpy_cuda(plan->dev_Dx, plan->Dx, cudaMemcpyHostToDevice);
        pms_memcpy_cuda(plan->dev_in_x, plan->in_x, cudaMemcpyHostToDevice);
        pms_memcpy_cuda(plan->dev_out_x, plan->out_x, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }
        pms_memcpy_cuda(plan->dev_in_y, plan->in_y, cudaMemcpyHostToDevice);
        execute_spline_plan_calc_Y_gpu(plan->dev_in_y, plan->dev_Dx, plan->dev_Y);
        cudaStatus = cudaGetLastError();
        //Transferring L to constant memory
        cudaStatus = cudaMemcpyToSymbol(dev_con_L, plan->L->pointer, plan->L->length * sizeof(double));
        cudaStatus = cudaMemcpyToSymbol(dev_con_U, plan->U->pointer, plan->U->length * sizeof(double));

        execute_spline_plan_calc_Um_gpu(plan->L, plan->dev_L, plan->dev_Um, plan->dev_Y);
        cudaStatus = cudaGetLastError();
        pms_memcpy_cuda(plan->dev_H3, plan->H3, cudaMemcpyHostToDevice);
        execute_spline_plan_calc_m_gpu(plan->dev_m, plan->dev_Um, plan->dev_U, plan->dev_H3);
        cudaStatus = cudaGetLastError();
        execute_spline_plan_calc_coefficient_abcd_gpu(plan->dev_in_y, plan->dev_m, plan->dev_Dx, plan->dev_coe_a, plan->dev_coe_b, plan->dev_coe_c, plan->dev_coe_d);
        cudaStatus = cudaGetLastError();
        switch (plan->inout_mode)
        {
        case MODE_SINX_MINY_SOUTX_MOUTY:
            execute_spline_plan_calc_outputy_gpu_0101(plan->dev_coe_a, plan->dev_coe_b, plan->dev_coe_c, plan->dev_coe_d, plan->dev_in_x, plan->dev_out_x, plan->dev_out_y);
            break;
        case MODE_SINX_MINY_MOUTX_MOUTY:
            execute_spline_plan_calc_input_x_offset_address(plan->out_x_offset_address, plan->out_x_length);
            cudaStatus = pms_memcpy_cuda(plan->dev_out_x_offset_address, plan->out_x_offset_address, cudaMemcpyHostToDevice);
            
            //execute_spline_plan_calc_outputy_gpu_0111(plan->dev_coe_a, plan->dev_coe_b, plan->dev_coe_c, plan->dev_coe_d, plan->dev_in_x, plan->dev_out_x, plan->dev_out_y, plan->dev_out_x_offset_address);
            execute_spline_plan_calc_outputy_gpu_0111(plan->dev_in_y, plan->dev_coe_b, plan->dev_coe_c, plan->dev_coe_d, plan->dev_in_x, plan->dev_out_x, plan->dev_out_y, plan->dev_out_x_offset_address);
            cudaStatus = cudaGetLastError();
            break;

        }


        pms_memcpy_cuda(plan->out_y, plan->dev_out_y, cudaMemcpyDeviceToHost);
        /*pms_memcpy_cuda(plan->Y, plan->dev_coe_a, cudaMemcpyDeviceToHost);
        pms_memcpy_cuda(plan->Y, plan->dev_coe_b, cudaMemcpyDeviceToHost);
        pms_memcpy_cuda(plan->Y, plan->dev_coe_c, cudaMemcpyDeviceToHost);
        pms_memcpy_cuda(plan->Y, plan->dev_coe_d, cudaMemcpyDeviceToHost);
        pms_memcpy_cuda(plan->Y, plan->dev_m, cudaMemcpyDeviceToHost);
        pms_memcpy_cuda(plan->Y, plan->dev_Um, cudaMemcpyDeviceToHost);
        pms_memcpy_cuda(plan->Y, plan->dev_Y, cudaMemcpyDeviceToHost);*/
    }


}

__global__ void change_spline_plan_y_pointer(spline_plan plan, _dpms* in_y, _dpms* out_y)
{
    plan->in_y->pointer = in_y->pointer;
}

void destory_spline_plan(spline_plan plan)
{
    cudaSetDevice(0);
    cudaError_t cudaStatue;

    pms_free_memory(plan->H1);
    pms_free_memory(plan->H2);
    pms_free_memory(plan->H3);
    pms_free_memory(plan->M);
    pms_free_memory(plan->Dx);
    pms_free_memory(plan->Y);
    pms_free_memory(plan->con_a);
    pms_free_memory(plan->con_b);
    pms_free_memory(plan->con_c);
    pms_free_memory(plan->con_d);

    //pms_free_memory_cuda(plan->dev_Dx);

    if (plan->run_place == RUN_ON_GPU)
    {
        cudaStatue = pms_free_memory_cuda(plan->dev_in_x);
        cudaStatue = pms_free_memory_cuda(plan->dev_out_x);
        cudaStatue = pms_free_memory_cuda(plan->dev_in_y);
        cudaStatue = pms_free_memory_cuda(plan->dev_out_y);
        cudaStatue = pms_free_memory_cuda(plan->dev_m);
        cudaStatue = pms_free_memory_cuda(plan->dev_Dx);
        cudaStatue = pms_free_memory_cuda(plan->dev_Y);
        cudaStatue = pms_free_memory_cuda(plan->dev_H3);
        cudaStatue = pms_free_memory_cuda(plan->dev_coe_a);
        cudaStatue = pms_free_memory_cuda(plan->dev_coe_b);
        cudaStatue = pms_free_memory_cuda(plan->dev_coe_c);
        cudaStatue = pms_free_memory_cuda(plan->dev_coe_d);
        pms_free_memory_cuda(plan->dev_Um);

    }
}


#endif