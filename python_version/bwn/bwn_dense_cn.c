/*
* MIT License
* 
* Copyright (c) 2019 UCLA NanoCAD Laboratory 
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

/*!
 * \file      bwn_dense_cn.c
 * \brief     Dense binary-weight convolutional layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#include "bwn_dense_cn.h"
#define NULL ((void*)0)
/**
 * @details Dense binary-weight Convolutional (CN) layer with output binarization.
 * Pooling/padding support
 * 
 * @param[in] pAct - pointer to the packed activation vector (depth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels
 * @param[in] pad  - padding size 
 * @param[in] pool - pooling window size 
 * @param[out] pOut - pointer to the packed output vector (depth-width-height)
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnBwn(float* __restrict pAct, float * __restrict pKrn, const uint16_t dpth, 
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, 
    const uint16_t khgt, const uint16_t knum, const uint16_t pad, const uint16_t pool, 
    bnDtype * __restrict pOut, bnDtype* __restrict bias, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {

   float     outTemp = 0;
   int32_t  outTempInt = 0;
   int  yCoeff  = wdth*dpth;
   int  xCoeff  = dpth;
   int  kCoeff  = khgt*kwdt*kdpt;
   int  kyCoeff = kwdt*kdpt;
   int  kxCoeff = kdpt;
   // Starting indices for padding
   int  xStart, yStart = 0;
   int  xxStart, yyStart = 0;
   // Ending indices for padding
   int  xEnd, yEnd = 0;
   int  xxEnd, yyEnd = 0;
   bnDtype  *means = mean;
   bnDtype   *vars = var;
   bnDtype* gammas = gamma;
   bnDtype* betas = beta;
   float   maxTemp = 0;
   float   convAct=0;
   int  xyCount = 0;

 
   // Y dim
   for (int y = 0; y < (hght-khgt+2*pad+1)/pool; y++) {
      // Account for padding - skip padded values
      // X dim
      for (int x = 0; x < (wdth-kwdt+2*pad+1)/pool; x++) {
         // Account for padding - skip padded values
         // Restart kernel bn pointer
         //threshLoc = thresh;
         //signs = sign;
          means = mean;
          vars = var;
          gammas = gamma;
          betas = beta;
         // Outer loop - kernels
         for (int k = 0; k<knum; k++) {
            // Packed slices
            pOut[y * ((wdth - kwdt + 2 * pad + 1) / pool) * knum + x * knum + k] = 0;
            maxTemp = -INFINITY;
            for (int yy = 0; yy < pool; yy++) {
                if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                for (int xx = 0; xx < pool; xx++) {
                   if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                   if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                   outTemp = 0;
                   xyCount = 0;
                   // K-Y dim
                   for (int ky = yStart; ky < yEnd; ky++) {
                      // K-X dim
                      for (int kx = xStart; kx < xEnd; kx++) {
                         // Z dim
                         for (int z = 0; z < dpth; z++) {
                            convAct = pAct[(y*pool+yy+ky-pad)*yCoeff + (x*pool+xx+kx-pad)*xCoeff + z];
                            outTemp += pKrn[k * kCoeff + ky * kyCoeff + kx * kxCoeff + z] * convAct;
                         } // Z
                      } // K-X
                   }// K-Y
                   if (outTemp > maxTemp) { maxTemp = outTemp;}
                }
             }
             // Batch normalize/ binarize
            if (gammas != NULL) {
                pOut[y * ((wdth - kwdt + 2 * pad + 1) / pool) * knum  + x * knum + k] = (float)*gammas++ * (((bnPrec)maxTemp+bias[k] - *means++) / (*vars++)) + *betas++;
            }
            else {
                pOut[y * ((wdth - kwdt + 2 * pad + 1) / pool) * knum + x * knum + k] = maxTemp+bias[k];
            }
             
         }
      }
   }
}

           
void fadd(float* weight, float* bias, int x, int y, int z, float* out) {
    int i = 0;
    int j = 0;
    int k = 0;
    int size = y;
    if (z == 0) {
        for (i = 0; i != size; i += 1) {
            out[i] = weight[i] + bias[i];
        }
    }
    else {
        for (i = 0; i != y; i += 1) {
            for (j = 0; j != x; j += 1) {
                for (k = 0; k != z; k += 1) {
                    out[i * x * z + j * z + k] += bias[k];
                }
            }
        }
    }

}
void frelu(int x, int y, int z, float* weight,float* output) {
    int size = x * y * z;
    if (size == 0) size = y * x;
    int i = 0;
    for (i = 0; i != size; i += 1) {
        output[i]=weight[i] > 0 ? weight[i] : 0;
    }
}
void fmul(float* input, int size, int number, float* output) {
    int i = 0;
    for (i = 0; i != size; i += 1) {
        output[i] = number * input[i];
    }
}

void freshape(float* input, int x, int y, int z, float* output) {
    int i = 0;
    int j = 0;
    int k = 0;
    for (i = 0; i != z; i += 1) {
        for (j = 0; j != y; j += 1) {
            for (k = 0; k != x; k += 1) {
                output[i * x * y + j * x + k] = input[j * x * z + k * z + i];
            }
        }
    }
}
