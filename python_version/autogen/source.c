#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include "datatypes.h"
#include "utils.h"
#include "xnor_base.h"
#include "xnor_fc.h"
#include "xnor_fc.h"
#include "bwn_dense_cn.h"
#define NULL ((void*)0) 
#include "classifier_bias.h" 
#include "features_0_weight.h" 
#include "features_0_bias.h" 
#include "features_1_running_mean.h" 
#include "features_1_running_var.h" 
#include "features_1_bias.h" 
#include "features_1_weight.h" 
#include "features_4_weight.h" 
#include "features_4_bias.h" 
#include "features_5_running_mean.h" 
#include "features_5_running_var.h" 
#include "features_5_bias.h" 
#include "features_5_weight.h" 
#include "features_8_weight.h" 
#include "features_8_bias.h" 
#include "features_9_running_mean.h" 
#include "features_9_running_var.h" 
#include "features_9_bias.h" 
#include "features_9_weight.h" 
#include "features_11_weight.h" 
#include "features_11_bias.h" 
#include "features_12_running_mean.h" 
#include "features_12_running_var.h" 
#include "features_12_bias.h" 
#include "features_12_weight.h" 
#include "features_15_weight.h" 
#include "features_15_bias.h" 
#include "features_16_running_mean.h" 
#include "features_16_running_var.h" 
#include "features_16_bias.h" 
#include "features_16_weight.h" 
#include "features_18_weight.h" 
#include "features_18_bias.h" 
#include "features_19_running_mean.h" 
#include "features_19_running_var.h" 
#include "features_19_bias.h" 
#include "features_19_weight.h" 
#include "features_22_weight.h" 
#include "features_22_bias.h" 
#include "features_23_running_mean.h" 
#include "features_23_running_var.h" 
#include "features_23_bias.h" 
#include "features_23_weight.h" 
#include "features_25_weight.h" 
#include "features_25_bias.h" 
#include "features_26_running_mean.h" 
#include "features_26_running_var.h" 
#include "features_26_bias.h" 
#include "features_26_weight.h" 
#include "classifier_weight.h" 
#include "image.h"
static uint16_t   labels[] = LABELS; 
static float variable_49[] = features_9_weight; 
static float variable_48[] = features_9_running_var; 
static float variable_47[] = features_9_running_mean; 
static float variable_46[] = features_9_bias; 
static float variable_45[] = features_8_weight; 
static float variable_44[] = features_8_bias; 
static float variable_43[] = features_5_weight; 
static float variable_42[] = features_5_running_var; 
static float variable_41[] = features_5_running_mean; 
static float variable_40[] = features_5_bias; 
static float variable_39[] = features_4_weight; 
static float variable_38[] = features_4_bias; 
static float variable_37[] = features_26_weight; 
static float variable_36[] = features_26_running_var; 
static float variable_35[] = features_26_running_mean; 
static float variable_34[] = features_26_bias; 
static float variable_33[] = features_25_weight; 
static float variable_32[] = features_25_bias; 
static float variable_31[] = features_23_weight; 
static float variable_30[] = features_23_running_var; 
static float variable_29[] = features_23_running_mean; 
static float variable_28[] = features_23_bias; 
static float variable_27[] = features_22_weight; 
static float variable_26[] = features_22_bias; 
static float variable_25[] = features_19_weight; 
static float variable_24[] = features_19_running_var; 
static float variable_23[] = features_19_running_mean; 
static float variable_22[] = features_19_bias; 
static float variable_21[] = features_18_weight; 
static float variable_20[] = features_18_bias; 
static float variable_19[] = features_16_weight; 
static float variable_18[] = features_16_running_var; 
static float variable_17[] = features_16_running_mean; 
static float variable_16[] = features_16_bias; 
static float variable_15[] = features_15_weight; 
static float variable_14[] = features_15_bias; 
static float variable_13[] = features_12_weight; 
static float variable_12[] = features_12_running_var; 
static float variable_11[] = features_12_running_mean; 
static float variable_10[] = features_12_bias; 
static float variable_9[] = features_11_weight; 
static float variable_8[] = features_11_bias; 
static float variable_7[] = features_1_weight; 
static float variable_6[] = features_1_running_var; 
static float variable_5[] = features_1_running_mean; 
static float variable_4[] = features_1_bias; 
static float variable_3[] = features_0_weight; 
static float variable_2[] = features_0_bias; 
static float variable_1[] = classifier_weight; 
static float variable[] = classifier_bias; 
static float input[] = IMAGES ; 
static float mul_1[10]; 
static float* unsqueeze=mul_1; 
static float max_pool[16384]; 
static float max_pool_1[8192]; 
static float relu_2[16384]; 
static float max_pool_2[4096]; 
static float relu_4[8192]; 
static float max_pool_3[2048]; 
static float relu_6[2048]; 
static float reshape[512]; 
static float max_pool_4[512]; 
static float matmul[10]; 
static float mul[10]; 
static float output[10]; 
int main(){ 
	int correct = 0; 
	for(int img = 0; img < 100; img++) {
		float *curr_im = input + img*32*32*3;
		 fmul(variable, 10,1.0,mul_1); 
		unsqueeze=mul_1; 
		 CnBnBwn(curr_im,variable_3,3,32,32,3,3,3,64,1,2,max_pool, variable_2,variable_5, variable_6,variable_7,variable_4); 
		 frelu(64,16,16,max_pool,max_pool); 
		 CnBnBwn(max_pool,variable_39,64,16,16,64,3,3,128,1,2,max_pool_1, variable_38,variable_41, variable_42,variable_43,variable_40); 
		 frelu(128,8,8,max_pool_1,max_pool_1); 
		 CnBnBwn(max_pool_1,variable_45,128,8,8,128,3,3,256,1,1,relu_2, variable_44,variable_47, variable_48,variable_49,variable_46); 
		 frelu(256,8,8,relu_2,relu_2); 
		 CnBnBwn(relu_2,variable_9,256,8,8,256,3,3,256,1,2,max_pool_2, variable_8,variable_11, variable_12,variable_13,variable_10); 
		 frelu(256,4,4,max_pool_2,max_pool_2); 
		 CnBnBwn(max_pool_2,variable_15,256,4,4,256,3,3,512,1,1,relu_4, variable_14,variable_17, variable_18,variable_19,variable_16); 
		 frelu(512,4,4,relu_4,relu_4); 
		 CnBnBwn(relu_4,variable_21,512,4,4,512,3,3,512,1,2,max_pool_3, variable_20,variable_23, variable_24,variable_25,variable_22); 
		 frelu(512,2,2,max_pool_3,max_pool_3); 
		 CnBnBwn(max_pool_3,variable_27,512,2,2,512,3,3,512,1,1,relu_6, variable_26,variable_29, variable_30,variable_31,variable_28); 
		 frelu(512,2,2,relu_6,relu_6); 
		 CnBnBwn(relu_6,variable_33,512,2,2,512,3,3,512,1,2,max_pool_4, variable_32,variable_35, variable_36,variable_37,variable_34); 
		 frelu(512,1,1,max_pool_4,max_pool_4); 
		 freshape(max_pool_4,1,1,512,reshape); 
		 FcBnXnorArrNoBin(reshape,variable_1,512,10,matmul,NULL, NULL, NULL, NULL); 
		 fmul(matmul, 10,1.0,mul); 
		 fadd(mul,unsqueeze,1,10,0,output); 
		 float max = -INFINITY; 
		int maxIdx = 0; 
		for (int i = 0; i <10; i++) { 
			 printf("%f, ", output[i]);
			 if (output[i] > max) { 
				 max = output[i]; 
				maxIdx = i;
			 }
		}
		printf("\n");printf("Image %d: label: %d, actual: %d\n",img, maxIdx, labels[img]); 
		if (maxIdx == labels[img]) correct += 1; 
	}
	printf("Accuracy: %f%%\n", 100.0*(float)correct/100); 
	return (EXIT_SUCCESS); 
}