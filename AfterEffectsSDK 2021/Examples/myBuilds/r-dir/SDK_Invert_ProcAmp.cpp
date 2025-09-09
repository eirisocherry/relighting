/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007-2018 Adobe Systems Incorporated                  */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Systems Incorporated and its suppliers, if    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Systems Incorporated and its    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Systems         */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/

/*
	SDK_Invert_ProcAmp.cpp
	
	A simple Invert ProcAmp effect. This effect adds color invert and ProcAmp to the layer.
	
	Revision History
		
	Version		Change													Engineer	Date
	=======		======													========	======
	1.0			created													ykuang		09/10/2018

*/


#if HAS_CUDA
	#include <cuda_runtime.h>
	// SDK_Invert_ProcAmp.h defines these and are needed whereas the cuda_runtime ones are not.
	#undef MAJOR_VERSION
	#undef MINOR_VERSION
#endif

#include "SDK_Invert_ProcAmp.h"
#include "Structures.h"
#include <iostream>

// brings in M_PI on Windows
#define _USE_MATH_DEFINES
#include <math.h>

inline PF_Err CL2Err(cl_int cl_result) {
	if (cl_result == CL_SUCCESS) {
		return PF_Err_NONE;
	} else {
		// set a breakpoint here to pick up OpenCL errors.
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
}

#define CL_ERR(FUNC) ERR(CL2Err(FUNC))


//  CUDA kernel; see SDK_Invert_ProcAmp.cu.
extern void firstPassCUDA(
	float const* src1,
	float const* src2,
	float* src3,

	unsigned int pitch1,
	unsigned int width1,
	unsigned int height1,
	int	is16f1,

	unsigned int pitch2,
	unsigned int width2,
	unsigned int height2,
	int	is16f2,

	unsigned int pitch3,
	unsigned int width3,
	unsigned int height3,
	int	is16f3,

	InvertProcAmpParams* d_infoP
);

static PF_Err 
About (	
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_SPRINTF(	out_data->return_msg, 
				"%s, v%d.%d\r%s",
				NAME, 
				MAJOR_VERSION, 
				MINOR_VERSION, 
				DESCRIPTION);

	return PF_Err_NONE;
}

static PF_Err 
GlobalSetup (
	PF_InData		*in_dataP,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err	err				= PF_Err_NONE;

	out_data->my_version	= PF_VERSION(	MAJOR_VERSION, 
											MINOR_VERSION,
											BUG_VERSION, 
											STAGE_VERSION, 
											BUILD_VERSION);
	
	out_data->out_flags		=	PF_OutFlag_PIX_INDEPENDENT	|
								PF_OutFlag_DEEP_COLOR_AWARE;

	out_data->out_flags2	=	PF_OutFlag2_FLOAT_COLOR_AWARE	|
								PF_OutFlag2_SUPPORTS_SMART_RENDER	|
								PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
								PF_OutFlag2_I_USE_3D_CAMERA |
								PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;

	return err;
}

static PF_Err 
ParamsSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err			err = PF_Err_NONE;
	PF_ParamDef		def;


	///////////
	// Debug //
	///////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Render Mode", 5, 1, "All|Ambient|Diffuse|Specular|Shadows", POPUP_RENDER_MODE);

	////////////////////
	// Depth Settings //
	////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Depth Settings", TOPIC_ADD_DEPTH_SETTINGS);
	
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Depth Far",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100000.000),
		PF_FpLong(25000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DEPTH_FAR
	);
	
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Depth Black Is Near", FALSE, 0, CHECKBOX_DEPTH_BLACK_IS_NEAR);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_DEPTH_SETTINGS);

	/////////////////////
	// Normal Settings //
	/////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Normal Settings", TOPIC_ADD_NORMAL_SETTINGS);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Normal Pass Normalized", 1, LAYER_NORMAL);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_NORMAL_SETTINGS);

	//////////////////
	// Light Layers //
	//////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Layers", TOPIC_ADD_LIGHT_LAYERS);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Start 1", 1, LAYER_LIGHT_START_1);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light End 1", 1, LAYER_LIGHT_END_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Start 2", 1, LAYER_LIGHT_START_2);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light End 2", 1, LAYER_LIGHT_END_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Start 3", 1, LAYER_LIGHT_START_3);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light End 3", 1, LAYER_LIGHT_END_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Start 4", 1, LAYER_LIGHT_START_4);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light End 4", 1, LAYER_LIGHT_END_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Start 5", 1, LAYER_LIGHT_START_5);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light End 5", 1, LAYER_LIGHT_END_5);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LIGHT_LAYERS);

	///////////////////////////
	// Global Light Settings //
	///////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Global Light Settings", TOPIC_ADD_GLOBAL_LIGHT_SETTINGS);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Intensity Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_MULTIPLIER
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_MULTIPLIER
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_GLOBAL_LIGHT_SETTINGS);

	////////////////////////////
	// Local Light Settings 1 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 1", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_1);

	// Light

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 1", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 1 1", 0, 0, 0, POINT_3D_LIGHT_POS_1_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 2 1", 0, 0, 0, POINT_3D_LIGHT_POS_2_1);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 1", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color 1", 255, 205, 120, COLOR_AMBIENT_COLOR_1);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 1", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color 1", 255, 205, 120, COLOR_DIFFUSE_COLOR_1);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 1", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color 1", 255, 205, 120, COLOR_SPECULAR_COLOR_1);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 1", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 1", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 1", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 1", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 1",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 1", 0, 0, 0, COLOR_SHADOW_COLOR_1);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_1);


	////////////////////////////
	// Local Light Settings 2 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 2", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_2);

	// Light

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 2", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 1 2", 0, 0, 0, POINT_3D_LIGHT_POS_1_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 2 2", 0, 0, 0, POINT_3D_LIGHT_POS_2_2);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 2", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color 2", 255, 205, 120, COLOR_AMBIENT_COLOR_2);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 2", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color 2", 255, 205, 120, COLOR_DIFFUSE_COLOR_2);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 2", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color 2", 255, 205, 120, COLOR_SPECULAR_COLOR_2);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 2", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 2", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 2", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 2", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 2",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 2", 0, 0, 0, COLOR_SHADOW_COLOR_2);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_2);


	////////////////////////////
	// Local Light Settings 3 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 3", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_3);

	// Light

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 3", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 1 3", 0, 0, 0, POINT_3D_LIGHT_POS_1_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 2 3", 0, 0, 0, POINT_3D_LIGHT_POS_2_3);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 3", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color 3", 255, 205, 120, COLOR_AMBIENT_COLOR_3);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 3", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color 3", 255, 205, 120, COLOR_DIFFUSE_COLOR_3);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 3", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color 3", 255, 205, 120, COLOR_SPECULAR_COLOR_3);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 3", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 3", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 3", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 3", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 3",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 3", 0, 0, 0, COLOR_SHADOW_COLOR_3);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_3);


	////////////////////////////
	// Local Light Settings 4 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 4", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_4);

	// Light

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 4", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 1 4", 0, 0, 0, POINT_3D_LIGHT_POS_1_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 2 4", 0, 0, 0, POINT_3D_LIGHT_POS_2_4);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 4", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color 4", 255, 205, 120, COLOR_AMBIENT_COLOR_4);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 4", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color 4", 255, 205, 120, COLOR_DIFFUSE_COLOR_4);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 4", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color 4", 255, 205, 120, COLOR_SPECULAR_COLOR_4);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 4", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 4", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 4", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 4", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 4",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 4", 0, 0, 0, COLOR_SHADOW_COLOR_4);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_4);


	////////////////////////////
	// Local Light Settings 5 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 5", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_5);

	// Light

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 5", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 1 5", 0, 0, 0, POINT_3D_LIGHT_POS_1_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 2 5", 0, 0, 0, POINT_3D_LIGHT_POS_2_5);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 5", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color 5", 255, 205, 120, COLOR_AMBIENT_COLOR_5);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 5", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color 5", 255, 205, 120, COLOR_DIFFUSE_COLOR_5);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 5", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color 5", 255, 205, 120, COLOR_SPECULAR_COLOR_5);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 5", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 5", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 5", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 5", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 5",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 5", 0, 0, 0, COLOR_SHADOW_COLOR_5);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_5);
	
	out_data->num_params = RELIGHTING_NUM_PARAMS;

	return err;
}

#if HAS_METAL
	PF_Err NSError2PFErr(NSError *inError)
	{
		if (inError)
		{
			return PF_Err_INTERNAL_STRUCT_DAMAGED;  //For debugging, uncomment above line and set breakpoint here
		}
		return PF_Err_NONE;
	}
#endif //HAS_METAL


// GPU data initialized at GPU setup and used during render.
struct OpenCLGPUData
{
	cl_kernel invert_kernel;
	cl_kernel procamp_kernel;
};

#if HAS_METAL
	struct MetalGPUData
	{
		id<MTLComputePipelineState>invert_pipeline;
		id<MTLComputePipelineState>procamp_pipeline;
	};
#endif


static PF_Err
GPUDeviceSetup(
	PF_InData	*in_dataP,
	PF_OutData	*out_dataP,
	PF_GPUDeviceSetupExtra *extraP)
{
	PF_Err err = PF_Err_NONE;

	PF_GPUDeviceInfo device_info;
	AEFX_CLR_STRUCT(device_info);

	AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(	in_dataP,
																					   kPFHandleSuite,
																					   kPFHandleSuiteVersion1,
																					   out_dataP);

	AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuDeviceSuite =
	AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP,
										 kPFGPUDeviceSuite,
										 kPFGPUDeviceSuiteVersion1,
										 out_dataP);

	gpuDeviceSuite->GetDeviceInfo(in_dataP->effect_ref,
								  extraP->input->device_index,
								  &device_info);

	// Load and compile the kernel - a real plugin would cache binaries to disk

	if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
		// Nothing to do here. CUDA Kernel statically linked
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	} else if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {

		PF_Handle gpu_dataH	= handle_suite->host_new_handle(sizeof(OpenCLGPUData));
		OpenCLGPUData *cl_gpu_data = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

		cl_int result = CL_SUCCESS;

		char const *k16fString = "#define GF_OPENCL_SUPPORTS_16F 0\n";

		size_t sizes[] = { strlen(k16fString), strlen(kSDK_Invert_ProcAmp_Kernel_OpenCLString) };
		char const *strings[] = { k16fString, kSDK_Invert_ProcAmp_Kernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if(!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->invert_kernel = clCreateKernel(program, "InvertColorKernel", &result);
			CL_ERR(result);
		}

		if (!err) {
			cl_gpu_data->procamp_kernel = clCreateKernel(program, "ProcAmp2Kernel", &result);
			CL_ERR(result);
		}

		extraP->output->gpu_data = gpu_dataH;

		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		//Create a library from source
		NSString *source = [NSString stringWithCString:kSDK_Invert_ProcAmp_Kernel_MetalString encoding:NSUTF8StringEncoding];
		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

		NSError *error = nil;
		id<MTLLibrary> library = [[device newLibraryWithSource:source options:nil error:&error] autorelease];

		// An error code is set for Metal compile warnings, so use nil library as the error signal
		if(!err && !library) {
			err = NSError2PFErr(error);
		}

		// For debugging only. This will contain Metal compile warnings and erorrs.
		NSString *getError = error.localizedDescription;

		PF_Handle metal_handle = handle_suite->host_new_handle(sizeof(MetalGPUData));
		MetalGPUData *metal_data = reinterpret_cast<MetalGPUData *>(*metal_handle);

		//Create pipeline state from function extracted from library
		if (err == PF_Err_NONE)
		{
			id<MTLFunction> invert_function = nil;
			id<MTLFunction> procamp_function = nil;
			NSString *invert_name = [NSString stringWithCString:"InvertColorKernel" encoding:NSUTF8StringEncoding];
			NSString *procamp_name = [NSString stringWithCString:"ProcAmp2Kernel" encoding:NSUTF8StringEncoding];

			invert_function =  [ [library newFunctionWithName:invert_name] autorelease];
			procamp_function = [ [library newFunctionWithName:procamp_name] autorelease];
			
			if (!invert_function || !procamp_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->invert_pipeline = [device newComputePipelineStateWithFunction:invert_function error:&error];
				err = NSError2PFErr(error);
			}

			if (!err) {
				metal_data->procamp_pipeline = [device newComputePipelineStateWithFunction:procamp_function error:&error];
				err = NSError2PFErr(error);
			}

			if(!err) {
				extraP->output->gpu_data = metal_handle;
				out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
			}
		}
	}
#endif
	return err;
}


static PF_Err
GPUDeviceSetdown(
	PF_InData	*in_dataP,
	PF_OutData	*out_dataP,
	PF_GPUDeviceSetdownExtra *extraP)
{
	PF_Err err = PF_Err_NONE;

	if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData *cl_gpu_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

		(void)clReleaseKernel (cl_gpu_dataP->invert_kernel);
		(void)clReleaseKernel (cl_gpu_dataP->procamp_kernel);
		
		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(	in_dataP,
																						   kPFHandleSuite,
																						   kPFHandleSuiteVersion1,
																						   out_dataP);
		
		handle_suite->host_dispose_handle(gpu_dataH);
	}
	
	return err;
}

static void
DisposePreRenderData(
	void *pre_render_dataPV)
{
	if(pre_render_dataPV) {
		InvertProcAmpParams *infoP = reinterpret_cast<InvertProcAmpParams *>(pre_render_dataPV);
		free(infoP);
	}
}

static PF_Err
PreRender(
	PF_InData			*in_dataP,
	PF_OutData			*out_dataP,
	PF_PreRenderExtra	*extraP)
{
	PF_Err err = PF_Err_NONE;
	PF_CheckoutResult in_result, normal_result;
	PF_RenderRequest req = extraP->input->output_request;

	extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

	InvertProcAmpParams *infoP	= reinterpret_cast<InvertProcAmpParams *>( malloc(sizeof(InvertProcAmpParams)) );

	if (infoP) {

		////////////
		// Camera //
		////////////

		AEGP_SuiteHandler	suites(in_dataP->pica_basicP);

		A_Matrix4			matrix;
		A_Time				comp_timeT = { 0,1 };
		A_FpLong			dst_to_planePF;
		A_short				plane_widthPL;
		A_short				plane_heightPL;
		AEGP_LayerH			camera_layerH = NULL;

		ERR(suites.PFInterfaceSuite1()->AEGP_ConvertEffectToCompTime(in_dataP->effect_ref,
			in_dataP->current_time,
			in_dataP->time_scale,
			&comp_timeT));

		ERR(suites.PFInterfaceSuite1()->AEGP_GetEffectCamera(in_dataP->effect_ref,
			&comp_timeT,
			&camera_layerH));

		ERR(suites.PFInterfaceSuite1()->AEGP_GetEffectCameraMatrix(
			in_dataP->effect_ref,
			&comp_timeT,
			&matrix,
			&dst_to_planePF,
			&plane_widthPL,
			&plane_heightPL
		));

		infoP->camVx1 = static_cast<float>(matrix.mat[0][0]);
		infoP->camVx2 = static_cast<float>(matrix.mat[0][1]);
		infoP->camVx3 = static_cast<float>(matrix.mat[0][2]);
		infoP->camVx4 = static_cast<float>(matrix.mat[0][3]);

		infoP->camVy1 = static_cast<float>(matrix.mat[1][0]);
		infoP->camVy2 = static_cast<float>(matrix.mat[1][1]);
		infoP->camVy3 = static_cast<float>(matrix.mat[1][2]);
		infoP->camVy4 = static_cast<float>(matrix.mat[1][3]);

		infoP->camVz1 = static_cast<float>(matrix.mat[2][0]);
		infoP->camVz2 = static_cast<float>(matrix.mat[2][1]);
		infoP->camVz3 = static_cast<float>(matrix.mat[2][2]);
		infoP->camVz4 = static_cast<float>(matrix.mat[2][3]);

		infoP->camPos1 = static_cast<float>(matrix.mat[3][0]);
		infoP->camPos2 = static_cast<float>(matrix.mat[3][1]);
		infoP->camPos3 = static_cast<float>(matrix.mat[3][2]);
		infoP->camPos4 = static_cast<float>(matrix.mat[3][3]);

		infoP->cameraZoom = static_cast<float>(dst_to_planePF);
		infoP->cameraWidth = static_cast<float>(plane_widthPL);
		infoP->cameraHeight = static_cast<float>(plane_heightPL);

		///////////
		// Debug //
		///////////

		PF_ParamDef cur_param;

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_RENDER_MODE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->renderMode = static_cast<int>(cur_param.u.pd.value);

		////////////////////
		// Depth Settings //
		////////////////////

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DEPTH_FAR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->depthFar = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DEPTH_BLACK_IS_NEAR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->depthBlackIsNear = static_cast<bool>(cur_param.u.bd.value);

		///////////////////////////
		// Global Light Settings //
		///////////////////////////

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensityMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturationMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		////////////////////////////
		// Local Light Settings 1 //
		////////////////////////////

		// Light Start

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_1_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX1[0] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY1[0] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ1[0] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_2_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX2[0] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY2[0] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ2[0] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[0] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);


		////////////////////////////
		// Local Light Settings 2 //
		////////////////////////////

		// Light Start

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_1_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX1[1] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY1[1] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ1[1] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_2_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX2[1] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY2[1] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ2[1] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[1] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);


		////////////////////////////
		// Local Light Settings 3 //
		////////////////////////////

		// Light Start

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_1_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX1[2] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY1[2] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ1[2] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_2_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX2[2] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY2[2] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ2[2] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[2] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);


		////////////////////////////
		// Local Light Settings 4 //
		////////////////////////////

		// Light Start

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_1_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX1[3] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY1[3] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ1[3] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_2_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX2[3] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY2[3] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ2[3] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[3] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);


		////////////////////////////
		// Local Light Settings 5 //
		////////////////////////////

		// Light Start

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_1_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX1[4] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY1[4] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ1[4] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POS_2_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->posX2[4] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->posY2[4] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->posZ2[4] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[4] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);
		
		// End

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;
		
		ERR(
			extraP->cb->checkout_layer(
				in_dataP->effect_ref,
				RELIGHTING_INPUT,
				RELIGHTING_INPUT,
				&req,
				in_dataP->current_time,
				in_dataP->time_step,
				in_dataP->time_scale,
				&in_result
			)
		);

		ERR(
			extraP->cb->checkout_layer(
				in_dataP->effect_ref,
				LAYER_NORMAL,
				LAYER_NORMAL,
				&req,
				in_dataP->current_time,
				in_dataP->time_step,
				in_dataP->time_scale,
				&normal_result
			)
		);
		
		UnionLRect(&in_result.result_rect, 		&extraP->output->result_rect);
		UnionLRect(&in_result.max_result_rect, 	&extraP->output->max_result_rect);
	} else {
		err = PF_Err_OUT_OF_MEMORY;
	}
	return err;
}

static size_t
RoundUp(
	size_t inValue,
	size_t inMultiple)
{
	return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0;
}


typedef struct
{
	int mSrcPitch;
	int mDstPitch;
	int m16f;
	int mWidth;
	int mHeight;
} InvertColorParams;



size_t DivideRoundUp(
					 size_t inValue,
					 size_t inMultiple)
{
	return inValue ? (inValue + inMultiple - 1) / inMultiple: 0;
}


/*
 **The ProcAmp2Params structure mirrors that used by the metal kernel.
 */
typedef struct
{
	int mSrcPitch;
	int mDstPitch;
	int m16f;
	int mWidth;
	int mHeight;
	float mBrightness;
	float mContrast;
	float mHueCosSaturation;
	float mHueSinSaturation;
} ProcAmp2Params;


static PF_Err
SmartRenderGPU(
	PF_InData				*in_dataP,
	PF_OutData				*out_dataP,
	PF_PixelFormat			pixel_format,
	PF_EffectWorld			*input_worldP,
	PF_EffectWorld          *normal_worldP,
	PF_EffectWorld			*output_worldP,
	PF_SmartRenderExtra		*extraP,
	InvertProcAmpParams		*infoP)
{
	PF_Err			err		= PF_Err_NONE;

	AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite = AEFX_SuiteScoper<PF_GPUDeviceSuite1>( in_dataP,
																						  kPFGPUDeviceSuite,
																						  kPFGPUDeviceSuiteVersion1,
																						  out_dataP);

	if(pixel_format != PF_PixelFormat_GPU_BGRA128) {
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}
	A_long bytes_per_pixel = 16;

	PF_GPUDeviceInfo device_info;
	ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref, extraP->input->device_index, &device_info));

	void *src_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

	void* normal_mem = 0;
	if (normal_worldP) {
		infoP->normalExistToggle = true;
		ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, normal_worldP, &normal_mem));
	}
	else {
		infoP->normalExistToggle = false;

		ERR(gpu_suite->CreateGPUWorld(in_dataP->effect_ref,
			extraP->input->device_index,
			input_worldP->width,
			input_worldP->height,
			input_worldP->pix_aspect_ratio,
			in_dataP->field,
			pixel_format,
			false,
			&normal_worldP));

		ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, normal_worldP, &normal_mem));
	}


	void *dst_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP, &dst_mem));

	// read the parameters
	InvertColorParams invert_params;
	ProcAmp2Params procamp_params;

	invert_params.mWidth  = procamp_params.mWidth  = input_worldP->width;
	invert_params.mHeight = procamp_params.mHeight = input_worldP->height;

	A_long src_row_bytes = input_worldP->rowbytes;
	A_long dst_row_bytes = output_worldP->rowbytes;

	procamp_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
	procamp_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
	procamp_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);



	// Send data to kernels
	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData *cl_gpu_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint invert_param_index = 0;
		cl_uint procamp_param_index = 0;

		// Set the arguments
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.mHeight));

		// Launch the kernel
		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(invert_params.mWidth, threadBlock[0]), RoundUp(invert_params.mHeight, threadBlock[1])};

		CL_ERR(clEnqueueNDRangeKernel(
									  (cl_command_queue)device_info.command_queuePV,
									  cl_gpu_dataP->invert_kernel,
									  2,
									  0,
									  grid,
									  threadBlock,
									  0,
									  0,
									  0));

		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &procamp_params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &procamp_params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &procamp_params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &procamp_params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &procamp_params.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(float), &procamp_params.mBrightness));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(float), &procamp_params.mContrast));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(float), &procamp_params.mHueCosSaturation));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(float), &procamp_params.mHueSinSaturation));
		
		CL_ERR(clEnqueueNDRangeKernel(
									(cl_command_queue)device_info.command_queuePV,
									cl_gpu_dataP->procamp_kernel,
									2,
									0,
									grid,
									threadBlock,
									0,
									0,
									0));
	}
	#if HAS_CUDA
		else if (!err && extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
			
			InvertProcAmpParams* d_infoP;
			cudaMalloc(&d_infoP, sizeof(InvertProcAmpParams));
			cudaMemcpy(d_infoP, infoP, sizeof(InvertProcAmpParams), cudaMemcpyHostToDevice);

			firstPassCUDA(
				(const float*)src_mem,
				(const float*)normal_mem,
				(float*)dst_mem,

				input_worldP->rowbytes / bytes_per_pixel,
				input_worldP->width,
				input_worldP->height,
				(pixel_format != PF_PixelFormat_GPU_BGRA128),

				normal_worldP->rowbytes / bytes_per_pixel,
				normal_worldP->width,
				normal_worldP->height,
				(pixel_format != PF_PixelFormat_GPU_BGRA128),

				input_worldP->rowbytes / bytes_per_pixel,
				input_worldP->width,
				input_worldP->height,
				(pixel_format != PF_PixelFormat_GPU_BGRA128),

				d_infoP
			);

			cudaFree(d_infoP);

			if (cudaPeekAtLastError() != cudaSuccess) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}
		}
	#endif
	#if HAS_METAL
		else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
		{
			ScopedAutoreleasePool pool;
			
			Handle metal_handle = (Handle)extraP->input->gpu_data;
			MetalGPUData *metal_dataP = reinterpret_cast<MetalGPUData *>(*metal_handle);


			//Set the arguments
			id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
			id<MTLBuffer> procamp_param_buffer = [[device newBufferWithBytes:&procamp_params
																length:sizeof(ProcAmp2Params)
																options:MTLResourceStorageModeManaged] autorelease];
			
			id<MTLBuffer> invert_param_buffer = [[device newBufferWithBytes:&invert_params
															    length:sizeof(InvertColorParams)
															    options:MTLResourceStorageModeManaged] autorelease];

			//Launch the command
			id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
			id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
			id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
			id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
			id<MTLBuffer> im_metal_buffer = (id<MTLBuffer>)im_mem;
			id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

			MTLSize threadsPerGroup1 = {[metal_dataP->invert_pipeline threadExecutionWidth], 16, 1};
			MTLSize numThreadgroups1 = {DivideRoundUp(invert_params.mWidth, threadsPerGroup1.width), DivideRoundUp(invert_params.mHeight, threadsPerGroup1.height), 1};
			
			MTLSize threadsPerGroup2 = {[metal_dataP->procamp_pipeline threadExecutionWidth], 16, 1};
			MTLSize numThreadgroups2 = {DivideRoundUp(procamp_params.mWidth, threadsPerGroup2.width), DivideRoundUp(procamp_params.mHeight, threadsPerGroup2.height), 1};

			[computeEncoder setComputePipelineState:metal_dataP->invert_pipeline];
			[computeEncoder setBuffer:src_metal_buffer offset:0 atIndex:0];
			[computeEncoder setBuffer:im_metal_buffer offset:0 atIndex:1];
			[computeEncoder setBuffer:invert_param_buffer offset:0 atIndex:2];
			[computeEncoder dispatchThreadgroups:numThreadgroups1 threadsPerThreadgroup:threadsPerGroup1];

			err = NSError2PFErr([commandBuffer error]);

			if (!err) {
				[computeEncoder setComputePipelineState:metal_dataP->procamp_pipeline];
				[computeEncoder setBuffer:im_metal_buffer offset:0 atIndex:0];
				[computeEncoder setBuffer:dst_metal_buffer offset:0 atIndex:1];
				[computeEncoder setBuffer:procamp_param_buffer offset:0 atIndex:2];
				[computeEncoder dispatchThreadgroups:numThreadgroups2 threadsPerThreadgroup:threadsPerGroup2];
				[computeEncoder endEncoding];
				[commandBuffer commit];

				err = NSError2PFErr([commandBuffer error]);
			}

		}
	#endif //HAS_METAL

	// Free up allocated buffers (put at the end of smartGPU function)
	if (!infoP->normalExistToggle) {
		ERR(gpu_suite->DisposeGPUWorld(in_dataP->effect_ref, normal_worldP));
	}

	return err;
}


static PF_Err
SmartRender(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_SmartRenderExtra		*extraP,
	bool					isGPU)
{

	PF_Err			err		= PF_Err_NONE,
					err2 	= PF_Err_NONE;
	
	PF_EffectWorld	*input_worldP	= NULL,
					*normal_worldP  = NULL,
					*output_worldP  = NULL;

	// Parameters can be queried during render. In this example, we pass them from PreRender as an example of using pre_render_data.
	InvertProcAmpParams *infoP = reinterpret_cast<InvertProcAmpParams *>(extraP->input->pre_render_data);

	if (infoP) {
		ERR((extraP->cb->checkout_layer_pixels(	in_data->effect_ref, RELIGHTING_INPUT, &input_worldP)));
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, LAYER_NORMAL, &normal_worldP)));
		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
																				kPFWorldSuite,
																				kPFWorldSuiteVersion2,
																				out_data);
		PF_PixelFormat	pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, normal_worldP, output_worldP, extraP, infoP));

		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, RELIGHTING_INPUT));
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, LAYER_NORMAL));
	} else {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
	return err;
}


extern "C" DllExport
PF_Err PluginDataEntryFunction(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT(
		inPtr,
		inPluginDataCallBackPtr,
		"r-dir", // Name
		"ADBE r-dir", // Match Name
		"Rikki", // Category
		AE_RESERVED_INFO); // Reserved Info

	return result;
}


PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_dataP,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;
	
	try {
		switch (cmd) 
		{
			case PF_Cmd_ABOUT:
				err = About(in_dataP,out_data,params,output);
				break;
			case PF_Cmd_GLOBAL_SETUP:
				err = GlobalSetup(in_dataP,out_data,params,output);
				break;
			case PF_Cmd_PARAMS_SETUP:
				err = ParamsSetup(in_dataP,out_data,params,output);
				break;
			case PF_Cmd_GPU_DEVICE_SETUP:
				err = GPUDeviceSetup(in_dataP, out_data, (PF_GPUDeviceSetupExtra *)extra);
				break;
			case PF_Cmd_GPU_DEVICE_SETDOWN:
				err = GPUDeviceSetdown(in_dataP, out_data, (PF_GPUDeviceSetdownExtra *)extra);
				break;
			case PF_Cmd_SMART_PRE_RENDER:
				err = PreRender(in_dataP, out_data, (PF_PreRenderExtra*)extra);
				break;
			case PF_Cmd_SMART_RENDER:
				err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra, false);
				break;
			case PF_Cmd_SMART_RENDER_GPU:
				err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra *)extra, true);
				break;
		}
	} catch(PF_Err &thrown_err) {
		// Never EVER throw exceptions into AE.
		err = thrown_err;
	}
	return err;
}
