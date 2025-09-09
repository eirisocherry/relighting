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
	PF_ADD_LAYER("Light 1", 1, LAYER_LIGHT_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 2", 1, LAYER_LIGHT_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 3", 1, LAYER_LIGHT_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 4", 1, LAYER_LIGHT_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 5", 1, LAYER_LIGHT_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 6", 1, LAYER_LIGHT_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 7", 1, LAYER_LIGHT_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 8", 1, LAYER_LIGHT_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 9", 1, LAYER_LIGHT_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 10", 1, LAYER_LIGHT_10);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LIGHT_LAYERS);

	///////////////////////////
	// Global Light Settings //
	///////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Global Light Settings", TOPIC_ADD_GLOBAL_LIGHT_SETTINGS);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_MULTIPLIER
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_MULTIPLIER
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_MULTIPLIER
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_MULTIPLIER
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_MULTIPLIER
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_MULTIPLIER
	);

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
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_MULTIPLIER
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_GLOBAL_LIGHT_SETTINGS);

	////////////////////////////
	// Local Light Settings 1 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 1", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_1);

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 1", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 1", 0, 0, 0, POINT_3D_LIGHT_POSITION_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 1", 0, 0, 0, POINT_3D_LIGHT_VX_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 1", 0, 0, 0, POINT_3D_LIGHT_VY_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 1", 0, 0, 0, POINT_3D_LIGHT_VZ_1);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 1", FALSE, 0, CHECKBOX_INVERT_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 1",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 1",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_1
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 1", 5, 1, "None|1|2|3|Custom", POPUP_IES_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 1", 0, 0, FALSE, POINT_2D_IES_1_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 1", 0, 0, FALSE, POINT_2D_IES_2_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 1", 0, 0, FALSE, POINT_2D_IES_3_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 1", 0, 0, FALSE, POINT_2D_IES_4_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 1", 0, 0, FALSE, POINT_2D_IES_5_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 1", 0, 0, FALSE, POINT_2D_IES_6_1);

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
	PF_ADD_COLOR("Ambient Color Near 1", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 1", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 1", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_1
	);

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
	PF_ADD_COLOR("Diffuse Color Near 1", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 1", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 1", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_1
	);

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
	PF_ADD_COLOR("Specular Color Near 1", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 1", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 1", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_1
	);

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
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 1", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_1);

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

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 2", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 2", 0, 0, 0, POINT_3D_LIGHT_POSITION_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 2", 0, 0, 0, POINT_3D_LIGHT_VX_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 2", 0, 0, 0, POINT_3D_LIGHT_VY_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 2", 0, 0, 0, POINT_3D_LIGHT_VZ_2);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 2", FALSE, 0, CHECKBOX_INVERT_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 2",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 2",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_2
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 2", 5, 1, "None|1|2|3|Custom", POPUP_IES_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 2", 0, 0, FALSE, POINT_2D_IES_1_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 2", 0, 0, FALSE, POINT_2D_IES_2_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 2", 0, 0, FALSE, POINT_2D_IES_3_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 2", 0, 0, FALSE, POINT_2D_IES_4_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 2", 0, 0, FALSE, POINT_2D_IES_5_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 2", 0, 0, FALSE, POINT_2D_IES_6_2);

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
	PF_ADD_COLOR("Ambient Color Near 2", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 2", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 2", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_2
	);

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
	PF_ADD_COLOR("Diffuse Color Near 2", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 2", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 2", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_2
	);

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
	PF_ADD_COLOR("Specular Color Near 2", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 2", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 2", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_2
	);

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
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 2", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_2);

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

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 3", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 3", 0, 0, 0, POINT_3D_LIGHT_POSITION_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 3", 0, 0, 0, POINT_3D_LIGHT_VX_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 3", 0, 0, 0, POINT_3D_LIGHT_VY_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 3", 0, 0, 0, POINT_3D_LIGHT_VZ_3);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 3", FALSE, 0, CHECKBOX_INVERT_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 3",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 3",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_3
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 3", 5, 1, "None|1|2|3|Custom", POPUP_IES_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 3", 0, 0, FALSE, POINT_2D_IES_1_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 3", 0, 0, FALSE, POINT_2D_IES_2_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 3", 0, 0, FALSE, POINT_2D_IES_3_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 3", 0, 0, FALSE, POINT_2D_IES_4_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 3", 0, 0, FALSE, POINT_2D_IES_5_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 3", 0, 0, FALSE, POINT_2D_IES_6_3);

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
	PF_ADD_COLOR("Ambient Color Near 3", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 3", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 3", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_3
	);

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
	PF_ADD_COLOR("Diffuse Color Near 3", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 3", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 3", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_3
	);

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
	PF_ADD_COLOR("Specular Color Near 3", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 3", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 3", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_3
	);

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
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 3", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_3);

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

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 4", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 4", 0, 0, 0, POINT_3D_LIGHT_POSITION_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 4", 0, 0, 0, POINT_3D_LIGHT_VX_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 4", 0, 0, 0, POINT_3D_LIGHT_VY_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 4", 0, 0, 0, POINT_3D_LIGHT_VZ_4);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 4", FALSE, 0, CHECKBOX_INVERT_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 4",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 4",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_4
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 4", 5, 1, "None|1|2|3|Custom", POPUP_IES_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 4", 0, 0, FALSE, POINT_2D_IES_1_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 4", 0, 0, FALSE, POINT_2D_IES_2_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 4", 0, 0, FALSE, POINT_2D_IES_3_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 4", 0, 0, FALSE, POINT_2D_IES_4_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 4", 0, 0, FALSE, POINT_2D_IES_5_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 4", 0, 0, FALSE, POINT_2D_IES_6_4);

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
	PF_ADD_COLOR("Ambient Color Near 4", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 4", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 4", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_4
	);

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
	PF_ADD_COLOR("Diffuse Color Near 4", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 4", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 4", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_4
	);

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
	PF_ADD_COLOR("Specular Color Near 4", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 4", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 4", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_4
	);

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
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 4", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_4);

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

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 5", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 5", 0, 0, 0, POINT_3D_LIGHT_POSITION_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 5", 0, 0, 0, POINT_3D_LIGHT_VX_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 5", 0, 0, 0, POINT_3D_LIGHT_VY_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 5", 0, 0, 0, POINT_3D_LIGHT_VZ_5);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 5", FALSE, 0, CHECKBOX_INVERT_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 5",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 5",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_5
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 5", 5, 1, "None|1|2|3|Custom", POPUP_IES_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 5", 0, 0, FALSE, POINT_2D_IES_1_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 5", 0, 0, FALSE, POINT_2D_IES_2_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 5", 0, 0, FALSE, POINT_2D_IES_3_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 5", 0, 0, FALSE, POINT_2D_IES_4_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 5", 0, 0, FALSE, POINT_2D_IES_5_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 5", 0, 0, FALSE, POINT_2D_IES_6_5);

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
	PF_ADD_COLOR("Ambient Color Near 5", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 5", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 5", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_5
	);

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
	PF_ADD_COLOR("Diffuse Color Near 5", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 5", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 5", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_5
	);

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
	PF_ADD_COLOR("Specular Color Near 5", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 5", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 5", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_5
	);

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
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 5", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_5);

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


	////////////////////////////
	// Local Light Settings 6 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 6", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_6);

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 6", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 6", 0, 0, 0, POINT_3D_LIGHT_POSITION_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 6", 0, 0, 0, POINT_3D_LIGHT_VX_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 6", 0, 0, 0, POINT_3D_LIGHT_VY_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 6", 0, 0, 0, POINT_3D_LIGHT_VZ_6);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 6", FALSE, 0, CHECKBOX_INVERT_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 6",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 6",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_6
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 6", 5, 1, "None|1|2|3|Custom", POPUP_IES_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 6", 0, 0, FALSE, POINT_2D_IES_1_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 6", 0, 0, FALSE, POINT_2D_IES_2_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 6", 0, 0, FALSE, POINT_2D_IES_3_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 6", 0, 0, FALSE, POINT_2D_IES_4_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 6", 0, 0, FALSE, POINT_2D_IES_5_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 6", 0, 0, FALSE, POINT_2D_IES_6_6);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 6", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Near 6", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 6", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 6", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_6
	);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 6", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Near 6", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 6", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 6", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_6
	);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 6", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Near 6", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 6", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 6", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_6
	);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 6", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 6", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 6", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 6", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 6", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 6",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 6", 0, 0, 0, COLOR_SHADOW_COLOR_6);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_6);


	////////////////////////////
	// Local Light Settings 7 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 7", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_7);

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 7", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 7", 0, 0, 0, POINT_3D_LIGHT_POSITION_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 7", 0, 0, 0, POINT_3D_LIGHT_VX_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 7", 0, 0, 0, POINT_3D_LIGHT_VY_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 7", 0, 0, 0, POINT_3D_LIGHT_VZ_7);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 7", FALSE, 0, CHECKBOX_INVERT_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 7",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 7",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_7
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 7", 5, 1, "None|1|2|3|Custom", POPUP_IES_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 7", 0, 0, FALSE, POINT_2D_IES_1_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 7", 0, 0, FALSE, POINT_2D_IES_2_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 7", 0, 0, FALSE, POINT_2D_IES_3_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 7", 0, 0, FALSE, POINT_2D_IES_4_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 7", 0, 0, FALSE, POINT_2D_IES_5_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 7", 0, 0, FALSE, POINT_2D_IES_6_7);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 7", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Near 7", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 7", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 7", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_7
	);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 7", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Near 7", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 7", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 7", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_7
	);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 7", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Near 7", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 7", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 7", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_7
	);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 7", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 7", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 7", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 7", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 7", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 7",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 7", 0, 0, 0, COLOR_SHADOW_COLOR_7);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_7);


	////////////////////////////
	// Local Light Settings 8 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 8", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_8);

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 8", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 8", 0, 0, 0, POINT_3D_LIGHT_POSITION_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 8", 0, 0, 0, POINT_3D_LIGHT_VX_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 8", 0, 0, 0, POINT_3D_LIGHT_VY_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 8", 0, 0, 0, POINT_3D_LIGHT_VZ_8);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 8", FALSE, 0, CHECKBOX_INVERT_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 8",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 8",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_8
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 8", 5, 1, "None|1|2|3|Custom", POPUP_IES_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 8", 0, 0, FALSE, POINT_2D_IES_1_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 8", 0, 0, FALSE, POINT_2D_IES_2_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 8", 0, 0, FALSE, POINT_2D_IES_3_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 8", 0, 0, FALSE, POINT_2D_IES_4_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 8", 0, 0, FALSE, POINT_2D_IES_5_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 8", 0, 0, FALSE, POINT_2D_IES_6_8);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 8", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Near 8", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 8", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 8", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_8
	);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 8", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Near 8", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 8", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 8", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_8
	);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 8", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Near 8", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 8", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 8", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_8
	);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 8", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 8", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 8", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 8", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 8", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 8",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 8", 0, 0, 0, COLOR_SHADOW_COLOR_8);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_8);


	////////////////////////////
	// Local Light Settings 9 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 9", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_9);

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 9", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 9", 0, 0, 0, POINT_3D_LIGHT_POSITION_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 9", 0, 0, 0, POINT_3D_LIGHT_VX_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 9", 0, 0, 0, POINT_3D_LIGHT_VY_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 9", 0, 0, 0, POINT_3D_LIGHT_VZ_9);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 9", FALSE, 0, CHECKBOX_INVERT_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 9",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 9",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_9
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 9", 5, 1, "None|1|2|3|Custom", POPUP_IES_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 9", 0, 0, FALSE, POINT_2D_IES_1_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 9", 0, 0, FALSE, POINT_2D_IES_2_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 9", 0, 0, FALSE, POINT_2D_IES_3_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 9", 0, 0, FALSE, POINT_2D_IES_4_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 9", 0, 0, FALSE, POINT_2D_IES_5_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 9", 0, 0, FALSE, POINT_2D_IES_6_9);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 9", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Near 9", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 9", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 9", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_9
	);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 9", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Near 9", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 9", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 9", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_9
	);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 9", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Near 9", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 9", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 9", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_9
	);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 9", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 9", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 9", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 9", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 9", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 9",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 9", 0, 0, 0, COLOR_SHADOW_COLOR_9);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_9);


	////////////////////////////
	// Local Light Settings 10 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 10", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_10);

	// Main

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 10", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 10", 0, 0, 0, POINT_3D_LIGHT_POSITION_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector X 10", 0, 0, 0, POINT_3D_LIGHT_VX_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Y 10", 0, 0, 0, POINT_3D_LIGHT_VY_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Vector Z 10", 0, 0, 0, POINT_3D_LIGHT_VZ_10);

	// Shape

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 10", FALSE, 0, CHECKBOX_INVERT_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Length 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_LENGTH_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle X 10",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEX_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Angle Y 10",
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(0.000),
		PF_FpLong(180.000),
		PF_FpLong(90.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_ANGLEY_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Curvature 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_CURVATURE_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Feather 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FEATHER_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Falloff 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(2.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_FALLOFF_10
	);


	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("IES 10", 5, 1, "None|1|2|3|Custom", POPUP_IES_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 1 10", 0, 0, FALSE, POINT_2D_IES_1_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 2 10", 0, 0, FALSE, POINT_2D_IES_2_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 3 10", 0, 0, FALSE, POINT_2D_IES_3_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 4 10", 0, 0, FALSE, POINT_2D_IES_4_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 5 10", 0, 0, FALSE, POINT_2D_IES_5_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Brightness / Distance 6 10", 0, 0, FALSE, POINT_2D_IES_6_10);

	// Ambient

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Toggle 10", FALSE, 0, CHECKBOX_AMBIENT_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Intensity 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_INTENSITY_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Saturation 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_SATURATION_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Near 10", 255, 205, 120, COLOR_AMBIENT_COLOR_NEAR_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Ambient Color Far Toggle 10", FALSE, 0, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Ambient Color Far 10", 255, 157, 0, COLOR_AMBIENT_COLOR_FAR_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Ambient Color Falloff 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_AMBIENT_COLOR_FALLOFF_10
	);

	// Diffuse

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Toggle 10", FALSE, 0, CHECKBOX_DIFFUSE_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Intensity 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_INTENSITY_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Saturation 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_SATURATION_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Near 10", 255, 205, 120, COLOR_DIFFUSE_COLOR_NEAR_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Diffuse Color Far Toggle 10", FALSE, 0, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Diffuse Color Far 10", 255, 157, 0, COLOR_DIFFUSE_COLOR_FAR_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Diffuse Color Falloff 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_DIFFUSE_COLOR_FALLOFF_10
	);

	// Specular

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Toggle 10", FALSE, 0, CHECKBOX_SPECULAR_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Size 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(32.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SIZE_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Intensity 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_INTENSITY_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Saturation 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_SATURATION_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Near 10", 255, 205, 120, COLOR_SPECULAR_COLOR_NEAR_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Specular Color Far Toggle 10", FALSE, 0, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Specular Color Far 10", 255, 157, 0, COLOR_SPECULAR_COLOR_FAR_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Specular Color Falloff 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SPECULAR_COLOR_FALLOFF_10
	);

	// Shadows

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Toggle 10", FALSE, 0, CHECKBOX_SHADOW_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Ambient Toggle 10", FALSE, 0, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Diffuse Toggle 10", FALSE, 0, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Ignore Specular Toggle 10", FALSE, 0, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Shadow Clip To Light Toggle 10", TRUE, 0, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Sample Step 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SAMPLE_STEP_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Improved Sample Radius 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(100.000),
		PF_FpLong(30.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Max Length 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(1000.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_MAX_LENGTH_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold Start 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_START_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Threshold End 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(5.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_THRESHOLD_END_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Radius 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_RADIUS_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Softness Samples 10",
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(1.000),
		PF_FpLong(128.000),
		PF_FpLong(16.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_SOFTNESS_SAMPLES_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Shadow Intensity 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SHADOW_INTENSITY_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Shadow Color 10", 0, 0, 0, COLOR_SHADOW_COLOR_10);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_10);
	
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

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lengthMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleXmultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleYmultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvatureMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->featherMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloffMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensityMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturationMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloffMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		////////////////////////////
		// Local Light Settings 1 //
		////////////////////////////

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[0] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[0] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[0] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[0] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[0] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[0] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[0] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[0] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[0] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[0] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[0] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[0] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[0] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[0] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[0] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[0] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[0] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[0] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[0] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[0] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[0] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[0] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[0] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[0] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[0] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[0] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[0] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[0] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[0] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[0] = static_cast<bool>(cur_param.u.bd.value);

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

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[1] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[1] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[1] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[1] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[1] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[1] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[1] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[1] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[1] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[1] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[1] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[1] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[1] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[1] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[1] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[1] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[1] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[1] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[1] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[1] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[1] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[1] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[1] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[1] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[1] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[1] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[1] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[1] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[1] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[1] = static_cast<bool>(cur_param.u.bd.value);

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

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[2] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[2] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[2] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[2] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[2] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[2] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[2] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[2] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[2] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[2] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[2] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[2] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[2] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[2] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[2] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[2] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[2] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[2] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[2] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[2] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[2] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[2] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[2] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[2] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[2] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[2] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[2] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[2] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[2] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[2] = static_cast<bool>(cur_param.u.bd.value);

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

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[3] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[3] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[3] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[3] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[3] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[3] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[3] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[3] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[3] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[3] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[3] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[3] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[3] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[3] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[3] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[3] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[3] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[3] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[3] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[3] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[3] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[3] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[3] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[3] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[3] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[3] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[3] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[3] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[3] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[3] = static_cast<bool>(cur_param.u.bd.value);

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

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[4] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[4] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[4] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[4] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[4] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[4] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[4] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[4] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[4] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[4] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[4] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[4] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[4] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[4] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[4] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[4] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[4] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[4] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[4] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[4] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[4] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[4] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[4] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[4] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[4] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[4] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[4] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[4] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[4] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[4] = static_cast<bool>(cur_param.u.bd.value);

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


		////////////////////////////
		// Local Light Settings 6 //
		////////////////////////////

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[5] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[5] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[5] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[5] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[5] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[5] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[5] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[5] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[5] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[5] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[5] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[5] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[5] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[5] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[5] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[5] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[5] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[5] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[5] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[5] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[5] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[5] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[5] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[5] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[5] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[5] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[5] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[5] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[5] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[5] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[5] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[5] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[5] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[5] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[5] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[5] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[5] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[5] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[5] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[5] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[5] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[5] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[5] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[5] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[5] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[5] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[5] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[5] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[5] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[5] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[5] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);


		////////////////////////////
		// Local Light Settings 7 //
		////////////////////////////

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[6] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[6] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[6] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[6] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[6] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[6] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[6] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[6] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[6] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[6] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[6] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[6] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[6] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[6] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[6] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[6] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[6] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[6] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[6] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[6] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[6] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[6] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[6] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[6] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[6] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[6] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[6] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[6] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[6] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[6] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[6] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[6] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[6] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[6] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[6] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[6] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[6] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[6] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[6] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[6] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[6] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[6] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[6] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[6] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[6] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[6] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[6] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[6] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[6] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[6] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[6] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);


		////////////////////////////
		// Local Light Settings 8 //
		////////////////////////////

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[7] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[7] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[7] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[7] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[7] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[7] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[7] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[7] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[7] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[7] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[7] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[7] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[7] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[7] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[7] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[7] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[7] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[7] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[7] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[7] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[7] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[7] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[7] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[7] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[7] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[7] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[7] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[7] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[7] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[7] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[7] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[7] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[7] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[7] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[7] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[7] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[7] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[7] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[7] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[7] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[7] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[7] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[7] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[7] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[7] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[7] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[7] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[7] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[7] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[7] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[7] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);


		////////////////////////////
		// Local Light Settings 9 //
		////////////////////////////

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[8] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[8] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[8] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[8] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[8] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[8] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[8] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[8] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[8] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[8] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[8] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[8] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[8] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[8] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[8] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[8] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[8] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[8] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[8] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[8] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[8] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[8] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[8] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[8] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[8] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[8] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[8] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[8] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[8] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[8] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[8] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[8] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[8] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[8] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[8] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[8] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[8] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[8] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[8] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[8] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[8] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[8] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[8] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[8] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[8] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[8] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[8] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[8] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[8] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[8] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[8] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);


		////////////////////////////
		// Local Light Settings 10 //
		////////////////////////////

		// Main

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[9] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[9] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[9] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VX_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVxX[9] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVxY[9] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVxZ[9] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VY_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVyX[9] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVyY[9] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVyZ[9] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_VZ_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightVzX[9] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightVzY[9] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightVzZ[9] = static_cast<float>(cur_param.u.point3d_d.z_value);

		// Shape

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->length[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleX[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->angleY[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->curvature[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[9] = static_cast<float>(cur_param.u.fs_d.value);

		// IES

		ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ies[9] = static_cast<int>(cur_param.u.pd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness1[9] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance1[9] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness2[9] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance2[9] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness3[9] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance3[9] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness4[9] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance4[9] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness5[9] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance5[9] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->iesBrightness6[9] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
		infoP->iesDistance6[9] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

		// Ambient

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_INTENSITY_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientIntensity[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_SATURATION_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientSaturation[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_NEAR_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorNearR[9] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorNearG[9] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorNearB[9] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_AMBIENT_COLOR_FAR_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_AMBIENT_COLOR_FAR_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFarR[9] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->ambientColorFarG[9] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->ambientColorFarB[9] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_AMBIENT_COLOR_FALLOFF_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->ambientColorFalloff[9] = static_cast<float>(cur_param.u.fs_d.value);

		// Diffuse

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_INTENSITY_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseIntensity[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_SATURATION_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseSaturation[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_NEAR_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorNearR[9] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorNearG[9] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorNearB[9] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DIFFUSE_COLOR_FAR_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_DIFFUSE_COLOR_FAR_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFarR[9] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->diffuseColorFarG[9] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->diffuseColorFarB[9] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DIFFUSE_COLOR_FALLOFF_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->diffuseColorFalloff[9] = static_cast<float>(cur_param.u.fs_d.value);

		// Specular

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SIZE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSize[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_INTENSITY_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularIntensity[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_SATURATION_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularSaturation[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_NEAR_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorNearR[9] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorNearG[9] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorNearB[9] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SPECULAR_COLOR_FAR_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SPECULAR_COLOR_FAR_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFarR[9] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->specularColorFarG[9] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->specularColorFarB[9] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SPECULAR_COLOR_FALLOFF_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->specularColorFalloff[9] = static_cast<float>(cur_param.u.fs_d.value);

		// Shadows

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_AMBIENT_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreAmbientToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_DIFFUSE_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreDiffuseToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_IGNORE_SPECULAR_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIgnoreSpecularToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_SHADOW_CLIP_TO_LIGHT_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowClipToLightToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SAMPLE_STEP_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSampleStep[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_IMPROVED_SAMPLE_RADIUS_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowImprovedSampleRadius[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_MAX_LENGTH_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowMaxLength[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_START_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdStart[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_THRESHOLD_END_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowThresholdEnd[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_RADIUS_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessRadius[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_SOFTNESS_SAMPLES_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowSoftnessSamples[9] = static_cast<int>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SHADOW_INTENSITY_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowIntensity[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_SHADOW_COLOR_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->shadowColorR[9] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->shadowColorG[9] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->shadowColorB[9] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);
		
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
		"r-spot-advanced", // Name
		"ADBE r-spot-advanced", // Match Name
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
