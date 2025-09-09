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
extern void ProcAmp_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int	is16f,
	unsigned int width,
	unsigned int height,

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
		"Radius Multiplier",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_MULTIPLIER
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

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 1", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 1", 0, 0, 0, POINT_3D_LIGHT_POSITION_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 1", FALSE, 0, CHECKBOX_INVERT_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_1
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 1", 255, 205, 120, COLOR_COLOR_NEAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 1", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 1", 255, 157, 0, COLOR_COLOR_FAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_1
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_1);


	////////////////////////////
	// Local Light Settings 2 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 2", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 2", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 2", 0, 0, 0, POINT_3D_LIGHT_POSITION_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 2", FALSE, 0, CHECKBOX_INVERT_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_2
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 2", 255, 205, 120, COLOR_COLOR_NEAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 2", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 2", 255, 157, 0, COLOR_COLOR_FAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_2
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_2);


	////////////////////////////
	// Local Light Settings 3 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 3", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 3", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 3", 0, 0, 0, POINT_3D_LIGHT_POSITION_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 3", FALSE, 0, CHECKBOX_INVERT_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_3
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 3", 255, 205, 120, COLOR_COLOR_NEAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 3", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 3", 255, 157, 0, COLOR_COLOR_FAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_3
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_3);


	////////////////////////////
	// Local Light Settings 4 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 4", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 4", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 4", 0, 0, 0, POINT_3D_LIGHT_POSITION_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 4", FALSE, 0, CHECKBOX_INVERT_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_4
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 4", 255, 205, 120, COLOR_COLOR_NEAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 4", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 4", 255, 157, 0, COLOR_COLOR_FAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_4
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_4);


	////////////////////////////
	// Local Light Settings 5 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 5", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 5", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 5", 0, 0, 0, POINT_3D_LIGHT_POSITION_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 5", FALSE, 0, CHECKBOX_INVERT_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_5
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 5", 255, 205, 120, COLOR_COLOR_NEAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 5", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 5", 255, 157, 0, COLOR_COLOR_FAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_5
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_5);


	////////////////////////////
	// Local Light Settings 6 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 6", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 6", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 6", 0, 0, 0, POINT_3D_LIGHT_POSITION_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 6", FALSE, 0, CHECKBOX_INVERT_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_6
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_6
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 6", 255, 205, 120, COLOR_COLOR_NEAR_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 6", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 6", 255, 157, 0, COLOR_COLOR_FAR_6);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 6",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_6
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_6);


	////////////////////////////
	// Local Light Settings 7 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 7", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 7", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 7", 0, 0, 0, POINT_3D_LIGHT_POSITION_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 7", FALSE, 0, CHECKBOX_INVERT_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_7
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_7
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 7", 255, 205, 120, COLOR_COLOR_NEAR_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 7", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 7", 255, 157, 0, COLOR_COLOR_FAR_7);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 7",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_7
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_7);


	////////////////////////////
	// Local Light Settings 8 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 8", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 8", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 8", 0, 0, 0, POINT_3D_LIGHT_POSITION_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 8", FALSE, 0, CHECKBOX_INVERT_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_8
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_8
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 8", 255, 205, 120, COLOR_COLOR_NEAR_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 8", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 8", 255, 157, 0, COLOR_COLOR_FAR_8);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 8",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_8
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_8);


	////////////////////////////
	// Local Light Settings 9 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 9", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 9", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 9", 0, 0, 0, POINT_3D_LIGHT_POSITION_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 9", FALSE, 0, CHECKBOX_INVERT_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_9
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_9
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 9", 255, 205, 120, COLOR_COLOR_NEAR_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 9", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 9", 255, 157, 0, COLOR_COLOR_FAR_9);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 9",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_9
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_9);


	////////////////////////////
	// Local Light Settings 10 //
	////////////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Settings 10", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Light Toggle 10", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Light Position 10", 0, 0, 0, POINT_3D_LIGHT_POSITION_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert Toggle 10", FALSE, 0, CHECKBOX_INVERT_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Radius 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1000.000),
		PF_FpLong(200.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RADIUS_10
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
	PF_ADD_FLOAT_SLIDERX(
		"Intensity 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_INTENSITY_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Saturation 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_SATURATION_10
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Near 10", 255, 205, 120, COLOR_COLOR_NEAR_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Color Far Toggle 10", FALSE, 0, CHECKBOX_COLOR_FAR_TOGGLE_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Color Far 10", 255, 157, 0, COLOR_COLOR_FAR_10);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Color Falloff 10",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_COLOR_FALLOFF_10
	);

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
	PF_CheckoutResult in_result;
	PF_RenderRequest req = extraP->input->output_request;

	extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

	InvertProcAmpParams *infoP	= reinterpret_cast<InvertProcAmpParams *>( malloc(sizeof(InvertProcAmpParams)) );

	if (infoP) {

		// Camera

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

		// Depth Settings

		PF_ParamDef cur_param;

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_DEPTH_FAR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->depthFar = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_DEPTH_BLACK_IS_NEAR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->depthBlackIsNear = static_cast<bool>(cur_param.u.bd.value);

		// Global Light Settings

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radiusMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloffMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensityMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturationMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloffMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		// Local Light Settings 1

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[0] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[0] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[0] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[0] = static_cast<float>(cur_param.u.fs_d.value);


		// Local Light Settings 2

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[1] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[1] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[1] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[1] = static_cast<float>(cur_param.u.fs_d.value);


		// Local Light Settings 3

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[2] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[2] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[2] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[2] = static_cast<float>(cur_param.u.fs_d.value);


		// Local Light Settings 4

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[3] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[3] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[3] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[3] = static_cast<float>(cur_param.u.fs_d.value);


		// Local Light Settings 5

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[4] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[4] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[4] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[4] = static_cast<float>(cur_param.u.fs_d.value);


		// Local Light Settings 6

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[5] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[5] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[5] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[5] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[5] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[5] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[5] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[5] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[5] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[5] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[5] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_6, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[5] = static_cast<float>(cur_param.u.fs_d.value);


		// Local Light Settings 7

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[6] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[6] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[6] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[6] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[6] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[6] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[6] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[6] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[6] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[6] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[6] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_7, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[6] = static_cast<float>(cur_param.u.fs_d.value);


		// Local Light Settings 8

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[7] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[7] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[7] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[7] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[7] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[7] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[7] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[7] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[7] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[7] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[7] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_8, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[7] = static_cast<float>(cur_param.u.fs_d.value);


		// Local Light Settings 9

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[8] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[8] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[8] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[8] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[8] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[8] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[8] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[8] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[8] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[8] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[8] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_9, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[8] = static_cast<float>(cur_param.u.fs_d.value);


		// Local Light Settings 10

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->lightPosX[9] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->lightPosY[9] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->lightPosZ[9] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->invertToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->falloff[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensity[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturation[9] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_NEAR_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorNearR[9] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorNearG[9] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorNearB[9] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_TOGGLE_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarToggle[9] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_COLOR_FAR_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFarR[9] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->colorFarG[9] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->colorFarB[9] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FALLOFF_10, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->colorFalloff[9] = static_cast<float>(cur_param.u.fs_d.value);
		
		// End

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;
		
		ERR(extraP->cb->checkout_layer(	in_dataP->effect_ref,
									   RELIGHTING_INPUT,
									   RELIGHTING_INPUT,
									   &req,
									   in_dataP->current_time,
									   in_dataP->time_step,
									   in_dataP->time_scale,
									   &in_result));
		
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

			ProcAmp_CUDA(
				(const float*)src_mem,
				(float*)dst_mem,
				procamp_params.mSrcPitch,
				procamp_params.mDstPitch,
				procamp_params.m16f,
				procamp_params.mWidth,
				procamp_params.mHeight,

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
					*output_worldP  = NULL;

	// Parameters can be queried during render. In this example, we pass them from PreRender as an example of using pre_render_data.
	InvertProcAmpParams *infoP = reinterpret_cast<InvertProcAmpParams *>(extraP->input->pre_render_data);

	if (infoP) {
		ERR((extraP->cb->checkout_layer_pixels(	in_data->effect_ref, RELIGHTING_INPUT, &input_worldP)));
		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
																				kPFWorldSuite,
																				kPFWorldSuiteVersion2,
																				out_data);
		PF_PixelFormat	pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));

		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, RELIGHTING_INPUT));
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
		"r-point-simple", // Name
		"ADBE r-point-simple", // Match Name
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
