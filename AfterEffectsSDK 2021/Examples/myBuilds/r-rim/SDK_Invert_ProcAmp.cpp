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

	//////////////////
	// Light Layers //
	//////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Light Layers", TOPIC_ADD_LIGHT_LAYERS);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 1", 1, LAYER_LIGHT_1);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Look At 1", 1, LAYER_LIGHT_LOOK_AT_1);
	
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 2", 1, LAYER_LIGHT_2);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Look At 2", 1, LAYER_LIGHT_LOOK_AT_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 3", 1, LAYER_LIGHT_3);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Look At 3", 1, LAYER_LIGHT_LOOK_AT_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 4", 1, LAYER_LIGHT_4);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Look At 4", 1, LAYER_LIGHT_LOOK_AT_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light 5", 1, LAYER_LIGHT_5);
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Light Look At 5", 1, LAYER_LIGHT_LOOK_AT_5);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_LIGHT_LAYERS);

	/////////////////////
	// Global Settings //
	/////////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Global Light Settings", TOPIC_ADD_GLOBAL_SETTINGS);

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
	PF_END_TOPIC(TOPIC_END_GLOBAL_SETTINGS);

	/////////////////
	// Rim Light 1 //
	/////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Rim Light 1", TOPIC_ADD_RIM_SETTINGS_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Toggle 1", FALSE, 0, CHECKBOX_RIM_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Position 1", 0, 0, 0, POINT_3D_LIGHT_POSITION_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Look At Position 1", 0, 0, 0, POINT_3D_LIGHT_LOOK_AT_POSITION_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Start 1",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_START_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim End 1",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_END_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Intensity 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_INTENSITY_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Saturation 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_SATURATION_1
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Near 1", 255, 205, 120, COLOR_RIM_COLOR_NEAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Color Far Toggle 1", TRUE, 0, CHECKBOX_RIM_COLOR_FAR_TOGGLE_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Far 1", 255, 157, 0, COLOR_RIM_COLOR_FAR_1);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Color Falloff 1",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_COLOR_FALLOFF_1
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_RIM_SETTINGS_1);


	/////////////////
	// Rim Light 2 //
	/////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Rim Light 2", TOPIC_ADD_RIM_SETTINGS_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Toggle 2", FALSE, 0, CHECKBOX_RIM_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Position 2", 0, 0, 0, POINT_3D_LIGHT_POSITION_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Look At Position 2", 0, 0, 0, POINT_3D_LIGHT_LOOK_AT_POSITION_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Start 2",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_START_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim End 2",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_END_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Intensity 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_INTENSITY_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Saturation 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_SATURATION_2
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Near 2", 255, 205, 120, COLOR_RIM_COLOR_NEAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Color Far Toggle 2", TRUE, 0, CHECKBOX_RIM_COLOR_FAR_TOGGLE_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Far 2", 255, 157, 0, COLOR_RIM_COLOR_FAR_2);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Color Falloff 2",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_COLOR_FALLOFF_2
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_RIM_SETTINGS_2);


	/////////////////
	// Rim Light 3 //
	/////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Rim Light 3", TOPIC_ADD_RIM_SETTINGS_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Toggle 3", FALSE, 0, CHECKBOX_RIM_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Position 3", 0, 0, 0, POINT_3D_LIGHT_POSITION_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Look At Position 3", 0, 0, 0, POINT_3D_LIGHT_LOOK_AT_POSITION_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Start 3",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_START_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim End 3",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_END_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Intensity 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_INTENSITY_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Saturation 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_SATURATION_3
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Near 3", 255, 205, 120, COLOR_RIM_COLOR_NEAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Color Far Toggle 3", TRUE, 0, CHECKBOX_RIM_COLOR_FAR_TOGGLE_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Far 3", 255, 157, 0, COLOR_RIM_COLOR_FAR_3);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Color Falloff 3",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_COLOR_FALLOFF_3
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_RIM_SETTINGS_3);


	/////////////////
	// Rim Light 4 //
	/////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Rim Light 4", TOPIC_ADD_RIM_SETTINGS_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Toggle 4", FALSE, 0, CHECKBOX_RIM_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Position 4", 0, 0, 0, POINT_3D_LIGHT_POSITION_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Look At Position 4", 0, 0, 0, POINT_3D_LIGHT_LOOK_AT_POSITION_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Start 4",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_START_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim End 4",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_END_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Intensity 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_INTENSITY_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Saturation 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_SATURATION_4
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Near 4", 255, 205, 120, COLOR_RIM_COLOR_NEAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Color Far Toggle 4", TRUE, 0, CHECKBOX_RIM_COLOR_FAR_TOGGLE_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Far 4", 255, 157, 0, COLOR_RIM_COLOR_FAR_4);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Color Falloff 4",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_COLOR_FALLOFF_4
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_RIM_SETTINGS_4);


	/////////////////
	// Rim Light 5 //
	/////////////////

	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Rim Light 5", TOPIC_ADD_RIM_SETTINGS_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Toggle 5", FALSE, 0, CHECKBOX_RIM_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Position 5", 0, 0, 0, POINT_3D_LIGHT_POSITION_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Rim Light Look At Position 5", 0, 0, 0, POINT_3D_LIGHT_LOOK_AT_POSITION_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Start 5",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(0.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_START_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim End 5",
		PF_FpLong(-9999999.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_END_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Intensity 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_INTENSITY_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Saturation 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_SATURATION_5
	);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Near 5", 255, 205, 120, COLOR_RIM_COLOR_NEAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Rim Color Far Toggle 5", TRUE, 0, CHECKBOX_RIM_COLOR_FAR_TOGGLE_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR("Rim Color Far 5", 255, 157, 0, COLOR_RIM_COLOR_FAR_5);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		"Rim Color Falloff 5",
		PF_FpLong(0.000),
		PF_FpLong(9999999.000),
		PF_FpLong(0.000),
		PF_FpLong(1.000),
		PF_FpLong(1.000),
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SLIDER_RIM_COLOR_FALLOFF_5
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(TOPIC_END_RIM_SETTINGS_5);

	// End

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
		
		/////////////////////
		// Global Settings //
		/////////////////////

		PF_ParamDef cur_param;

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->intensityMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_MULTIPLIER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->saturationMultiplier = static_cast<float>(cur_param.u.fs_d.value);

		/////////////////
		// Rim Light 1 //
		/////////////////

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightPositionX[0] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightPositionY[0] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightPositionZ[0] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_LOOK_AT_POSITION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightLookAtPositionX[0] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightLookAtPositionY[0] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightLookAtPositionZ[0] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_START_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimStart[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_END_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimEnd[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_INTENSITY_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimIntensity[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_SATURATION_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimSaturation[0] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_NEAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorNearR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorNearG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorNearB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_COLOR_FAR_TOGGLE_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarToggle[0] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_FAR_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarR[0] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorFarG[0] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorFarB[0] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_COLOR_FALLOFF_1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFalloff[0] = static_cast<float>(cur_param.u.fs_d.value);


		/////////////////
		// Rim Light 2 //
		/////////////////

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightPositionX[1] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightPositionY[1] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightPositionZ[1] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_LOOK_AT_POSITION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightLookAtPositionX[1] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightLookAtPositionY[1] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightLookAtPositionZ[1] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_START_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimStart[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_END_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimEnd[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_INTENSITY_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimIntensity[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_SATURATION_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimSaturation[1] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_NEAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorNearR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorNearG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorNearB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_COLOR_FAR_TOGGLE_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarToggle[1] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_FAR_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarR[1] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorFarG[1] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorFarB[1] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_COLOR_FALLOFF_2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFalloff[1] = static_cast<float>(cur_param.u.fs_d.value);


		/////////////////
		// Rim Light 3 //
		/////////////////

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightPositionX[2] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightPositionY[2] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightPositionZ[2] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_LOOK_AT_POSITION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightLookAtPositionX[2] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightLookAtPositionY[2] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightLookAtPositionZ[2] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_START_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimStart[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_END_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimEnd[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_INTENSITY_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimIntensity[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_SATURATION_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimSaturation[2] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_NEAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorNearR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorNearG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorNearB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_COLOR_FAR_TOGGLE_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarToggle[2] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_FAR_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarR[2] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorFarG[2] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorFarB[2] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_COLOR_FALLOFF_3, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFalloff[2] = static_cast<float>(cur_param.u.fs_d.value);


		/////////////////
		// Rim Light 4 //
		/////////////////

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightPositionX[3] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightPositionY[3] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightPositionZ[3] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_LOOK_AT_POSITION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightLookAtPositionX[3] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightLookAtPositionY[3] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightLookAtPositionZ[3] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_START_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimStart[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_END_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimEnd[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_INTENSITY_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimIntensity[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_SATURATION_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimSaturation[3] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_NEAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorNearR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorNearG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorNearB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_COLOR_FAR_TOGGLE_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarToggle[3] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_FAR_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarR[3] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorFarG[3] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorFarB[3] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_COLOR_FALLOFF_4, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFalloff[3] = static_cast<float>(cur_param.u.fs_d.value);


		/////////////////
		// Rim Light 5 //
		/////////////////

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_POSITION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightPositionX[4] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightPositionY[4] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightPositionZ[4] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_LIGHT_LOOK_AT_POSITION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimLightLookAtPositionX[4] = static_cast<float>(cur_param.u.point3d_d.x_value);
		infoP->rimLightLookAtPositionY[4] = static_cast<float>(cur_param.u.point3d_d.y_value);
		infoP->rimLightLookAtPositionZ[4] = static_cast<float>(cur_param.u.point3d_d.z_value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_START_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimStart[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_END_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimEnd[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_INTENSITY_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimIntensity[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_SATURATION_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimSaturation[4] = static_cast<float>(cur_param.u.fs_d.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_NEAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorNearR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorNearG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorNearB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_RIM_COLOR_FAR_TOGGLE_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarToggle[4] = static_cast<bool>(cur_param.u.bd.value);

		ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_RIM_COLOR_FAR_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFarR[4] = static_cast<float>(cur_param.u.cd.value.red / 255.0f);
		infoP->rimColorFarG[4] = static_cast<float>(cur_param.u.cd.value.green / 255.0f);
		infoP->rimColorFarB[4] = static_cast<float>(cur_param.u.cd.value.blue / 255.0f);

		ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RIM_COLOR_FALLOFF_5, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rimColorFalloff[4] = static_cast<float>(cur_param.u.fs_d.value);

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
		"r-rim", // Name
		"ADBE r-rim", // Match Name
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
