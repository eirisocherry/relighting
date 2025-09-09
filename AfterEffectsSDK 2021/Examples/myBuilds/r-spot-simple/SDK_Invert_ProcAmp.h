/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007 Adobe Systems Incorporated                       */
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

#pragma once
#ifndef SDK_Invert_ProcAmp_H
#define SDK_Invert_ProcAmp_H

#include "SDK_Invert_ProcAmp_Kernel.cl.h"
#include "AEConfig.h"
#include "entry.h"
#include "AEFX_SuiteHelper.h"
#include "PrSDKAESupport.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_Macros.h"
#include "AEGP_SuiteHandler.h"
#include "String_Utils.h"
#include "Param_Utils.h"
#include "Smart_Utils.h"


#if _WIN32
#include <CL/cl.h>
#define HAS_METAL 0
#else
#include <OpenCL/cl.h>
#define HAS_METAL 1
#include <Metal/Metal.h>
#include "SDK_Invert_ProcAmp_Kernel.metal.h"
#endif
#include <math.h>

#include "AEConfig.h"
#include "entry.h"
#ifdef AE_OS_WIN
	#include <Windows.h>
#endif
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEGP_SuiteHandler.h"


#define DESCRIPTION	""

#define NAME			"r-spot-simple"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	RELIGHTING_INPUT=0,

	// Depth settings

	TOPIC_ADD_DEPTH_SETTINGS,
	SLIDER_DEPTH_FAR,
	CHECKBOX_DEPTH_BLACK_IS_NEAR,
	TOPIC_END_DEPTH_SETTINGS,

	// Light Layers

	TOPIC_ADD_LIGHT_LAYERS,
	LAYER_LIGHT_1,
	LAYER_LIGHT_2,
	LAYER_LIGHT_3,
	LAYER_LIGHT_4,
	LAYER_LIGHT_5,
	LAYER_LIGHT_6,
	LAYER_LIGHT_7,
	LAYER_LIGHT_8,
	LAYER_LIGHT_9,
	LAYER_LIGHT_10,
	TOPIC_END_LIGHT_LAYERS,

	// Global Light Settings

	TOPIC_ADD_GLOBAL_LIGHT_SETTINGS,
	SLIDER_LENGTH_MULTIPLIER,
	SLIDER_ANGLEX_MULTIPLIER,
	SLIDER_ANGLEY_MULTIPLIER,
	SLIDER_CURVATURE_MULTIPLIER,
	SLIDER_FEATHER_MULTIPLIER,
	SLIDER_FALLOFF_MULTIPLIER,
	SLIDER_INTENSITY_MULTIPLIER,
	SLIDER_SATURATION_MULTIPLIER,
	SLIDER_COLOR_FALLOFF_MULTIPLIER,
	TOPIC_END_GLOBAL_LIGHT_SETTINGS,
	
	// Local Light Settings 1

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_1,
	CHECKBOX_TOGGLE_1,
	POINT_3D_POSITION_1,
	POINT_3D_VX_1,
	POINT_3D_VY_1,
	POINT_3D_VZ_1,
	CHECKBOX_INVERT_1, //---
	SLIDER_LENGTH_1,
	SLIDER_ANGLEX_1,
	SLIDER_ANGLEY_1,
	SLIDER_CURVATURE_1,
	SLIDER_FEATHER_1,
	SLIDER_FALLOFF_1,
	POPUP_IES_1,	//---
	POINT_2D_IES_1_1,
	POINT_2D_IES_2_1,
	POINT_2D_IES_3_1,
	POINT_2D_IES_4_1,
	POINT_2D_IES_5_1,
	POINT_2D_IES_6_1,
	SLIDER_INTENSITY_1, //---
	SLIDER_SATURATION_1,
	COLOR_COLOR_NEAR_1,
	CHECKBOX_COLOR_FAR_TOGGLE_1,
	COLOR_COLOR_FAR_1,
	SLIDER_COLOR_FALLOFF_1,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_1,


	// Local Light Settings 2

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_2,
	CHECKBOX_TOGGLE_2,
	POINT_3D_POSITION_2,
	POINT_3D_VX_2,
	POINT_3D_VY_2,
	POINT_3D_VZ_2,
	CHECKBOX_INVERT_2, //---
	SLIDER_LENGTH_2,
	SLIDER_ANGLEX_2,
	SLIDER_ANGLEY_2,
	SLIDER_CURVATURE_2,
	SLIDER_FEATHER_2,
	SLIDER_FALLOFF_2,
	POPUP_IES_2,	//---
	POINT_2D_IES_1_2,
	POINT_2D_IES_2_2,
	POINT_2D_IES_3_2,
	POINT_2D_IES_4_2,
	POINT_2D_IES_5_2,
	POINT_2D_IES_6_2,
	SLIDER_INTENSITY_2, //---
	SLIDER_SATURATION_2,
	COLOR_COLOR_NEAR_2,
	CHECKBOX_COLOR_FAR_TOGGLE_2,
	COLOR_COLOR_FAR_2,
	SLIDER_COLOR_FALLOFF_2,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_2,


	// Local Light Settings 3

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_3,
	CHECKBOX_TOGGLE_3,
	POINT_3D_POSITION_3,
	POINT_3D_VX_3,
	POINT_3D_VY_3,
	POINT_3D_VZ_3,
	CHECKBOX_INVERT_3, //---
	SLIDER_LENGTH_3,
	SLIDER_ANGLEX_3,
	SLIDER_ANGLEY_3,
	SLIDER_CURVATURE_3,
	SLIDER_FEATHER_3,
	SLIDER_FALLOFF_3,
	POPUP_IES_3,	//---
	POINT_2D_IES_1_3,
	POINT_2D_IES_2_3,
	POINT_2D_IES_3_3,
	POINT_2D_IES_4_3,
	POINT_2D_IES_5_3,
	POINT_2D_IES_6_3,
	SLIDER_INTENSITY_3, //---
	SLIDER_SATURATION_3,
	COLOR_COLOR_NEAR_3,
	CHECKBOX_COLOR_FAR_TOGGLE_3,
	COLOR_COLOR_FAR_3,
	SLIDER_COLOR_FALLOFF_3,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_3,


	// Local Light Settings 4

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_4,
	CHECKBOX_TOGGLE_4,
	POINT_3D_POSITION_4,
	POINT_3D_VX_4,
	POINT_3D_VY_4,
	POINT_3D_VZ_4,
	CHECKBOX_INVERT_4, //---
	SLIDER_LENGTH_4,
	SLIDER_ANGLEX_4,
	SLIDER_ANGLEY_4,
	SLIDER_CURVATURE_4,
	SLIDER_FEATHER_4,
	SLIDER_FALLOFF_4,
	POPUP_IES_4,	//---
	POINT_2D_IES_1_4,
	POINT_2D_IES_2_4,
	POINT_2D_IES_3_4,
	POINT_2D_IES_4_4,
	POINT_2D_IES_5_4,
	POINT_2D_IES_6_4,
	SLIDER_INTENSITY_4, //---
	SLIDER_SATURATION_4,
	COLOR_COLOR_NEAR_4,
	CHECKBOX_COLOR_FAR_TOGGLE_4,
	COLOR_COLOR_FAR_4,
	SLIDER_COLOR_FALLOFF_4,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_4,


	// Local Light Settings 5

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_5,
	CHECKBOX_TOGGLE_5,
	POINT_3D_POSITION_5,
	POINT_3D_VX_5,
	POINT_3D_VY_5,
	POINT_3D_VZ_5,
	CHECKBOX_INVERT_5, //---
	SLIDER_LENGTH_5,
	SLIDER_ANGLEX_5,
	SLIDER_ANGLEY_5,
	SLIDER_CURVATURE_5,
	SLIDER_FEATHER_5,
	SLIDER_FALLOFF_5,
	POPUP_IES_5,	//---
	POINT_2D_IES_1_5,
	POINT_2D_IES_2_5,
	POINT_2D_IES_3_5,
	POINT_2D_IES_4_5,
	POINT_2D_IES_5_5,
	POINT_2D_IES_6_5,
	SLIDER_INTENSITY_5, //---
	SLIDER_SATURATION_5,
	COLOR_COLOR_NEAR_5,
	CHECKBOX_COLOR_FAR_TOGGLE_5,
	COLOR_COLOR_FAR_5,
	SLIDER_COLOR_FALLOFF_5,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_5,


	// Local Light Settings 6

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_6,
	CHECKBOX_TOGGLE_6,
	POINT_3D_POSITION_6,
	POINT_3D_VX_6,
	POINT_3D_VY_6,
	POINT_3D_VZ_6,
	CHECKBOX_INVERT_6, //---
	SLIDER_LENGTH_6,
	SLIDER_ANGLEX_6,
	SLIDER_ANGLEY_6,
	SLIDER_CURVATURE_6,
	SLIDER_FEATHER_6,
	SLIDER_FALLOFF_6,
	POPUP_IES_6,	//---
	POINT_2D_IES_1_6,
	POINT_2D_IES_2_6,
	POINT_2D_IES_3_6,
	POINT_2D_IES_4_6,
	POINT_2D_IES_5_6,
	POINT_2D_IES_6_6,
	SLIDER_INTENSITY_6, //---
	SLIDER_SATURATION_6,
	COLOR_COLOR_NEAR_6,
	CHECKBOX_COLOR_FAR_TOGGLE_6,
	COLOR_COLOR_FAR_6,
	SLIDER_COLOR_FALLOFF_6,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_6,


	// Local Light Settings 7

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_7,
	CHECKBOX_TOGGLE_7,
	POINT_3D_POSITION_7,
	POINT_3D_VX_7,
	POINT_3D_VY_7,
	POINT_3D_VZ_7,
	CHECKBOX_INVERT_7, //---
	SLIDER_LENGTH_7,
	SLIDER_ANGLEX_7,
	SLIDER_ANGLEY_7,
	SLIDER_CURVATURE_7,
	SLIDER_FEATHER_7,
	SLIDER_FALLOFF_7,
	POPUP_IES_7,	//---
	POINT_2D_IES_1_7,
	POINT_2D_IES_2_7,
	POINT_2D_IES_3_7,
	POINT_2D_IES_4_7,
	POINT_2D_IES_5_7,
	POINT_2D_IES_6_7,
	SLIDER_INTENSITY_7, //---
	SLIDER_SATURATION_7,
	COLOR_COLOR_NEAR_7,
	CHECKBOX_COLOR_FAR_TOGGLE_7,
	COLOR_COLOR_FAR_7,
	SLIDER_COLOR_FALLOFF_7,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_7,


	// Local Light Settings 8

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_8,
	CHECKBOX_TOGGLE_8,
	POINT_3D_POSITION_8,
	POINT_3D_VX_8,
	POINT_3D_VY_8,
	POINT_3D_VZ_8,
	CHECKBOX_INVERT_8, //---
	SLIDER_LENGTH_8,
	SLIDER_ANGLEX_8,
	SLIDER_ANGLEY_8,
	SLIDER_CURVATURE_8,
	SLIDER_FEATHER_8,
	SLIDER_FALLOFF_8,
	POPUP_IES_8,	//---
	POINT_2D_IES_1_8,
	POINT_2D_IES_2_8,
	POINT_2D_IES_3_8,
	POINT_2D_IES_4_8,
	POINT_2D_IES_5_8,
	POINT_2D_IES_6_8,
	SLIDER_INTENSITY_8, //---
	SLIDER_SATURATION_8,
	COLOR_COLOR_NEAR_8,
	CHECKBOX_COLOR_FAR_TOGGLE_8,
	COLOR_COLOR_FAR_8,
	SLIDER_COLOR_FALLOFF_8,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_8,


	// Local Light Settings 9

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_9,
	CHECKBOX_TOGGLE_9,
	POINT_3D_POSITION_9,
	POINT_3D_VX_9,
	POINT_3D_VY_9,
	POINT_3D_VZ_9,
	CHECKBOX_INVERT_9, //---
	SLIDER_LENGTH_9,
	SLIDER_ANGLEX_9,
	SLIDER_ANGLEY_9,
	SLIDER_CURVATURE_9,
	SLIDER_FEATHER_9,
	SLIDER_FALLOFF_9,
	POPUP_IES_9,	//---
	POINT_2D_IES_1_9,
	POINT_2D_IES_2_9,
	POINT_2D_IES_3_9,
	POINT_2D_IES_4_9,
	POINT_2D_IES_5_9,
	POINT_2D_IES_6_9,
	SLIDER_INTENSITY_9, //---
	SLIDER_SATURATION_9,
	COLOR_COLOR_NEAR_9,
	CHECKBOX_COLOR_FAR_TOGGLE_9,
	COLOR_COLOR_FAR_9,
	SLIDER_COLOR_FALLOFF_9,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_9,


	// Local Light Settings 10

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_10,
	CHECKBOX_TOGGLE_10,
	POINT_3D_POSITION_10,
	POINT_3D_VX_10,
	POINT_3D_VY_10,
	POINT_3D_VZ_10,
	CHECKBOX_INVERT_10, //---
	SLIDER_LENGTH_10,
	SLIDER_ANGLEX_10,
	SLIDER_ANGLEY_10,
	SLIDER_CURVATURE_10,
	SLIDER_FEATHER_10,
	SLIDER_FALLOFF_10,
	POPUP_IES_10,	//---
	POINT_2D_IES_1_10,
	POINT_2D_IES_2_10,
	POINT_2D_IES_3_10,
	POINT_2D_IES_4_10,
	POINT_2D_IES_5_10,
	POINT_2D_IES_6_10,
	SLIDER_INTENSITY_10, //---
	SLIDER_SATURATION_10,
	COLOR_COLOR_NEAR_10,
	CHECKBOX_COLOR_FAR_TOGGLE_10,
	COLOR_COLOR_FAR_10,
	SLIDER_COLOR_FALLOFF_10,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_10,

	RELIGHTING_NUM_PARAMS
};

extern "C" {

	DllExport 
	PF_Err
	EffectMain (	
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);

}

#if HAS_METAL
	/*
	 ** Plugins must not rely on a host autorelease pool.
	 ** Create a pool if autorelease is used, or Cocoa convention calls, such as Metal, might internally autorelease.
	 */
	struct ScopedAutoreleasePool
	{
		ScopedAutoreleasePool()
		:  mPool([[NSAutoreleasePool alloc] init])
		{
		}
	
		~ScopedAutoreleasePool()
		{
			[mPool release];
		}
	
		NSAutoreleasePool *mPool;
	};
#endif 

#endif // SDK_Invert_ProcAmp_H
