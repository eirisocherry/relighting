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

#define NAME			"r-depth2rect-matte"
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
	LAYER_LIGHT_START_1,
	LAYER_LIGHT_END_1,
	LAYER_LIGHT_START_2,
	LAYER_LIGHT_END_2,
	LAYER_LIGHT_START_3,
	LAYER_LIGHT_END_3,
	LAYER_LIGHT_START_4,
	LAYER_LIGHT_END_4,
	LAYER_LIGHT_START_5,
	LAYER_LIGHT_END_5,
	LAYER_LIGHT_START_6,
	LAYER_LIGHT_END_6,
	LAYER_LIGHT_START_7,
	LAYER_LIGHT_END_7,
	LAYER_LIGHT_START_8,
	LAYER_LIGHT_END_8,
	LAYER_LIGHT_START_9,
	LAYER_LIGHT_END_9,
	LAYER_LIGHT_START_10,
	LAYER_LIGHT_END_10,

	TOPIC_END_LIGHT_LAYERS,

	// Global Light Settings

	TOPIC_ADD_GLOBAL_LIGHT_SETTINGS,
	SLIDER_FEATHER_MULTIPLIER,
	SLIDER_FALLOFF_MULTIPLIER,
	SLIDER_INTENSITY_MULTIPLIER,
	SLIDER_SATURATION_MULTIPLIER,
	SLIDER_COLOR_FALLOFF_MULTIPLIER,
	TOPIC_END_GLOBAL_LIGHT_SETTINGS,
	
	// Local Light Settings 1

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_1,
	CHECKBOX_LIGHT_TOGGLE_1,
	POINT_3D_POS1_1, //---
	POINT_3D_VX1_1,
	POINT_3D_VY1_1,
	POINT_3D_VZ1_1,
	POINT_3D_RES1_1,
	POINT_3D_SCALE1_1,
	POINT_3D_POS2_1, //---
	POINT_3D_VX2_1,
	POINT_3D_VY2_1,
	POINT_3D_VZ2_1,
	POINT_3D_RES2_1,
	POINT_3D_SCALE2_1,
	CHECKBOX_INVERT_1, //---
	CHECKBOX_FEATHER_NORMALIZE_1,
	POINT_2D_FEATHERX_1,
	POINT_2D_FEATHERY_1,
	POINT_2D_FEATHERZ_1,
	SLIDER_FALLOFF_1,
	SLIDER_INTENSITY_1, //---
	SLIDER_SATURATION_1,
	COLOR_NEAR_1,
	CHECKBOX_COLOR_FAR_1,
	COLOR_FAR_1,
	SLIDER_COLOR_FAR_FALLOFF_1,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_1,


	// Local Light Settings 2

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_2,
	CHECKBOX_LIGHT_TOGGLE_2,
	POINT_3D_POS1_2, //---
	POINT_3D_VX1_2,
	POINT_3D_VY1_2,
	POINT_3D_VZ1_2,
	POINT_3D_RES1_2,
	POINT_3D_SCALE1_2,
	POINT_3D_POS2_2, //---
	POINT_3D_VX2_2,
	POINT_3D_VY2_2,
	POINT_3D_VZ2_2,
	POINT_3D_RES2_2,
	POINT_3D_SCALE2_2,
	CHECKBOX_INVERT_2, //---
	CHECKBOX_FEATHER_NORMALIZE_2,
	POINT_2D_FEATHERX_2,
	POINT_2D_FEATHERY_2,
	POINT_2D_FEATHERZ_2,
	SLIDER_FALLOFF_2,
	SLIDER_INTENSITY_2, //---
	SLIDER_SATURATION_2,
	COLOR_NEAR_2,
	CHECKBOX_COLOR_FAR_2,
	COLOR_FAR_2,
	SLIDER_COLOR_FAR_FALLOFF_2,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_2,


	// Local Light Settings 3

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_3,
	CHECKBOX_LIGHT_TOGGLE_3,
	POINT_3D_POS1_3, //---
	POINT_3D_VX1_3,
	POINT_3D_VY1_3,
	POINT_3D_VZ1_3,
	POINT_3D_RES1_3,
	POINT_3D_SCALE1_3,
	POINT_3D_POS2_3, //---
	POINT_3D_VX2_3,
	POINT_3D_VY2_3,
	POINT_3D_VZ2_3,
	POINT_3D_RES2_3,
	POINT_3D_SCALE2_3,
	CHECKBOX_INVERT_3, //---
	CHECKBOX_FEATHER_NORMALIZE_3,
	POINT_2D_FEATHERX_3,
	POINT_2D_FEATHERY_3,
	POINT_2D_FEATHERZ_3,
	SLIDER_FALLOFF_3,
	SLIDER_INTENSITY_3, //---
	SLIDER_SATURATION_3,
	COLOR_NEAR_3,
	CHECKBOX_COLOR_FAR_3,
	COLOR_FAR_3,
	SLIDER_COLOR_FAR_FALLOFF_3,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_3,


	// Local Light Settings 4

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_4,
	CHECKBOX_LIGHT_TOGGLE_4,
	POINT_3D_POS1_4, //---
	POINT_3D_VX1_4,
	POINT_3D_VY1_4,
	POINT_3D_VZ1_4,
	POINT_3D_RES1_4,
	POINT_3D_SCALE1_4,
	POINT_3D_POS2_4, //---
	POINT_3D_VX2_4,
	POINT_3D_VY2_4,
	POINT_3D_VZ2_4,
	POINT_3D_RES2_4,
	POINT_3D_SCALE2_4,
	CHECKBOX_INVERT_4, //---
	CHECKBOX_FEATHER_NORMALIZE_4,
	POINT_2D_FEATHERX_4,
	POINT_2D_FEATHERY_4,
	POINT_2D_FEATHERZ_4,
	SLIDER_FALLOFF_4,
	SLIDER_INTENSITY_4, //---
	SLIDER_SATURATION_4,
	COLOR_NEAR_4,
	CHECKBOX_COLOR_FAR_4,
	COLOR_FAR_4,
	SLIDER_COLOR_FAR_FALLOFF_4,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_4,


	// Local Light Settings 5

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_5,
	CHECKBOX_LIGHT_TOGGLE_5,
	POINT_3D_POS1_5, //---
	POINT_3D_VX1_5,
	POINT_3D_VY1_5,
	POINT_3D_VZ1_5,
	POINT_3D_RES1_5,
	POINT_3D_SCALE1_5,
	POINT_3D_POS2_5, //---
	POINT_3D_VX2_5,
	POINT_3D_VY2_5,
	POINT_3D_VZ2_5,
	POINT_3D_RES2_5,
	POINT_3D_SCALE2_5,
	CHECKBOX_INVERT_5, //---
	CHECKBOX_FEATHER_NORMALIZE_5,
	POINT_2D_FEATHERX_5,
	POINT_2D_FEATHERY_5,
	POINT_2D_FEATHERZ_5,
	SLIDER_FALLOFF_5,
	SLIDER_INTENSITY_5, //---
	SLIDER_SATURATION_5,
	COLOR_NEAR_5,
	CHECKBOX_COLOR_FAR_5,
	COLOR_FAR_5,
	SLIDER_COLOR_FAR_FALLOFF_5,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_5,


	// Local Light Settings 6

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_6,
	CHECKBOX_LIGHT_TOGGLE_6,
	POINT_3D_POS1_6, //---
	POINT_3D_VX1_6,
	POINT_3D_VY1_6,
	POINT_3D_VZ1_6,
	POINT_3D_RES1_6,
	POINT_3D_SCALE1_6,
	POINT_3D_POS2_6, //---
	POINT_3D_VX2_6,
	POINT_3D_VY2_6,
	POINT_3D_VZ2_6,
	POINT_3D_RES2_6,
	POINT_3D_SCALE2_6,
	CHECKBOX_INVERT_6, //---
	CHECKBOX_FEATHER_NORMALIZE_6,
	POINT_2D_FEATHERX_6,
	POINT_2D_FEATHERY_6,
	POINT_2D_FEATHERZ_6,
	SLIDER_FALLOFF_6,
	SLIDER_INTENSITY_6, //---
	SLIDER_SATURATION_6,
	COLOR_NEAR_6,
	CHECKBOX_COLOR_FAR_6,
	COLOR_FAR_6,
	SLIDER_COLOR_FAR_FALLOFF_6,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_6,


	// Local Light Settings 7

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_7,
	CHECKBOX_LIGHT_TOGGLE_7,
	POINT_3D_POS1_7, //---
	POINT_3D_VX1_7,
	POINT_3D_VY1_7,
	POINT_3D_VZ1_7,
	POINT_3D_RES1_7,
	POINT_3D_SCALE1_7,
	POINT_3D_POS2_7, //---
	POINT_3D_VX2_7,
	POINT_3D_VY2_7,
	POINT_3D_VZ2_7,
	POINT_3D_RES2_7,
	POINT_3D_SCALE2_7,
	CHECKBOX_INVERT_7, //---
	CHECKBOX_FEATHER_NORMALIZE_7,
	POINT_2D_FEATHERX_7,
	POINT_2D_FEATHERY_7,
	POINT_2D_FEATHERZ_7,
	SLIDER_FALLOFF_7,
	SLIDER_INTENSITY_7, //---
	SLIDER_SATURATION_7,
	COLOR_NEAR_7,
	CHECKBOX_COLOR_FAR_7,
	COLOR_FAR_7,
	SLIDER_COLOR_FAR_FALLOFF_7,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_7,


	// Local Light Settings 8

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_8,
	CHECKBOX_LIGHT_TOGGLE_8,
	POINT_3D_POS1_8, //---
	POINT_3D_VX1_8,
	POINT_3D_VY1_8,
	POINT_3D_VZ1_8,
	POINT_3D_RES1_8,
	POINT_3D_SCALE1_8,
	POINT_3D_POS2_8, //---
	POINT_3D_VX2_8,
	POINT_3D_VY2_8,
	POINT_3D_VZ2_8,
	POINT_3D_RES2_8,
	POINT_3D_SCALE2_8,
	CHECKBOX_INVERT_8, //---
	CHECKBOX_FEATHER_NORMALIZE_8,
	POINT_2D_FEATHERX_8,
	POINT_2D_FEATHERY_8,
	POINT_2D_FEATHERZ_8,
	SLIDER_FALLOFF_8,
	SLIDER_INTENSITY_8, //---
	SLIDER_SATURATION_8,
	COLOR_NEAR_8,
	CHECKBOX_COLOR_FAR_8,
	COLOR_FAR_8,
	SLIDER_COLOR_FAR_FALLOFF_8,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_8,


	// Local Light Settings 9

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_9,
	CHECKBOX_LIGHT_TOGGLE_9,
	POINT_3D_POS1_9, //---
	POINT_3D_VX1_9,
	POINT_3D_VY1_9,
	POINT_3D_VZ1_9,
	POINT_3D_RES1_9,
	POINT_3D_SCALE1_9,
	POINT_3D_POS2_9, //---
	POINT_3D_VX2_9,
	POINT_3D_VY2_9,
	POINT_3D_VZ2_9,
	POINT_3D_RES2_9,
	POINT_3D_SCALE2_9,
	CHECKBOX_INVERT_9, //---
	CHECKBOX_FEATHER_NORMALIZE_9,
	POINT_2D_FEATHERX_9,
	POINT_2D_FEATHERY_9,
	POINT_2D_FEATHERZ_9,
	SLIDER_FALLOFF_9,
	SLIDER_INTENSITY_9, //---
	SLIDER_SATURATION_9,
	COLOR_NEAR_9,
	CHECKBOX_COLOR_FAR_9,
	COLOR_FAR_9,
	SLIDER_COLOR_FAR_FALLOFF_9,
	TOPIC_END_LOCAL_LIGHT_SETTINGS_9,


	// Local Light Settings 10

	TOPIC_ADD_LOCAL_LIGHT_SETTINGS_10,
	CHECKBOX_LIGHT_TOGGLE_10,
	POINT_3D_POS1_10, //---
	POINT_3D_VX1_10,
	POINT_3D_VY1_10,
	POINT_3D_VZ1_10,
	POINT_3D_RES1_10,
	POINT_3D_SCALE1_10,
	POINT_3D_POS2_10, //---
	POINT_3D_VX2_10,
	POINT_3D_VY2_10,
	POINT_3D_VZ2_10,
	POINT_3D_RES2_10,
	POINT_3D_SCALE2_10,
	CHECKBOX_INVERT_10, //---
	CHECKBOX_FEATHER_NORMALIZE_10,
	POINT_2D_FEATHERX_10,
	POINT_2D_FEATHERY_10,
	POINT_2D_FEATHERZ_10,
	SLIDER_FALLOFF_10,
	SLIDER_INTENSITY_10, //---
	SLIDER_SATURATION_10,
	COLOR_NEAR_10,
	CHECKBOX_COLOR_FAR_10,
	COLOR_FAR_10,
	SLIDER_COLOR_FAR_FALLOFF_10,
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
