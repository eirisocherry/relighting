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

#define NAME			"r-rim"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	RELIGHTING_INPUT=0,

	// Light Layers

	TOPIC_ADD_LIGHT_LAYERS,
	LAYER_LIGHT_1,
	LAYER_LIGHT_LOOK_AT_1,
	LAYER_LIGHT_2,
	LAYER_LIGHT_LOOK_AT_2,
	LAYER_LIGHT_3,
	LAYER_LIGHT_LOOK_AT_3,
	LAYER_LIGHT_4,
	LAYER_LIGHT_LOOK_AT_4,
	LAYER_LIGHT_5,
	LAYER_LIGHT_LOOK_AT_5,
	TOPIC_END_LIGHT_LAYERS,

	// Global Light Settings

	TOPIC_ADD_GLOBAL_SETTINGS,
	SLIDER_INTENSITY_MULTIPLIER,
	SLIDER_SATURATION_MULTIPLIER,
	TOPIC_END_GLOBAL_SETTINGS,
	
	// Rim Light 1

	TOPIC_ADD_RIM_SETTINGS_1,
	CHECKBOX_RIM_TOGGLE_1,
	POINT_3D_LIGHT_POSITION_1,
	POINT_3D_LIGHT_LOOK_AT_POSITION_1,
	SLIDER_RIM_START_1,
	SLIDER_RIM_END_1,
	SLIDER_RIM_INTENSITY_1,
	SLIDER_RIM_SATURATION_1,
	COLOR_RIM_COLOR_NEAR_1,
	CHECKBOX_RIM_COLOR_FAR_TOGGLE_1,
	COLOR_RIM_COLOR_FAR_1,
	SLIDER_RIM_COLOR_FALLOFF_1,
	TOPIC_END_RIM_SETTINGS_1,


	// Rim Light 2

	TOPIC_ADD_RIM_SETTINGS_2,
	CHECKBOX_RIM_TOGGLE_2,
	POINT_3D_LIGHT_POSITION_2,
	POINT_3D_LIGHT_LOOK_AT_POSITION_2,
	SLIDER_RIM_START_2,
	SLIDER_RIM_END_2,
	SLIDER_RIM_INTENSITY_2,
	SLIDER_RIM_SATURATION_2,
	COLOR_RIM_COLOR_NEAR_2,
	CHECKBOX_RIM_COLOR_FAR_TOGGLE_2,
	COLOR_RIM_COLOR_FAR_2,
	SLIDER_RIM_COLOR_FALLOFF_2,
	TOPIC_END_RIM_SETTINGS_2,


	// Rim Light 3

	TOPIC_ADD_RIM_SETTINGS_3,
	CHECKBOX_RIM_TOGGLE_3,
	POINT_3D_LIGHT_POSITION_3,
	POINT_3D_LIGHT_LOOK_AT_POSITION_3,
	SLIDER_RIM_START_3,
	SLIDER_RIM_END_3,
	SLIDER_RIM_INTENSITY_3,
	SLIDER_RIM_SATURATION_3,
	COLOR_RIM_COLOR_NEAR_3,
	CHECKBOX_RIM_COLOR_FAR_TOGGLE_3,
	COLOR_RIM_COLOR_FAR_3,
	SLIDER_RIM_COLOR_FALLOFF_3,
	TOPIC_END_RIM_SETTINGS_3,


	// Rim Light 4

	TOPIC_ADD_RIM_SETTINGS_4,
	CHECKBOX_RIM_TOGGLE_4,
	POINT_3D_LIGHT_POSITION_4,
	POINT_3D_LIGHT_LOOK_AT_POSITION_4,
	SLIDER_RIM_START_4,
	SLIDER_RIM_END_4,
	SLIDER_RIM_INTENSITY_4,
	SLIDER_RIM_SATURATION_4,
	COLOR_RIM_COLOR_NEAR_4,
	CHECKBOX_RIM_COLOR_FAR_TOGGLE_4,
	COLOR_RIM_COLOR_FAR_4,
	SLIDER_RIM_COLOR_FALLOFF_4,
	TOPIC_END_RIM_SETTINGS_4,


	// Rim Light 5

	TOPIC_ADD_RIM_SETTINGS_5,
	CHECKBOX_RIM_TOGGLE_5,
	POINT_3D_LIGHT_POSITION_5,
	POINT_3D_LIGHT_LOOK_AT_POSITION_5,
	SLIDER_RIM_START_5,
	SLIDER_RIM_END_5,
	SLIDER_RIM_INTENSITY_5,
	SLIDER_RIM_SATURATION_5,
	COLOR_RIM_COLOR_NEAR_5,
	CHECKBOX_RIM_COLOR_FAR_TOGGLE_5,
	COLOR_RIM_COLOR_FAR_5,
	SLIDER_RIM_COLOR_FALLOFF_5,
	TOPIC_END_RIM_SETTINGS_5,

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
