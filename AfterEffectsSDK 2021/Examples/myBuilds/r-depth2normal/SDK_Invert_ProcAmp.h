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

#define NAME			"r-depth2normal"
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

	// Normal

	TOPIC_ADD_NORMAL,
	CHECKBOX_NORMAL_IMPROVE,
	CHECKBOX_NORMAL_NORMALIZE,
	TOPIC_END_NORMAL,


	// Smooth Normal

	TOPIC_ADD_SMOOTH_NORMAL,
	CHECKBOX_SMOOTH,
	SLIDER_SMOOTH_RADIUS,
	SLIDER_SMOOTH_THRESHOLD,
	SLIDER_SMOOTH_DEPTH_WEIGHT,
	TOPIC_END_SMOOTH_NORMAL,

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
