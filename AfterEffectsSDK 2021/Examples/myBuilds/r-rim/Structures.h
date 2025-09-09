#pragma once
#ifndef STRUCTURES_H
#define STRUCTURES_H

typedef struct
{

	// Global Settings

	float intensityMultiplier;
	float saturationMultiplier;

	// Rim Light

	bool rimToggle[5];
	float rimLightPositionX[5]; float rimLightPositionY[5]; float rimLightPositionZ[5];
	float rimLightLookAtPositionX[5]; float rimLightLookAtPositionY[5]; float rimLightLookAtPositionZ[5];
	float rimStart[5];
	float rimEnd[5];
	float rimIntensity[5];
	float rimSaturation[5];
	float rimColorNearR[5]; float rimColorNearG[5]; float rimColorNearB[5];
	bool rimColorFarToggle[5];
	float rimColorFarR[5]; float rimColorFarG[5]; float rimColorFarB[5];
	float rimColorFalloff[5];


} InvertProcAmpParams;

#endif // STRUCTURES_H
