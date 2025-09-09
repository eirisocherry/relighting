#pragma once
#ifndef STRUCTURES_H
#define STRUCTURES_H

typedef struct
{

	// Camera

	float camVx1; float camVx2; float camVx3; float camVx4;
	float camVy1; float camVy2; float camVy3; float camVy4;
	float camVz1; float camVz2; float camVz3; float camVz4;
	float camPos1; float camPos2; float camPos3; float camPos4;
	float cameraZoom; float cameraWidth; float cameraHeight;

	// Debug

	int renderMode;
	// 1: All
	// 2: Ambient
	// 3: Specular
	// 4: Diffuse
	// 5: Shadows

	// Depth Settings

	float depthFar;
	bool depthBlackIsNear;

	// Normal Settings
	bool normalExistToggle;
	
	// Global Light Settings

	float radiusMultiplier;
	float intensityMultiplier;
	float saturationMultiplier;

	// Local Light Settings

	// Main
	float lightToggle[10];
	float lightPosX[10]; float lightPosY[10]; float lightPosZ[10];
	float radius[10];
	
	// Ambient
	bool ambientToggle[10];
	float ambientFalloff[10];
	float ambientIntensity[10];
	float ambientSaturation[10];
	float ambientColorNearR[10]; float ambientColorNearG[10]; float ambientColorNearB[10];
	bool ambientColorFarToggle[10];
	float ambientColorFarR[10]; float ambientColorFarG[10]; float ambientColorFarB[10];
	float ambientColorFalloff[10];

	// Diffuse
	bool diffuseToggle[10];
	float diffuseFalloff[10];
	float diffuseIntensity[10];
	float diffuseSaturation[10];
	float diffuseColorNearR[10]; float diffuseColorNearG[10]; float diffuseColorNearB[10];
	bool diffuseColorFarToggle[10];
	float diffuseColorFarR[10]; float diffuseColorFarG[10]; float diffuseColorFarB[10];
	float diffuseColorFalloff[10];

	// Specular
	bool specularToggle[10];
	float specularSize[10];
	float specularFalloff[10];
	float specularIntensity[10];
	float specularSaturation[10];
	float specularColorNearR[10]; float specularColorNearG[10]; float specularColorNearB[10];
	bool specularColorFarToggle[10];
	float specularColorFarR[10]; float specularColorFarG[10]; float specularColorFarB[10];
	float specularColorFalloff[10];

	// Shadows
	bool shadowToggle[10];
	bool shadowIgnoreAmbientToggle[10];
	bool shadowIgnoreDiffuseToggle[10];
	bool shadowIgnoreSpecularToggle[10];
	bool shadowClipToLightToggle[10];
	float shadowSampleStep[10];
	float shadowImprovedSampleRadius[10];
	float shadowMaxLength[10];
	float shadowThresholdStart[10];
	float shadowThresholdEnd[10];
	float shadowSoftnessRadius[10];
	int shadowSoftnessSamples[10];
	float shadowIntensity[10];
	float shadowColorR[10]; float shadowColorG[10]; float shadowColorB[10];

} InvertProcAmpParams;

#endif // STRUCTURES_H
