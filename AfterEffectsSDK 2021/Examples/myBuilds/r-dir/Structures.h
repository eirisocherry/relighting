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

	float intensityMultiplier;
	float saturationMultiplier;

	// Local Light Settings

	// Light Start
	bool lightToggle[5];
	float posX1[5]; float posY1[5]; float posZ1[5];
	float posX2[5]; float posY2[5]; float posZ2[5];

	// Ambient
	bool ambientToggle[5];
	float ambientIntensity[5];
	float ambientSaturation[5];
	float ambientColorR[5]; float ambientColorG[5]; float ambientColorB[5];

	// Diffuse
	bool diffuseToggle[5];
	float diffuseIntensity[5];
	float diffuseSaturation[5];
	float diffuseColorR[5]; float diffuseColorG[5]; float diffuseColorB[5];


	// Specular
	bool specularToggle[5];
	float specularSize[5];
	float specularIntensity[5];
	float specularSaturation[5];
	float specularColorR[5]; float specularColorG[5]; float specularColorB[5];

	// Shadows
	bool shadowToggle[5];
	bool shadowIgnoreAmbientToggle[5];
	bool shadowIgnoreDiffuseToggle[5];
	bool shadowIgnoreSpecularToggle[5];
	float shadowSampleStep[5];
	float shadowImprovedSampleRadius[5];
	float shadowMaxLength[5];
	float shadowThresholdStart[5];
	float shadowThresholdEnd[5];
	float shadowSoftnessRadius[5];
	int shadowSoftnessSamples[5];
	float shadowIntensity[5];
	float shadowColorR[5]; float shadowColorG[5]; float shadowColorB[5];

} InvertProcAmpParams;

#endif // STRUCTURES_H
