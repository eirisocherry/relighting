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

	float featherMultiplier;
	float falloffMultiplier;
	float intensityMultiplier;
	float saturationMultiplier;
	float colorFalloffMultiplier;

	// Local Light Settings

	// Light Start
	bool lightToggle[10];
	float posX1[10]; float posY1[10]; float posZ1[10];
	float vXx1[10]; float vXy1[10]; float vXz1[10];
	float vYx1[10]; float vYy1[10]; float vYz1[10];
	float vZx1[10]; float vZy1[10]; float vZz1[10];
	float resX1[10]; float resY1[10]; float resZ1[10];
	float scaleX1[10]; float scaleY1[10]; float scaleZ1[10];

	// Light End
	float posX2[10]; float posY2[10]; float posZ2[10];
	float vXx2[10]; float vXy2[10]; float vXz2[10];
	float vYx2[10]; float vYy2[10]; float vYz2[10];
	float vZx2[10]; float vZy2[10]; float vZz2[10];
	float resX2[10]; float resY2[10]; float resZ2[10];
	float scaleX2[10]; float scaleY2[10]; float scaleZ2[10];

	// Shape
	bool featherNormalize[10];
	float featherX1[10]; float featherX2[10];
	float featherY1[10]; float featherY2[10];
	float featherZ1[10]; float featherZ2[10];
	float falloff[10];

	// Ambient
	bool ambientToggle[10];
	float ambientIntensity[10];
	float ambientSaturation[10];
	float ambientColorNearR[10]; float ambientColorNearG[10]; float ambientColorNearB[10];
	bool ambientColorFarToggle[10];
	float ambientColorFarR[10]; float ambientColorFarG[10]; float ambientColorFarB[10];
	float ambientColorFalloff[10];

	// Diffuse
	bool diffuseToggle[10];
	float diffuseIntensity[10];
	float diffuseSaturation[10];
	float diffuseColorNearR[10]; float diffuseColorNearG[10]; float diffuseColorNearB[10];
	bool diffuseColorFarToggle[10];
	float diffuseColorFarR[10]; float diffuseColorFarG[10]; float diffuseColorFarB[10];
	float diffuseColorFalloff[10];

	// Specular
	bool specularToggle[10];
	float specularSize[10];
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
