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

	// Depth Settings

	float depthFar;
	bool depthBlackIsNear;

	// Global Light Settings

	float radiusMultiplier;
	float falloffMultiplier;
	float intensityMultiplier;
	float saturationMultiplier;
	float colorFalloffMultiplier;

	// Local Light Settings

	bool lightToggle[10];
	float lightPosX[10]; float lightPosY[10]; float lightPosZ[10];
	bool invertToggle[10];
	float radius[10];
	float falloff[10];
	float intensity[10];
	float saturation[10];
	float colorNearR[10]; float colorNearG[10]; float colorNearB[10];
	bool colorFarToggle[10];
	float colorFarR[10]; float colorFarG[10]; float colorFarB[10];
	float colorFalloff[10];

} InvertProcAmpParams;

#endif // STRUCTURES_H
