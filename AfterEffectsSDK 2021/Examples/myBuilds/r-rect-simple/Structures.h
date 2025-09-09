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

	float featherMultiplier;
	float falloffMultiplier;
	float intensityMultiplier;
	float saturationMultiplier;
	float colorFalloffMultiplier;

	// Light Settings

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
	bool invertToggle[10];
	bool featherNormalize[10];
	float featherX1[10]; float featherX2[10];
	float featherY1[10]; float featherY2[10];
	float featherZ1[10]; float featherZ2[10];
	float falloff[10];
	// Color
	float intensity[10];
	float saturation[10];
	float colorNearR[10]; float colorNearG[10]; float colorNearB[10];
	bool colorFarToggle[10];
	float colorFarR[10]; float colorFarG[10]; float colorFarB[10];
	float colorFalloff[10];

} InvertProcAmpParams;

#endif // STRUCTURES_H
