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

	float lengthMultiplier;
	float angleXmultiplier;
	float angleYmultiplier;
	float curvatureMultiplier;
	float featherMultiplier;
	float falloffMultiplier;
	float intensityMultiplier;
	float saturationMultiplier;
	float colorFalloffMultiplier;

	// Local Light Settings

	bool lightToggle[10];
	float lightPosX[10]; float lightPosY[10]; float lightPosZ[10];
	float lightVxX[10]; float lightVxY[10]; float lightVxZ[10];
	float lightVyX[10]; float lightVyY[10]; float lightVyZ[10];
	float lightVzX[10]; float lightVzY[10]; float lightVzZ[10];
	//shape
	bool invertToggle[10];
	float length[10];
	float angleX[10];
	float angleY[10];
	float curvature[10];
	float feather[10];
	float falloff[10];
	//ies
	int ies[10];
	float iesBrightness1[10]; float iesDistance1[10];
	float iesBrightness2[10]; float iesDistance2[10];
	float iesBrightness3[10]; float iesDistance3[10];
	float iesBrightness4[10]; float iesDistance4[10];
	float iesBrightness5[10]; float iesDistance5[10];
	float iesBrightness6[10]; float iesDistance6[10];
	//color
	float intensity[10];
	float saturation[10];
	float colorNearR[10]; float colorNearG[10]; float colorNearB[10];
	bool colorFarToggle[10];
	float colorFarR[10]; float colorFarG[10]; float colorFarB[10];
	float colorFalloff[10];

} InvertProcAmpParams;

#endif // STRUCTURES_H
