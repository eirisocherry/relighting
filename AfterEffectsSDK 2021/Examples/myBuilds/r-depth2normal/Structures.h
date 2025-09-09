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

	// Normal

	bool improve;
	bool normalize;

	// Smooth Normal

	bool smooth;
	int radius;
	float normalThreshold;
	float depthWeight;

	// Layers

	bool layerNormalExist;

} InvertProcAmpParams;

#endif // STRUCTURES_H
