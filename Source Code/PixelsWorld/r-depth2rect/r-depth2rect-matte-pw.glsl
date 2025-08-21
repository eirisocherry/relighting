float depthFar = 0.0;
bool depthBlackIsNear = false;

// Get

float getDepth(vec2 uv) {

    float depth = texture(_PixelsWorld_inLayer, uv).x;

    if (depthBlackIsNear) {
        depth = depth * depthFar;
    } else {
        depth = (1.0 - depth) * depthFar;
    }
    
    return depth;
    
}

vec3 getPosition(vec2 uv, bool screenSpace) {

    vec2 fragCoord = uv * iResolution.xy;
    vec3 screenPos = vec3(
        fragCoord.x - 0.5 * iResolution.x,          // [-halfRes..halfRes]
        (fragCoord.y - 0.5 * iResolution.y) * -1,   // invert y -> [halfRes..-halfRes]
        _PixelsWorld_camera_info.z                  // focal length (zoom)
    );

    vec3 localPos = vec3(0.0);
    localPos.z = getDepth(uv);
    float diff = localPos.z / screenPos.z;
    localPos.xy = screenPos.xy * diff;

    if (screenSpace) {
        return localPos;
    } else {
        vec4 worldPos = _PixelsWorld_camera_matrix * vec4(localPos, 1.0); // Multiplication order is important (Matrix x Vec4)
        return worldPos.xyz;
    }
    
}

// Read

vec3 read3dPoint(int n) {

    return vec3(
        _PixelsWorld_point3d[n].x * iResolution.x,
        (1 - _PixelsWorld_point3d[n].y) * iResolution.y,
        _PixelsWorld_point3d[n].z
    );
    
}

vec2 readPoint(int n) {

    return vec2(
        _PixelsWorld_point[n].x * iResolution.x,
        (1 - _PixelsWorld_point[n].y) * iResolution.y
    );
    
}

// Color

vec3 rgb2hsv(vec3 c) {

    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);

}

vec3 hsv2rgb(vec3 c) {

    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);

}

// Main

vec3 rect(vec3 curPos, vec3 pos1, vec3 vX1, vec3 vY1, vec3 vZ1, vec3 res1, vec3 scale1, vec3 pos2, vec3 vX2, vec3 vY2, vec3 vZ2, vec3 res2, vec3 scale2, bool invertToggle, vec2 featherX, vec2 featherY, vec2 featherZ, bool featherNormalized, float falloff, float intensityMultiplier, float saturation, vec3 lightColorNear, bool lightColorFarToggle, vec3 lightColorFar, float colorFalloff) {
    
    // Sizes
    float sizeX1 = res1.x * abs(scale1.x);
    float sizeY1 = res1.y * abs(scale1.y);
    float sizeX2 = res2.x * abs(scale2.x);
    float sizeY2 = res2.y * abs(scale2.y);

    // Vertices A
    vec3 TL1 = pos1 + vX1 * sizeX1;
    vec3 TR1 = pos1;
    vec3 BR1 = pos1 - vY1 * sizeY1;
    vec3 BL1 = pos1 + vX1 * sizeX1 - vY1 * sizeY1;


    // Vertices B Temp
    vec3 TL2_Temp = pos2 + vX2 * sizeX2;
    vec3 TR2_Temp = pos2;
    vec3 BR2_Temp = pos2 - vY2 * sizeY2;
    vec3 BL2_Temp = pos2 + vX2 * sizeX2 - vY2 * sizeY2;


    // ----- Match vertices -----
    // Pattern that has the smallest total distance is the right one

    vec3 rect1_vertices[4] = vec3[4](TL1, TR1, BR1, BL1);

    // Predefine 8 possible patterns
    vec3 config_patterns[32] = vec3[32](
        // config 0: BL2, TL2, TR2, BR2
        BL2_Temp, TL2_Temp, TR2_Temp, BR2_Temp,
        // config 1: BL2, BR2, TR2, TL2
        BL2_Temp, BR2_Temp, TR2_Temp, TL2_Temp,
        // config 2: BR2, BL2, TL2, TR2
        BR2_Temp, BL2_Temp, TL2_Temp, TR2_Temp,
        // config 3: BR2, TR2, TL2, BL2
        BR2_Temp, TR2_Temp, TL2_Temp, BL2_Temp,
        // config 4: TL2, BL2, BR2, TR2
        TL2_Temp, BL2_Temp, BR2_Temp, TR2_Temp,
        // config 5: TL2, TR2, BR2, BL2
        TL2_Temp, TR2_Temp, BR2_Temp, BL2_Temp,
        // config 6: TR2, TL2, BL2, BR2
        TR2_Temp, TL2_Temp, BL2_Temp, BR2_Temp,
        // config 7: TR2, BR2, BL2, TL2
        TR2_Temp, BR2_Temp, BL2_Temp, TL2_Temp
    );

    float min_distance = 1e6;
    int best_config = 0;

    // Check what pattern has the smallest total distance
    for (int config = 0; config < 8; config++) {
        float total_dist = 0.0;
        for (int j = 0; j < 4; j++) {
            vec3 target_vertex = config_patterns[config * 4 + j];  // j-th vertex in config
            total_dist += distance(rect1_vertices[j], target_vertex);
        }

        if (total_dist < min_distance) {
            min_distance = total_dist;
            best_config = config;
        }
    }

    // Apply the correct pattern
    vec3 TL2 = config_patterns[best_config * 4 + 0];
    vec3 TR2 = config_patterns[best_config * 4 + 1];
    vec3 BR2 = config_patterns[best_config * 4 + 2];
    vec3 BL2 = config_patterns[best_config * 4 + 3];



    // ----- Rect Light Projection -----

    // 6 faces x 3 vertices
    vec3 faceVertices[18];

    // Face 0: Bottom (pos1) — BL1, BR1, TL1
    faceVertices[0] = BL1;
    faceVertices[1] = BR1;
    faceVertices[2] = TL1;

    // Face 1: Top (pos2) — BL2, TL2, BR2
    faceVertices[3] = BL2;
    faceVertices[4] = TL2;
    faceVertices[5] = BR2;

    // Face 2: Front — BL1, BR1, BR2
    faceVertices[6] = BL1;
    faceVertices[7] = BR1;
    faceVertices[8] = BR2;

    // Face 3: Right — BR1, TR1, TR2
    faceVertices[9]  = BR1;
    faceVertices[10] = TR1;
    faceVertices[11] = TR2;

    // Face 4: Back — TR1, TL1, TL2
    faceVertices[12] = TR1;
    faceVertices[13] = TL1;
    faceVertices[14] = TL2;

    // Face 5: Left — TL1, BL1, BL2
    faceVertices[15] = TL1;
    faceVertices[16] = BL1;
    faceVertices[17] = BL2;

    vec3 center = (BL1 + BR1 + TL1 + TR1 + BL2 + BR2 + TL2 + TR2) * 0.125;
    
    float distanceToLeftFace = 1e6;   // Face 5 (Left) - X
    float distanceToRightFace = 1e6;  // Face 3 (Right) - X
    float distanceToFrontFace = 1e6;  // Face 2 (Front) - Y
    float distanceToBackFace = 1e6;   // Face 4 (Back) - Y
    float distanceToBottomFace = 1e6; // Face 0 (Bottom) - Z
    float distanceToTopFace = 1e6;    // Face 1 (Top) - Z
    
    bool outsideLeft = false, outsideRight = false;
    bool outsideFront = false, outsideBack = false;
    bool outsideBottom = false, outsideTop = false;

    for (int i = 0; i < 6; i++) {
        vec3 v1 = faceVertices[i * 3 + 0];
        vec3 v2 = faceVertices[i * 3 + 1];
        vec3 v3 = faceVertices[i * 3 + 2];

        vec3 edge1 = v2 - v1;
        vec3 edge2 = v3 - v1;
        vec3 normal = normalize(cross(edge1, edge2));

        vec3 toPoint = curPos - v1;
        vec3 toCenter = center - v1;

        float distanceToPlane = dot(normal, toPoint);
        float sideCenter = dot(normal, toCenter);

        bool outsideThisFace = (distanceToPlane * sideCenter < 0.0);

        if (i == 0) { // Bottom face
            distanceToBottomFace = abs(distanceToPlane);
            outsideBottom = outsideThisFace;
        } else if (i == 1) { // Top face
            distanceToTopFace = abs(distanceToPlane);
            outsideTop = outsideThisFace;
        } else if (i == 2) { // Front face
            distanceToFrontFace = abs(distanceToPlane);
            outsideFront = outsideThisFace;
        } else if (i == 3) { // Right face 
            distanceToRightFace = abs(distanceToPlane);
            outsideRight = outsideThisFace;
        } else if (i == 4) { // Back face 
            distanceToBackFace = abs(distanceToPlane);
            outsideBack = outsideThisFace;
        } else if (i == 5) { // Left face 
            distanceToLeftFace = abs(distanceToPlane);
            outsideLeft = outsideThisFace;
        }
    }

    // Parameters for feather normalization
    float halfWidthX = (sizeX1 + sizeX2) * 0.5;
    float halfWidthY = (sizeY1 + sizeY2) * 0.5;
    float halfWidthZ = length(pos2 - pos1) * 0.5;
    
    float featherValue = 1.0;

    // Feather X Right
    if (outsideLeft) {
        featherValue = 0.0;
    } else {
        if (featherNormalized) {
            float normalizedDistance = distanceToLeftFace / halfWidthX;
            featherValue *= (featherX.x > 0.0) ? smoothstep(0.0, featherX.x, normalizedDistance) : 1.0;
        } else {
            featherValue *= (featherX.x > 0.0) ? smoothstep(0.0, featherX.x, distanceToLeftFace) : 1.0;
        }
    }

    // Feather X Left
    if (outsideRight) {
        featherValue = 0.0;
    } else {
        if (featherNormalized) {
            float normalizedDistance = distanceToRightFace / halfWidthX;
            featherValue *= (featherX.y > 0.0) ? smoothstep(0.0, featherX.y, normalizedDistance) : 1.0;
        } else {
            featherValue *= (featherX.y > 0.0) ? smoothstep(0.0, featherX.y, distanceToRightFace) : 1.0;
        }
    }

    // Feather Y Down
    if (outsideFront) {
        featherValue = 0.0;
    } else {
        if (featherNormalized) {
            float normalizedDistance = distanceToFrontFace / halfWidthY;
            featherValue *= (featherY.x > 0.0) ? smoothstep(0.0, featherY.x, normalizedDistance) : 1.0;
        } else {
            featherValue *= (featherY.x > 0.0) ? smoothstep(0.0, featherY.x, distanceToFrontFace) : 1.0;
        }
    }

    // Feather Y Up
    if (outsideBack) {
        featherValue = 0.0;
    } else {
        if (featherNormalized) {
            float normalizedDistance = distanceToBackFace / halfWidthY;
            featherValue *= (featherY.y > 0.0) ? smoothstep(0.0, featherY.y, normalizedDistance) : 1.0;
        } else {
            featherValue *= (featherY.y > 0.0) ? smoothstep(0.0, featherY.y, distanceToBackFace) : 1.0;
        }
    }

    // Feather Z Near
    if (outsideBottom) {
        featherValue = 0.0;
    } else {
        if (featherNormalized) {
            float normalizedDistance = distanceToBottomFace / halfWidthZ;
            featherValue *= (featherZ.x > 0.0) ? smoothstep(0.0, featherZ.x, normalizedDistance) : 1.0;
        } else {
            featherValue *= (featherZ.x > 0.0) ? smoothstep(0.0, featherZ.x, distanceToBottomFace) : 1.0;
        }
    }

    // Feather Z Far
    if (outsideTop) {
        featherValue = 0.0;
    } else {
        if (featherNormalized) {
            float normalizedDistance = distanceToTopFace / halfWidthZ;
            featherValue *= (featherZ.y > 0.0) ? smoothstep(0.0, featherZ.y, normalizedDistance) : 1.0;
        } else {
            featherValue *= (featherZ.y > 0.0) ? smoothstep(0.0, featherZ.y, distanceToTopFace) : 1.0;
        }
    }

    // Falloff Z
    float normalizedDistance = distanceToTopFace / (halfWidthZ * 2.0);
    float t = clamp(normalizedDistance, 0.0, 1.0);

    float intensity = featherValue;
    if (falloff > 0.0) {
        intensity *= pow( t, falloff);
    }

    float colorIntensity = 1.0;
    if (colorFalloff > 0.0) {
        colorIntensity *= pow( t, colorFalloff);
    }

    // Invert
    if (invertToggle) {
        intensity = 1.0 - intensity;
        colorIntensity = intensity;
    }

    // Mix Color
    vec3 color = vec3(0.0);
    if (lightColorFarToggle) {
        color = mix(lightColorFar, lightColorNear, colorIntensity);
    } else {
        color = lightColorNear;
    }

    // Saturation
    vec3 colorHSV = rgb2hsv(color);
    vec3 colorRGB = hsv2rgb(vec3(colorHSV.x, clamp(colorHSV.y * saturation, 0.0, 1.0), colorHSV.z));
    vec3 colorFinal = colorRGB * intensity * intensityMultiplier;

    return colorFinal;

}

void mainImage(out vec4 outColor, in vec2 fragCoord) {

    // Depth

    depthFar = _PixelsWorld_slider[0];
    depthBlackIsNear = _PixelsWorld_checkbox[0];

    // Light Start

    vec3 pos1 = read3dPoint(0);
    vec3 vX1 = read3dPoint(1);
    vec3 vY1 = read3dPoint(2);
    vec3 vZ1 = read3dPoint(3); // not used in the code
    vec3 res1 = read3dPoint(4);
    vec3 scale1 = read3dPoint(5);

    // Light End

    vec3 pos2 = read3dPoint(6);
    vec3 vX2 = read3dPoint(7);
    vec3 vY2 = read3dPoint(8);
    vec3 vZ2 = read3dPoint(9); // not used in the code
    vec3 res2 = _PixelsWorld_color[0].xyz;
    vec3 scale2 = _PixelsWorld_color[1].xyz;

    // Shape

    bool invertToggle = _PixelsWorld_checkbox[3];
    float falloff = _PixelsWorld_slider[1];
    float feather = _PixelsWorld_slider[2];
    vec2 featherX = readPoint(0);
    vec2 featherY = readPoint(1);
    vec2 featherZ = readPoint(2);
    bool featherUniformToggle = _PixelsWorld_checkbox[4];
    if (featherUniformToggle) {
        featherX = vec2(feather);
        featherY = vec2(feather);
        featherZ = vec2(feather);
    }
    bool featherNormalized = _PixelsWorld_checkbox[1];

    // Color

    float intensity = _PixelsWorld_slider[3];
    float saturation = _PixelsWorld_slider[4];
    vec3 colorNear = _PixelsWorld_color[2].xyz;
    bool colorFarToggle = _PixelsWorld_checkbox[2];
    vec3 colorFar = _PixelsWorld_color[3].xyz;
    float colorFalloff = _PixelsWorld_slider[5];

    // Code

    vec2 uv = _PixelsWorld_uv;

    vec3 curPos = getPosition(uv, false);

    vec3 draw = rect(curPos, pos1, vX1, vY1, vZ1, res1, scale1, pos2, vX2, vY2, vZ2, res2, scale2, invertToggle, featherX, featherY, featherZ, featherNormalized, falloff, intensity, saturation, colorNear, colorFarToggle, colorFar, colorFalloff);
    
    outColor = vec4(draw, 1.0);

}