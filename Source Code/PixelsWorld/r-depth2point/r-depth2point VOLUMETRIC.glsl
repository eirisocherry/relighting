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

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Unmult

float luminance(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec4 unmult(vec4 color) {

    float luma = luminance(color.xyz);

    float alpha = clamp(luma, 0.0, 1.0);

    vec4 screen = vec4(vec3(color.rgb / alpha), alpha * color.a);

    return screen;

}

// Clamp

vec4 clampValuesKeepAlpha(vec4 color) {

    float maxChannel = max(max(color.r, color.g), color.b);
    if (maxChannel > 1.0) {
        float realValue = maxChannel * color.a;

        if (realValue <= 1.0) {
            color.rgb = color.rgb / maxChannel;
            color.a = color.a * maxChannel;
        } else {
            color.rgb = color.rgb / maxChannel * realValue;
            color.a = color.a * maxChannel;
        }

    }

    color.rgb = clamp(color.rgb, 0.0, 1.0);

    return color;

}

// Main

bool intersectSphere(vec3 ro, vec3 rd, vec3 sphereCenter, float radius, out float t0, out float t1) {

    vec3 oc = ro - sphereCenter;
    float a = dot(rd, rd);
    float b = 2.0 * dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0) {
        return false;
    }

    float sqrtD = sqrt(discriminant);
    t0 = (-b - sqrtD) / (2.0 * a);
    t1 = (-b + sqrtD) / (2.0 * a);

    // t = t0 > 0.0 ? t0 : t1;

    // if (t < 0.0) {
    //     return false;
    // }

    return true;

}

vec3 volumetric(vec3 curPos, vec3 lightPos, float radius, float fallOff, float intensityMultiplier, float saturation, vec3 lightColorNear, bool lightColorFarToggle, vec3 lightColorFar, float colorFalloff) {

    vec3 ro = vec3(0.0);        // ray origin
    vec3 rd = normalize(curPos);    // ray direction
    float t0;                          // distance between "ro" and "intersect point"
    float t1;
    if (!intersectSphere(ro, rd, lightPos, radius, t0, t1)) {
        return vec3(0.0);
    }

    float tMin = min(t0, t1); // near distance between "ro" and "intersect point"
    float tMax = max(t0, t1); // far distance between "ro" and "intersect point"

    if (tMax < 0.0) return vec3(0.0); // if sphere behind the camera

    float enter = tMin < 0.0 ? 0.0 : tMin;  // if true - camera inside the sphere
    float exit = tMax;

    float distanceInside = exit - enter;

    if (distanceInside <= 0.0) return vec3(0.0);

    //Collision
    float distToSphere = tMin;
    float distToObject = length(curPos - ro);
    if (distToSphere > distToObject) {
        return vec3(0.0);
    }

    // Falloff
    vec3 intersectPoint = ro + enter * rd;     // Найти точку входа в сферу
    vec3 toCenter = lightPos - intersectPoint;
    float tClosest = dot(toCenter, rd); // Спроецировать этот вектор на направление луча
    vec3 closestToCenter = intersectPoint + tClosest * rd; // Точка внутри сферы, которая ближе всего к центру
    float distanceToCenter = length(closestToCenter - lightPos);     // Расстояние от центра до этой точки

    float normalizedDist = clamp(distanceToCenter / radius, 0.0, 1.0);
    float intensity = 1.0 - normalizedDist;
    return vec3(intensity);
    // float intensity = pow(1.0 - normalizedDist, fallOff);


    // float intensity = 1.0;
    // float colorIntensity = 1.0;
    // if (fallOff > 0.0) {
    //     intensity = pow(1.0 - normalizedDist, fallOff);
    // }
    // if (colorFalloff > 0.0) {
    //     colorIntensity = pow(1.0 - normalizedDist, colorFalloff);
    // }


    // // Mix Color
    // vec3 color = vec3(0.0);
    // if (lightColorFarToggle) {
    //     color = mix(lightColorFar, lightColorNear, colorIntensity);
    // } else {
    //     color = lightColorNear;
    // }

    // // Saturation
    // vec3 colorHSV = rgb2hsv(color);
    // vec3 colorRGB = hsv2rgb(vec3(colorHSV.x, clamp(colorHSV.y * saturation, 0.0, 1.0), colorHSV.z));
    // vec3 colorFinal = colorRGB * intensity * intensityMultiplier;

    // return colorFinal;

}

    // vec3 intersectPoint = ro + t * rd;

    // float distToSphere = length(intersectPoint - ro);
    // float distToObject = length(curPos - ro);

    // if (distToSphere > distToObject) {
    //     return vec3(0.0);
    // }

    // return vec3(1.0);



    // falloff
    // float distToCenter = length(intersectPoint - lightPos);

    // float tFalloff = clamp(distToCenter / radius, 0.0, 1.0);

    // float intensity = 1.0;
    // if (fallOff > 0.0) {
    //     intensity = pow(1.0 - tFalloff, fallOff);
    // }

    // return vec3(intensity);

void mainImage(out vec4 outColor, in vec2 fragCoord) {

    // Angles

    // Points
    vec2 lightParameters2D1 = readPoint(0); // saturation / color falloff
    vec2 lightParameters2D2 = readPoint(1); 
    vec2 lightParameters2D3 = readPoint(2); 
    vec2 lightParameters2D4 = readPoint(3); 

    // 3D Points
    vec3 lightPos1 = read3dPoint(0);
    vec3 lightPos2 = read3dPoint(1);
    vec3 lightPos3 = read3dPoint(2);
    vec3 lightPos4 = read3dPoint(3);
    vec3 lightParameters3D1 = read3dPoint(4); // radius / falloff / intensity
    vec3 lightParameters3D2 = read3dPoint(5);
    vec3 lightParameters3D3 = read3dPoint(6);
    vec3 lightParameters3D4 = read3dPoint(7);

    // Sliders
    depthFar = _PixelsWorld_slider[0];
    float radiusMultiplier = _PixelsWorld_slider[1];
    float falloffMultiplier = _PixelsWorld_slider[2];
    float intensityMultiplier = _PixelsWorld_slider[3];
    float saturationMultiplier = _PixelsWorld_slider[4];
    float colorFalloffMultiplier = _PixelsWorld_slider[5];
    float clampValuesSlider = _PixelsWorld_slider[6];
    float saturation1 = lightParameters2D1.x * saturationMultiplier;
    float saturation2 = lightParameters2D2.x * saturationMultiplier;
    float saturation3 = lightParameters2D3.x * saturationMultiplier;
    float saturation4 = lightParameters2D4.x * saturationMultiplier;
    float colorFalloff1 = lightParameters2D1.y * colorFalloffMultiplier;
    float colorFalloff2 = lightParameters2D2.y * colorFalloffMultiplier;
    float colorFalloff3 = lightParameters2D3.y * colorFalloffMultiplier;
    float colorFalloff4 = lightParameters2D4.y * colorFalloffMultiplier;
    float radius1 = lightParameters3D1.x * radiusMultiplier;
    float radius2 = lightParameters3D2.x * radiusMultiplier;
    float radius3 = lightParameters3D3.x * radiusMultiplier;
    float radius4 = lightParameters3D4.x * radiusMultiplier;
    float falloff1 = lightParameters3D1.y * falloffMultiplier;
    float falloff2 = lightParameters3D2.y * falloffMultiplier;
    float falloff3 = lightParameters3D3.y * falloffMultiplier;
    float falloff4 = lightParameters3D4.y * falloffMultiplier;
    float intensity1 = lightParameters3D1.z * intensityMultiplier;
    float intensity2 = lightParameters3D2.z * intensityMultiplier;
    float intensity3 = lightParameters3D3.z * intensityMultiplier;
    float intensity4 = lightParameters3D4.z * intensityMultiplier;

    // Checkboxes
    depthBlackIsNear = _PixelsWorld_checkbox[0];
    bool lightToggle1 = _PixelsWorld_checkbox[1];
    bool lightToggle2 = _PixelsWorld_checkbox[2];
    bool lightToggle3 = _PixelsWorld_checkbox[3];
    bool lightToggle4 = _PixelsWorld_checkbox[4];
    bool lightColorFarToggle1 = _PixelsWorld_checkbox[5];
    bool lightColorFarToggle2 = _PixelsWorld_checkbox[6];
    bool lightColorFarToggle3 = _PixelsWorld_checkbox[7];
    bool lightColorFarToggle4 = _PixelsWorld_checkbox[8];
    bool removeBlackToggle = _PixelsWorld_checkbox[9];
    bool clampValuesToggle = false;
    if (clampValuesSlider > 0.5) {
        clampValuesToggle = true;
    }

    // Colors
    vec3 lightColorNear1 = _PixelsWorld_color[0].xyz;
    vec3 lightColorFar1 = _PixelsWorld_color[1].xyz;
    vec3 lightColorNear2 = _PixelsWorld_color[2].xyz;
    vec3 lightColorFar2 = _PixelsWorld_color[3].xyz;
    vec3 lightColorNear3 = _PixelsWorld_color[4].xyz;
    vec3 lightColorFar3 = _PixelsWorld_color[5].xyz;
    vec3 lightColorNear4 = _PixelsWorld_color[6].xyz;
    vec3 lightColorFar4 = _PixelsWorld_color[7].xyz;

    // Layers

    // Textures

    // Code

    vec2 uv = _PixelsWorld_uv;

    vec3 curPos = getPosition(uv, true);

    vec4 draw = vec4(0.0, 0.0, 0.0, 1.0);

    // if (lightToggle1) {
    //     draw.rgb += point(curPos, lightPos1, radius1, falloff1, intensity1, saturation1, lightColorNear1, lightColorFarToggle1, lightColorFar1, colorFalloff1);
    // }

    // if (lightToggle2) {
    //     draw.rgb += point(curPos, lightPos2, radius2, falloff2, intensity2, saturation2, lightColorNear2, lightColorFarToggle2, lightColorFar2, colorFalloff2);
    // }

    // if (lightToggle3) {
    //     draw.rgb += point(curPos, lightPos3, radius3, falloff3, intensity3, saturation3, lightColorNear3, lightColorFarToggle3, lightColorFar3, colorFalloff3);
    // }

    // if (lightToggle4) {
    //     draw.rgb += point(curPos, lightPos4, radius4, falloff4, intensity4, saturation4, lightColorNear4, lightColorFarToggle4, lightColorFar4, colorFalloff4);
    // }

    draw.rgb += volumetric(curPos, lightPos1, radius1, falloff1, intensity1, saturation1, lightColorNear1, lightColorFarToggle1, lightColorFar1, colorFalloff1);

    if (removeBlackToggle) {
        draw = unmult(draw);
    }

    if (clampValuesToggle) {
        draw = clampValuesKeepAlpha(draw);
    }

    

    

    outColor = draw;

}