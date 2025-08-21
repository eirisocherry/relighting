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

// Get IES Preset

vec4 getIESpreset(int preset, int index) {

    vec4 colorsDistances[6];

    // default preset
    colorsDistances[0] = vec4(1.0, 1.0, 1.0, 0.0);
    colorsDistances[1] = vec4(1.0, 1.0, 1.0, 0.2);
    colorsDistances[2] = vec4(1.0, 1.0, 1.0, 0.4);
    colorsDistances[3] = vec4(1.0, 1.0, 1.0, 0.6);
    colorsDistances[4] = vec4(1.0, 1.0, 1.0, 0.8);
    colorsDistances[5] = vec4(1.0, 1.0, 1.0, 1.0);

    if (preset == 2) {
        colorsDistances[0] = vec4(1.0, 1.0, 1.0, 0.0);
        colorsDistances[1] = vec4(0.0, 0.0, 0.0, 1.0);
        colorsDistances[2] = vec4(0.0, 0.0, 0.0, 1.0);
        colorsDistances[3] = vec4(0.0, 0.0, 0.0, 1.0);
        colorsDistances[4] = vec4(0.0, 0.0, 0.0, 1.0);
        colorsDistances[5] = vec4(0.0, 0.0, 0.0, 1.0);
    }

    if (preset == 3) {
        colorsDistances[0] = vec4(0.0, 0.0, 0.0, 0.0);
        colorsDistances[1] = vec4(1.0, 1.0, 1.0, 0.5);
        colorsDistances[2] = vec4(0.0, 0.0, 0.0, 1.0);
        colorsDistances[3] = vec4(0.0, 0.0, 0.0, 1.0);
        colorsDistances[4] = vec4(0.0, 0.0, 0.0, 1.0);
        colorsDistances[5] = vec4(0.0, 0.0, 0.0, 1.0);
    }

    if (preset == 4) {
        colorsDistances[0] = vec4(1.0, 1.0, 1.0, 0.1);
        colorsDistances[1] = vec4(0.4, 0.4, 0.4, 0.3);
        colorsDistances[2] = vec4(1.0, 1.0, 1.0, 0.4);
        colorsDistances[3] = vec4(0.3, 0.3, 0.3, 0.6);
        colorsDistances[4] = vec4(0.0, 0.0, 0.0, 0.9);
        colorsDistances[5] = vec4(0.0, 0.0, 0.0, 1.0);
    }

    return colorsDistances[index];

}

// Map 6 color gradient to 0-1 float 

vec3 ramp(vec4 colsDist[6], float x) {
    x = clamp(x, 0.0, 1.0);
    
    for(int i = 0; i < 5; i++) {
        if(x <= colsDist[i + 1].w) {
            float denom = colsDist[i + 1].w - colsDist[i].w;
            if(denom == 0.0) {
                return colsDist[i].xyz;
            }
            
            float t = (x - colsDist[i].w) / denom;
            return mix(colsDist[i].xyz, colsDist[i + 1].xyz, smoothstep(0.0, 1.0, t));
        }
    }
    
    // if (x > other distances), return last color 
    return colsDist[5].xyz;
}

// Main

vec3 spot(vec3 worldPos, vec3 lightPos, vec3 vX, vec3 vY, vec3 vZ, bool invertToggle, float z, vec2 angles, float curvature, float falloff, float feather, float intensityMultiplier, float saturation, vec3 lightColorNear, bool lightColorFarToggle, vec3 lightColorFar, float colorFalloff, vec4 colorsDistances[6]) {

    if (z == 0.0) return vec3(0.0);

    vec3 localPos = worldPos - lightPos;

    float curZ = dot(localPos, vZ);

    float intensity = 1.0;

    if (curZ < 0.0 || curZ > z) {
        if (invertToggle) {
            intensity = 0.0;
        } else {
            return vec3(0.0);
        }
    }

    float x = dot(localPos, vX);
    float y = dot(localPos, vY);

    float t = clamp(curZ / z, 0.0, 1.0);

    float endX = z * tan(radians(angles.x));
    float endY = z * tan(radians(angles.y));
    float curEndX = endX * pow( t, curvature);
    float curEndY = endY * pow( t, curvature);

    // Ellipse check
    float inside = (x * x) / (curEndX * curEndX) +
                    (y * y) / (curEndY * curEndY);

    // IES
    vec2 curXY = vec2(
        x / curEndX,
        y / curEndY
    );
    float curXYlength = length(curXY);
    intensity *= ramp(colorsDistances, curXYlength).x;
    // intensity = 1.0 - length(curXY);

    // Feather
    if (feather == 0.0) {
        intensity *= step(inside, 1.0);
    } else {
        float fOut = 1.0;
        float fIn = 1.0 - feather;
        intensity *= smoothstep(fOut, fIn, inside);
    }

    // Falloff
    if (falloff > 0.0) {
      intensity *= pow(1.0 - t, falloff);
    }

    // Invert
    if (invertToggle) {
        intensity = 1.0 - intensity;
        t = 1.0 - intensity;
    }

    // Color Falloff
    float colorIntensity = 1.0;
    if (colorFalloff > 0.0) {
        colorIntensity *= pow(1.0 - t, colorFalloff);
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

    // Light Settings

    vec3 lightPos = read3dPoint(0);
    vec3 lightVx = read3dPoint(1);
    vec3 lightVy = read3dPoint(2);
    vec3 lightVz = read3dPoint(3);

    // Shape

    bool invertToggle = _PixelsWorld_checkbox[2]; 
    float length = _PixelsWorld_slider[1];
    vec2 angles = clamp(vec2(_PixelsWorld_slider[2], _PixelsWorld_slider[3]) / 2.0, 0.001, 90.0);
    float curvature = _PixelsWorld_slider[4];
    float falloff = _PixelsWorld_slider[5];
    float feather = _PixelsWorld_slider[6];

    // IES

    vec3 iesColor1 = vec3(readPoint(0).x);
    vec3 iesColor2 = vec3(readPoint(1).x);
    vec3 iesColor3 = vec3(readPoint(2).x);
    vec3 iesColor4 = vec3(readPoint(3).x);
    vec3 iesColor5 = vec3(readPoint(4).x);
    vec3 iesColor6 = vec3(readPoint(5).x);

    float iesDistance1 = readPoint(0).y;
    float iesDistance2 = readPoint(1).y;
    float iesDistance3 = readPoint(2).y;
    float iesDistance4 = readPoint(3).y;
    float iesDistance5 = readPoint(4).y;
    float iesDistance6 = readPoint(5).y;

    float iesPreset = readPoint(6).x;
    vec4 iesChosenPreset[6];

    if (iesPreset != 5) {
        for (int i = 0; i < 6; i++) {
            iesChosenPreset[i] = getIESpreset(int(iesPreset), i);
        }
    } else {
        iesChosenPreset[0] = vec4(iesColor1, iesDistance1);
        iesChosenPreset[1] = vec4(iesColor2, iesDistance2);
        iesChosenPreset[2] = vec4(iesColor3, iesDistance3);
        iesChosenPreset[3] = vec4(iesColor4, iesDistance4);
        iesChosenPreset[4] = vec4(iesColor5, iesDistance5);
        iesChosenPreset[5] = vec4(iesColor6, iesDistance6);
    }

    // Color

    float intensity = _PixelsWorld_slider[7];
    float saturation = _PixelsWorld_slider[8];
    vec3 colorNear = _PixelsWorld_color[0].xyz;
    bool colorFarToggle = _PixelsWorld_checkbox[1];
    vec3 colorFar = _PixelsWorld_color[1].xyz;
    float colorFalloff = _PixelsWorld_slider[9];

    // Code

    vec2 uv = _PixelsWorld_uv;

    vec3 curPos = getPosition(uv, false);

    vec3 draw = spot(curPos, lightPos, lightVx, lightVy, lightVz, invertToggle, length, angles, curvature, falloff, feather, intensity, saturation, colorNear, colorFarToggle, colorFar, colorFalloff, iesChosenPreset);
    
    outColor = vec4(draw, 1.0);

}