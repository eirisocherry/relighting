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

// Main

vec3 point(vec3 curPos, vec3 lightPos, bool invertToggle, float radius, float fallOff, float intensityMultiplier, float saturation, vec3 lightColorNear, bool lightColorFarToggle, vec3 lightColorFar, float colorFalloff) {

    float dist = distance(curPos, lightPos);

    if (!invertToggle) {
        if (radius <= 0.0 || dist > radius) {
            return vec3(0.0);
        }
    }

    // Falloff
    float intensity = 1.0;
    float colorIntensity = 1.0;
    float t = clamp(dist / radius, 0.0, 1.0);
    if (invertToggle) {
        t = 1.0 - t;
    }
    if (fallOff > 0.0) {
        intensity *= pow(1.0 - t, fallOff);
    }
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

    // Shape

    bool invertToggle = _PixelsWorld_checkbox[1];
    float radius = _PixelsWorld_slider[1];
    float falloff = _PixelsWorld_slider[2];

    // Color

    float intensity = _PixelsWorld_slider[3];
    float saturation = _PixelsWorld_slider[4];
    vec3 colorNear = _PixelsWorld_color[0].xyz;
    bool colorFarToggle = _PixelsWorld_checkbox[2];
    vec3 colorFar = _PixelsWorld_color[1].xyz;
    float colorFalloff = _PixelsWorld_slider[5];

    // Code

    vec2 uv = _PixelsWorld_uv;

    vec3 curPos = getPosition(uv, true);

    vec3 draw = point(curPos, lightPos, invertToggle, radius, falloff, intensity, saturation, colorNear, colorFarToggle, colorFar, colorFalloff);
    
    outColor = vec4(draw, 1.0);

}