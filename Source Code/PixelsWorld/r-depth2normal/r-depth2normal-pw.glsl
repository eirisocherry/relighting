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

vec3 getPosition(vec2 uv) {

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

    vec4 worldPos = _PixelsWorld_camera_matrix * vec4(localPos, 1.0); // Multiplication order is important (Matrix x Vec4)

    return worldPos.xyz;
    
}

vec3 getNormal(vec2 uv) {

    vec2 uvPixel = 1.0 / (_PixelsWorld_resolution * _PixelsWorld_downsample);

    vec3 right  = getPosition(uv + vec2(uvPixel.x, 0.0));
    vec3 left   = getPosition(uv - vec2(uvPixel.x, 0.0));
    vec3 top    = getPosition(uv + vec2(0.0, uvPixel.y));
    vec3 bottom = getPosition(uv - vec2(0.0, uvPixel.y));

    vec3 dx = right - left;
    vec3 dy = top - bottom;

    return normalize(cross(dx, dy));

}

vec3 getNormalImproved(vec2 uv) {
    
    vec2 uvPixel = 1.0 / (_PixelsWorld_resolution * _PixelsWorld_downsample);

    vec3 c  = getPosition(uv);
    vec3 l1 = getPosition(uv - vec2(uvPixel.x, 0.0));
    vec3 r1 = getPosition(uv + vec2(uvPixel.x, 0.0));
    vec3 l2 = getPosition(uv - vec2(2.0 * uvPixel.x, 0.0));
    vec3 r2 = getPosition(uv + vec2(2.0 * uvPixel.x, 0.0));
    vec3 b1 = getPosition(uv - vec2(0.0, uvPixel.y));
    vec3 t1 = getPosition(uv + vec2(0.0, uvPixel.y));
    vec3 b2 = getPosition(uv - vec2(0.0, 2.0 * uvPixel.y));
    vec3 t2 = getPosition(uv + vec2(0.0, 2.0 * uvPixel.y));

    float dl = abs(l1.x * l2.x / (2.0 * l2.x - l1.x) - c.x);
    float dr = abs(r1.x * r2.x / (2.0 * r2.x - r1.x) - c.x);
    float db = abs(b1.y * b2.y / (2.0 * b2.y - b1.y) - c.y);
    float dt = abs(t1.y * t2.y / (2.0 * t2.y - t1.y) - c.y);

    vec3 dpdx = (dl < dr) ? c - l1 : r1 - c;
    vec3 dpdy = (db < dt) ? c - b1 : t1 - c;

    return normalize(cross(dpdx, dpdy));

}

// Main

void mainImage(out vec4 outColor, in vec2 fragCoord) {

    // Depth

    depthFar = _PixelsWorld_slider[0];
    depthBlackIsNear = _PixelsWorld_checkbox[0];

    // Settings

    bool normalize = bool(_PixelsWorld_checkbox[1]);
    bool improve = bool(_PixelsWorld_checkbox[2]);

    // Code

    vec2 uv = _PixelsWorld_uv;

    vec3 normal = vec3(0.0);

    if (improve) {
        normal = getNormalImproved(uv);
    } else {
        normal = getNormal(uv);
    }

    if (normalize) {
        outColor = vec4(normal * 0.5 + 0.5, 1.0);
    } else {
        outColor = vec4(normal, 1.0);
    }

}