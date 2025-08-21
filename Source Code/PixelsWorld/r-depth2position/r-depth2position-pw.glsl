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

void mainImage(out vec4 fragColor, in vec2 fragCoord) {

    // Depth

    depthFar = _PixelsWorld_slider[0];
    depthBlackIsNear = _PixelsWorld_checkbox[0];

    // Code

    vec2 uv = _PixelsWorld_uv;

    vec3 worldPos = getPosition(uv, false);

    fragColor = vec4(worldPos, 1.0); 

}