void mainImage(out vec4 outColor, in vec2 fragCoord) {

    vec3 draw = vec3(0.0);

    vec3 layer0 = texture(_PixelsWorld_layer[0], _PixelsWorld_uv).xyz;
    vec3 layer1 = texture(_PixelsWorld_layer[1], _PixelsWorld_uv).xyz;
    vec3 layer2 = texture(_PixelsWorld_layer[2], _PixelsWorld_uv).xyz;
    vec3 layer3 = texture(_PixelsWorld_layer[3], _PixelsWorld_uv).xyz;
    vec3 layer4 = texture(_PixelsWorld_layer[4], _PixelsWorld_uv).xyz;
    vec3 layer5 = texture(_PixelsWorld_layer[5], _PixelsWorld_uv).xyz;
    vec3 layer6 = texture(_PixelsWorld_layer[6], _PixelsWorld_uv).xyz;
    vec3 layer7 = texture(_PixelsWorld_layer[7], _PixelsWorld_uv).xyz;
    vec3 layer8 = texture(_PixelsWorld_layer[8], _PixelsWorld_uv).xyz;
    vec3 layer9 = texture(_PixelsWorld_layer[9], _PixelsWorld_uv).xyz;

    draw += layer0;
    draw += layer1;
    draw += layer2;
    draw += layer3;
    draw += layer4;
    draw += layer5;
    draw += layer6;
    draw += layer7;
    draw += layer8;
    draw += layer9;

    outColor = vec4(draw, 1.0);

}