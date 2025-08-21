
function h1(index) {
console.log(`
// Local Light Settings ${index}

TOPIC_ADD_LOCAL_LIGHT_SETTINGS_${index},
CHECKBOX_TOGGLE_${index},
POINT_3D_POSITION_${index},
CHECKBOX_INVERT_${index},
SLIDER_RADIUS_${index},
SLIDER_FALLOFF_${index},
SLIDER_INTENSITY_${index},
SLIDER_SATURATION_${index},
COLOR_NEAR_${index},
CHECKBOX_COLOR_FAR_${index},
COLOR_FAR_${index},
SLIDER_COLOR_FAR_FALLOFF_${index},
TOPIC_END_LOCAL_LIGHT_SETTINGS_${index},
`);
}

function h2(index) {
console.log(`
// Local Light Settings ${index}

bool lightToggle${index};
float lightPosX${index}; float lightPosY${index}; float lightPosZ${index};
float radius${index};
float falloff${index};
float intensity${index};
float saturation${index};
float colorNearR${index}; float colorNearG${index}; float colorNearB${index};
bool colorFarToggle${index};
float colorFarR${index}; float colorFarG${index}; float colorFarB${index};
float colorFalloff${index};
`);
}

function cpp1(index) {
console.log(`
////////////////////////////
// Local Light Settings ${index} //
////////////////////////////

AEFX_CLR_STRUCT(def);
PF_ADD_TOPIC("Light Settings ${index}", TOPIC_ADD_LOCAL_LIGHT_SETTINGS_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_CHECKBOXX("Toggle ${index}", FALSE, 0, CHECKBOX_TOGGLE_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_POINT_3D("Position ${index}", 0, 0, 0, POINT_3D_POSITION_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_CHECKBOXX("Invert ${index}", FALSE, 0, CHECKBOX_INVERT_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Radius ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(1000.000),
  PF_FpLong(200.000),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_RADIUS_${index}
);

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Falloff ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(1.000),
  PF_FpLong(2.000),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_FALLOFF_${index}
);

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Intensity ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(1.000),
  PF_FpLong(1.000),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_INTENSITY_${index}
);

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Saturation ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(1.000),
  PF_FpLong(1.000),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_SATURATION_${index}
);

AEFX_CLR_STRUCT(def);
PF_ADD_COLOR("Near Color ${index}", 255, 205, 120, COLOR_NEAR_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_CHECKBOXX("Far Color Toggle ${index}", FALSE, 0, CHECKBOX_COLOR_FAR_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_COLOR("Far Color ${index}", 255, 157, 0, COLOR_FAR_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Far Color Falloff ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(1.000),
  PF_FpLong(1.000),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_COLOR_FAR_FALLOFF_${index}
);

AEFX_CLR_STRUCT(def);
PF_END_TOPIC(TOPIC_END_LOCAL_LIGHT_SETTINGS_${index});
`);
}

function cpp2(index) {
console.log(`
// Local Light Settings ${index}

ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_TOGGLE_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->lightToggle[${index - 1}] = static_cast<bool>(cur_param.u.bd.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_POSITION_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->lightPosX[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->lightPosY[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->lightPosZ[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->invertToggle[${index - 1}] = static_cast<bool>(cur_param.u.bd.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_RADIUS_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->radius[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->falloff[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_INTENSITY_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->intensity[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_SATURATION_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->saturation[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_NEAR_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->colorNearR[${index - 1}] = static_cast<float>(cur_param.u.cd.value.red);
infoP->colorNearG[${index - 1}] = static_cast<float>(cur_param.u.cd.value.green);
infoP->colorNearB[${index - 1}] = static_cast<float>(cur_param.u.cd.value.blue);

ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_COLOR_FAR_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->colorFarToggle[${index - 1}] = static_cast<bool>(cur_param.u.bd.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, COLOR_FAR_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->colorFarR[${index - 1}] = static_cast<float>(cur_param.u.cd.value.red);
infoP->colorFarG[${index - 1}] = static_cast<float>(cur_param.u.cd.value.green);
infoP->colorFarB[${index - 1}] = static_cast<float>(cur_param.u.cd.value.blue);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_COLOR_FAR_FALLOFF_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->colorFalloff[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);
`);
}

function cpp3(index) {
console.log(`
// Local Light Settings ${index}

infoP->lightToggle${index},
infoP->lightPosX${index}, infoP->lightPosY${index}, infoP->lightPosZ${index},
infoP->radius${index},
infoP->falloff${index},
infoP->intensity${index},
infoP->saturation${index},
infoP->colorNearR${index}, infoP->colorNearG${index}, infoP->colorNearB${index},
infoP->colorFarToggle${index},
infoP->colorFarR${index}, infoP->colorFarG${index}, infoP->colorFarB${index},
infoP->colorFalloff${index},
`);
}

function cpp4(index) {
console.log(`
// Local Light Settings ${index}

bool lightToggle${index},
float lightPosX${index}, float lightPosY${index}, float lightPosZ${index},
float radius${index},
float falloff${index},
float intensity${index},
float saturation${index},
float colorNearR${index}, float colorNearG${index}, float colorNearB${index},
bool colorFarToggle${index},
float colorFarR${index}, float colorFarG${index}, float colorFarB${index},
float colorFalloff${index},
`);
}

function cu1(index) {
  cpp4(index);
}

function cu2(index) {
console.log(`
// Local Light Settings ${index}

lightToggle${index},
lightPosX${index}, lightPosY${index}, lightPosZ${index},
radius${index},
falloff${index},
intensity${index},
saturation${index},
colorNearR${index}, colorNearG${index}, colorNearB${index},
colorFarToggle${index},
colorFarR${index}, colorFarG${index}, colorFarB${index},
colorFalloff${index},
`);
}

function cu3(index) {
console.log(`
// Local Light Settings ${index}

((bool)(lightToggle${index}))
((float)(lightPosX${index})) ((float)(lightPosY${index})) ((float)(lightPosZ${index}))
((float)(radius${index}))
((float)(falloff${index}))
((float)(intensity${index}))
((float)(saturation${index}))
((float)(colorNearR${index})) ((float)(colorNearG${index})) ((float)(colorNearB${index}))
((bool)(colorFarToggle${index}))
((float)(colorFarR${index})) ((float)(colorFarG${index})) ((float)(colorFarB${index}))
((float)(colorFalloff${index}))
`);
}



for (let i = 1; i <= 10; i++) {
  //h1(i);
  //  h2(i);

  //cpp1(i);
  cpp2(i);
  //cpp3(i);
  //cpp4(i);

  // cu1(i);
  //cu2(i);
  //cu3(i);
}

