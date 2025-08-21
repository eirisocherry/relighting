
function h1(index) {
console.log(`
// Local Light Settings ${index}

TOPIC_ADD_LOCAL_LIGHT_SETTINGS_${index},
CHECKBOX_TOGGLE_${index},
POINT_3D_POSITION_${index},
POINT_3D_VX_${index},
POINT_3D_VY_${index},
POINT_3D_VZ_${index},
CHECKBOX_INVERT_${index}, //---
SLIDER_LENGTH_${index},
SLIDER_ANGLEX_${index},
SLIDER_ANGLEY_${index},
SLIDER_CURVATURE_${index},
SLIDER_FEATHER_${index},
SLIDER_FALLOFF_${index},
POPUP_IES_${index},	//---
POINT_2D_IES_1_${index},
POINT_2D_IES_2_${index},
POINT_2D_IES_3_${index},
POINT_2D_IES_4_${index},
POINT_2D_IES_5_${index},
POINT_2D_IES_6_${index},
SLIDER_INTENSITY_${index}, //---
SLIDER_SATURATION_${index},
COLOR_NEAR_${index},
CHECKBOX_COLOR_FAR_${index},
COLOR_FAR_${index},
SLIDER_COLOR_FAR_FALLOFF_${index},
TOPIC_END_LOCAL_LIGHT_SETTINGS_${index},
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
PF_ADD_POINT_3D("Vector X ${index}", 0, 0, 0, POINT_3D_VX_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_POINT_3D("Vector Y ${index}", 0, 0, 0, POINT_3D_VY_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_POINT_3D("Vector Z ${index}", 0, 0, 0, POINT_3D_VZ_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_CHECKBOXX("Invert ${index}", FALSE, 0, CHECKBOX_INVERT_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Length ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(1000.000),
  PF_FpLong(200.000),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_LENGTH_${index}
);

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Angle X ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(100.000),
  PF_FpLong(30.000),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_ANGLEX_${index}
);

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Angle Y ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(100.000),
  PF_FpLong(30.000),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_ANGLEY_${index}
);

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Curvature ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(1.000),
  PF_FpLong(1.000),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_CURVATURE_${index}
);

AEFX_CLR_STRUCT(def);
PF_ADD_FLOAT_SLIDERX(
  "Feather ${index}",
  PF_FpLong(0.000),
  PF_FpLong(9999999.000),
  PF_FpLong(0.000),
  PF_FpLong(1.000),
  PF_FpLong(0.500),
  PF_Precision_THOUSANDTHS,
  0,
  0,
  SLIDER_FEATHER_${index}
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
PF_ADD_POPUP("IES ${index}", 5, 1, "None|1|2|3|Custom", POPUP_IES_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_POINT("Brightness / Distance 1 ${index}", 0, 0, FALSE, POINT_2D_IES_1_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_POINT("Brightness / Distance 2 ${index}", 0, 0, FALSE, POINT_2D_IES_2_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_POINT("Brightness / Distance 3 ${index}", 0, 0, FALSE, POINT_2D_IES_3_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_POINT("Brightness / Distance 4 ${index}", 0, 0, FALSE, POINT_2D_IES_4_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_POINT("Brightness / Distance 5 ${index}", 0, 0, FALSE, POINT_2D_IES_5_${index});

AEFX_CLR_STRUCT(def);
PF_ADD_POINT("Brightness / Distance 6 ${index}", 0, 0, FALSE, POINT_2D_IES_6_${index});


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

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_VX_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->lightVxX[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->lightVxY[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->lightVxZ[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_VY_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->lightVyX[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->lightVyY[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->lightVyZ[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_VZ_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->lightVzX[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->lightVzY[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->lightVzZ[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);


ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->invertToggle[${index - 1}] = static_cast<bool>(cur_param.u.bd.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_LENGTH_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->length[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEX_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->angleX[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_ANGLEY_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->angleY[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_CURVATURE_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->curvature[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FEATHER_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->feather[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, SLIDER_FALLOFF_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->falloff[${index - 1}] = static_cast<float>(cur_param.u.fs_d.value);


ERR(PF_CHECKOUT_PARAM(in_dataP, POPUP_IES_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->ies[${index - 1}] = static_cast<int>(cur_param.u.pd.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_1_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->iesBrightness1[${index - 1}] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
infoP->iesDistance1[${index - 1}] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_2_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->iesBrightness2[${index - 1}] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
infoP->iesDistance2[${index - 1}] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_3_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->iesBrightness3[${index - 1}] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
infoP->iesDistance3[${index - 1}] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_4_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->iesBrightness4[${index - 1}] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
infoP->iesDistance4[${index - 1}] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_5_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->iesBrightness5[${index - 1}] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
infoP->iesDistance5[${index - 1}] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_IES_6_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->iesBrightness6[${index - 1}] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
infoP->iesDistance6[${index - 1}] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);


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

for (let i = 1; i <= 10; i++) {
  //h1(i);

  //cpp1(i);
  cpp2(i);

}