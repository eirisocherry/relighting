
function h1(index) {
console.log(`
// Local Light Settings ${index}

TOPIC_ADD_LOCAL_LIGHT_SETTINGS_${index},
CHECKBOX_LIGHT_TOGGLE_${index},
POINT_3D_POS1_${index}, //---
POINT_3D_VX1_${index},
POINT_3D_VY1_${index},
POINT_3D_VZ1_${index},
POINT_3D_RES1_${index},
POINT_3D_SCALE1_${index},
POINT_3D_POS2_${index}, //---
POINT_3D_VX2_${index},
POINT_3D_VY2_${index},
POINT_3D_VZ2_${index},
POINT_3D_RES2_${index},
POINT_3D_SCALE2_${index},
CHECKBOX_INVERT_${index}, //---
CHECKBOX_FEATHER_NORMALIZE_${index},
POINT_2D_FEATHERX_${index}, 
POINT_2D_FEATHERY_${index},
POINT_2D_FEATHERZ_${index},
SLIDER_FALLOFF_${index},
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
	PF_ADD_CHECKBOXX("Light Toggle ${index}", FALSE, 0, CHECKBOX_LIGHT_TOGGLE_${index});


	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Position 1 ${index}", 0, 0, 0, POINT_3D_POS1_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Vector X 1 ${index}", 0, 0, 0, POINT_3D_VX1_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Vector Y 1 ${index}", 0, 0, 0, POINT_3D_VY1_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Vector Z 1 ${index}", 0, 0, 0, POINT_3D_VZ1_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Resolution 1 ${index}", 0, 0, 0, POINT_3D_RES1_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Scale 1 ${index}", 0, 0, 0, POINT_3D_SCALE1_${index});


	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Position 2 ${index}", 0, 0, 0, POINT_3D_POS2_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Vector X 2 ${index}", 0, 0, 0, POINT_3D_VX2_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Vector Y 2 ${index}", 0, 0, 0, POINT_3D_VY2_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Vector Z 2 ${index}", 0, 0, 0, POINT_3D_VZ2_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Resolution 2 ${index}", 0, 0, 0, POINT_3D_RES2_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT_3D("Scale 2 ${index}", 0, 0, 0, POINT_3D_SCALE2_${index});


	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Invert ${index}", FALSE, 0, CHECKBOX_INVERT_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Feather Normalize ${index}", FALSE, 0, CHECKBOX_FEATHER_NORMALIZE_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Feather X ${index}", 0, 0, FALSE, POINT_2D_FEATHERX_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Feather Y ${index}", 0, 0, FALSE, POINT_2D_FEATHERY_${index});

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Feather Z ${index}", 0, 0, FALSE, POINT_2D_FEATHERZ_${index});

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

ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_LIGHT_TOGGLE_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->lightToggle[${index - 1}] = static_cast<bool>(cur_param.u.bd.value);


ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_POS1_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->posX1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->posY1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->posZ1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_VX1_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->vXx1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->vXy1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->vXz1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_VY1_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->vYx1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->vYy1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->vYz1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_VZ1_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->vZx1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->vZy1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->vZz1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_RES1_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->resX1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->resY1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->resZ1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_SCALE1_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->scaleX1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->scaleY1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->scaleZ1[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);


ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_POS2_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->posX2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->posY2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->posZ2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_VX2_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->vXx2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->vXy2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->vXz2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_VY2_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->vYx2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->vYy2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->vYz2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_VZ2_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->vZx2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->vZy2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->vZz2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_RES2_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->resX2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->resY2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->resZ2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_3D_SCALE2_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->scaleX2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.x_value);
infoP->scaleY2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.y_value);
infoP->scaleZ2[${index - 1}] = static_cast<float>(cur_param.u.point3d_d.z_value);


ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_INVERT_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->invertToggle[${index - 1}] = static_cast<bool>(cur_param.u.bd.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, CHECKBOX_FEATHER_NORMALIZE_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->featherNormalize[${index - 1}] = static_cast<bool>(cur_param.u.bd.value);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_FEATHERX_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->featherX1[${index - 1}] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
infoP->featherX2[${index - 1}] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_FEATHERY_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->featherY1[${index - 1}] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
infoP->featherY2[${index - 1}] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

ERR(PF_CHECKOUT_PARAM(in_dataP, POINT_2D_FEATHERZ_${index}, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
infoP->featherZ1[${index - 1}] = static_cast<float>(cur_param.u.td.x_value / 65536.0f);
infoP->featherZ2[${index - 1}] = static_cast<float>(cur_param.u.td.y_value / 65536.0f);

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

for (let i = 1; i <= 10; i++) {
  //h1(i);

  //cpp1(i);
  cpp2(i);

}

