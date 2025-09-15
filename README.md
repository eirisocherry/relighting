# Relighting
An After Effects tool that allows you to quickly relight a scene  
Tutorial: https://www.youtube.com/watch?v=RzH9I8wUUOc  
Author: https://www.youtube.com/@shy_rikki  
Report a bug: https://discord.gg/AAJxThhbBf  
Free AE project: https://t.me/works_by_rikki/68  
If you wish to donate ❤️: https://boosty.to/shy_rikki  

<img width="960" height="540" alt="Screenshot_1" src="https://github.com/user-attachments/assets/1341d857-e10f-43ac-ad97-b7234c545be3" />  

## Features
- Point Light  
- Spot Light  
- Rect Light  
- Directional Light  
- Rim Light  
- Ambient Occlusion (edge shadows)  
- Position Pass  
- Normal Pass  
- Normal Remap  
- Clamp (clamps values to 0-1 range)  
- Unmult (removes black)  

## Requirements
- After Effects 2019/2021-2025 (The tool doesn't work in AE 2020)  
- NVIDIA GPU that supports CUDA  
- EXR Depth Map  
- Camera Data  
- Normal Pass (Optional)  

What can meet the requirements?  
**Games**: CS2, CSGO, COD4 (partially, cuz bad depth), BO2 (partially, cuz bad depth)  
**3D Software**: almost any (Blender, Cinema 4D, Unreal Engine 5 etc)  

## Installation
1. Download the tool: https://github.com/eirisocherry/relighting/releases  
2. Move `r-relighting` folder and `r-relighting.jsx` script to:  
`C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Scripts\ScriptUI Panels`  
3. Move `Rikki` folder that contains plugins to:  
   `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore`  
4. Restart After Effects  

## Usage

<details>
<summary> First Steps </summary>
<br>

1. Launch After Effects, go to `Window`, scroll down and open the `r-relighting.jsx` script
<img width="641" height="840" alt="image" src="https://github.com/user-attachments/assets/d9c96967-3840-493e-860f-cee18a0d444b" />  

2. Set Rendering Engine to `Mercury GPU Acceleration (CUDA)`  
<img width="335" height="118" alt="image" src="https://github.com/user-attachments/assets/b4d03007-479e-466a-be39-2e66043c8519" />

3. Set project to `32 bit`
<img width="296" height="68" alt="image" src="https://github.com/user-attachments/assets/125d2801-12e4-48ed-be26-3540c6546c3e" />  
  
4. Set expressions engine to `JavaScript`  
<img width="497" height="94" alt="image" src="https://github.com/user-attachments/assets/113f8f1e-2245-4d46-9283-593e6517a340" />  
  
<br>
</details>

<details>
<summary> CSGO & CS2 </summary>
<br>

1. Record the required layers: game, exr depth, camera  
CSGO Tutorials: https://www.youtube.com/watch?v=PtdO_I-fBRo&list=PLiyMyFJsq2_VbQNn3nL4sYAXbaIQRQsZH  
EXR Depth Guide (10:49): https://youtu.be/NE5nAPHn_P4?list=PLiyMyFJsq2_VbQNn3nL4sYAXbaIQRQsZH&t=649  
CS2 Tutorials: https://github.com/eirisocherry/cs2-editing/wiki  
3. Import layers, camera data and setup an exr depth: https://www.youtube.com/watch?v=FWEqkaiXNM0  
4. Select a depth map and create a setup by pressing `[+]` button  

<img width="624" height="379" alt="image" src="https://github.com/user-attachments/assets/33a08da5-0bc1-4c15-8377-b5ffbe99de25" />  

4. Adjust `Depth Settings`:  
`Depth Black Is Near` whether a black color is near on your depth map or not: `unchecked`  
`Depth Far` the farthest depth point value: `25000` (the same value you set in `EXtractoR` effect)  

<img width="352" height="254" alt="Screenshot_5" src="https://github.com/user-attachments/assets/d0c7bbd9-0ee5-41a8-91ee-500354944968" />  
  
<img width="336" height="152" alt="Screenshot_4" src="https://github.com/user-attachments/assets/619dc92a-11f8-4010-b684-510ed7beab39" />  
  
5. a) Select `Depth Projection` depth layer and use `Project On Point` cursor to select where you want to project an object  
b) Select something from the dropdowm menu, ex: `Point Advanced`  
c) Press `Project` and wait a bit... Done!  
  
<img width="1284" height="611" alt="Screenshot_2" src="https://github.com/user-attachments/assets/3ad52698-e915-4db4-8b0c-57c14c66a5e5" />  
<img width="1261" height="524" alt="Screenshot_3" src="https://github.com/user-attachments/assets/a7206384-f213-4173-b5b8-50919014d5f8" />  

<br>
</details>



<details>
<summary> COD4 </summary>
<br>

**[NOTE]**  
**"Auto Orient" function, Shadows, Ambient Occlusion, Rim Light and Depth2Normal are not going to work correctly, because COD4's depth map is not 32-bit.**  

1. Record the required layers  
Tutorial by Politoo: https://www.youtube.com/watch?v=VjyNZsYzVWg  
CODMVM: https://codmvm.com/  

<img width="552" height="326" alt="Screenshot_1" src="https://github.com/user-attachments/assets/8060be0e-6acc-4fc8-b1de-dc582c08892c" />  
  
```
mvm_output_directory "S:\Screens"      // output directory
mvm_avidemo_fps 0                      // disables default screen recording
mvm_streams_fps 125                    // fps to record layers in
mvm_streams_passes mvm_w mvm_wd mvm_wn // layers: game, depth, normal
mvm_streams_depthFormat 2              // depth format: 2 - rainbow (more precise)
mvm_export_format avi                  // output format must be set to avi, otherwise depth format will not work
mvm_streams_aeExport 1                 // export camera
mvm_streams_aeExport_sun 1             // export sun
```
2. Convert the layers using my `ftool-converter.bat`  
Download: https://github.com/eirisocherry/ftools/blob/main/ftool-converter.bat  
Guide: https://github.com/eirisocherry/ftools/tree/main  
3. a) Download: https://github.com/gmzorz/MVMAETools/blob/main/Support%20Files/Scripts/ScriptUI%20Panels/MVMTools.jsx  
b) Move `MVMTools.jsx` script to:  
`C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Scripts\ScriptUI Panels`  
c) Import the layers, camera data and convert a depth  
5. a) Rename Normal Map to `Normal Pass 1`  
b) Apply `Normal Remap` effect to the `Normal Pass 1` layer  
c) Copy the settings  
```
Input Is Normalized: `Checked`
X -> -X
Y -> -Z
Z -> -Y
Normalize Output: `Checked`
```  

<img width="1500" height="792" alt="Screenshot_7" src="https://github.com/user-attachments/assets/e9aef0bc-276e-45d0-8298-5df4402ea581" />  
  
5. Select a depth map and create a setup by pressing `[+]` button  
<img width="619" height="394" alt="Screenshot_8" src="https://github.com/user-attachments/assets/06709fe2-1a31-4250-b06d-f62e2d20a999" />  
  
6. Adjust `Depth Settings`:  
`Depth Black Is Near` whether a black color is near on your depth map or not: `unchecked`  
`Depth Far` the farthest depth point value: `4080`  
  
<img width="337" height="129" alt="image" src="https://github.com/user-attachments/assets/9f6993fd-d248-4986-8323-1dffca0aad65" />  
  
7. a) Select `Depth Projection` depth layer and use `Project On Point` cursor to select where you want to project an object  
b) Select something from the dropdowm menu, ex: `Point Advanced`  
c) Press `Project` and wait a bit... Done!  
  
<img width="1494" height="775" alt="image" src="https://github.com/user-attachments/assets/a2cea40f-fc40-4387-b86a-c2ce2fb6d22a" />  
  
<img width="1478" height="699" alt="image" src="https://github.com/user-attachments/assets/56f94c7e-5e44-4088-bacf-5b65f2ecdcf0" />  
  
<br>
</details>



<details>
<summary> Blender </summary>
<br>

**[NOTE]**  
**"Why would I use the relighting tool with 3d software?" You might ask.**  
**The only benefit I see is quick object placement.**  
**You can easily place nulls and link anything to them: images, overlays, flares and more**  

1. Export and import the camera to after effects: https://www.youtube.com/watch?v=V1ZpQJ2jZ8Q  
Blender Camera Exporter Script: https://github.com/sobotka/blender-addons-contrib/blob/master/io_export_after_effects.py  

<img width="1919" height="727" alt="image" src="https://github.com/user-attachments/assets/2d23b7ab-d2bc-492f-97a8-83f460c021ba" />  

Remember the scale value, it will be used for "Depth Far" parameter later:  

<img width="351" height="524" alt="image" src="https://github.com/user-attachments/assets/d0c31105-1f7c-4e91-b870-3b1433822c39" />  

2. Render the required layers:  
a) Render settings:  
<img width="415" height="223" alt="image" src="https://github.com/user-attachments/assets/a3cbe6d6-2ff3-41c1-a2e7-6bd48ffead6c" />

b) Color Managment settings (set to your liking):  
<img width="415" height="544" alt="image" src="https://github.com/user-attachments/assets/a3944871-c22d-4c71-9791-7a16e5a6d121" />  

c) Required Layers:  
<img width="1091" height="663" alt="Screenshot_48" src="https://github.com/user-attachments/assets/b2670c89-9430-4f67-83b6-730f1e248ae3" />  

3. Import the layers, setup them and rename as on the screenshots:  

**Beauty**  
OpenColorIO plugin: https://www.fnord.com/  

<img width="1711" height="821" alt="image" src="https://github.com/user-attachments/assets/27fb13b4-a554-4295-96e1-a403bf33af45" />  
  
  
**Normal Pass**  
  
<img width="1708" height="765" alt="image" src="https://github.com/user-attachments/assets/641a1067-b04f-49f2-9387-2f198cdce808" />  
  
  
**Depth**  
  
<img width="821" height="147" alt="image" src="https://github.com/user-attachments/assets/f0dd21c8-3afa-415f-87ce-f41c3b6bae94" />  
  
<img width="701" height="530" alt="image" src="https://github.com/user-attachments/assets/dbacbf1a-dcd8-47cf-87f5-1711d9797c80" />  

4. Select a depth map and create a setup by pressing `[+]` button  

<img width="687" height="403" alt="image" src="https://github.com/user-attachments/assets/c525a637-67c5-4994-9877-aa4c4956b820" />  

5. Adjust `Depth Settings`:  
`Depth Black Is Near`: `checked` (yes, visually the depth is white, but because the depth's range is not 0-1, you need to check it)  
`Depth Far`: `100` (because, when exporting the camera, scale was set to 100)  
  
<img width="510" height="125" alt="image" src="https://github.com/user-attachments/assets/b9fef017-7c5d-46a9-a4b5-8187aae10cbc" />  

6. a) Select `Depth Projection` depth layer and use `Project On Point` cursor to select where you want to project an object  
b) Select something from the dropdowm menu, ex: `Point Advanced`  
c) Press `Project` and wait a bit... Done!  

<img width="1315" height="806" alt="image" src="https://github.com/user-attachments/assets/f2eb303c-5e59-45a3-af5e-e0a0051997bd" />  
  
<img width="1130" height="952" alt="image" src="https://github.com/user-attachments/assets/822cd1bd-732c-4fc1-925c-bca2533e6fe1" />  

<br>
</details>



<details>
<summary> Unreal Engine 5 </summary>
<br>

1. Export UE camera as fbx and import it to Blender: https://www.youtube.com/watch?v=zcAIfq8WfNU  
2. Export Blender camera and import it to After Effects (check the "Blender" category I wrote above)  

I haven't tested this workflow yet, but, theoretically, it should work just fine.  

<br>
</details>



## Important Notes
- The script is heavy and may crash your After Effects, but don't worry!  
It automatically saves your project before doing any actions, so even if your AE will crash, you will not lose any of your progress  
- During the process AE may freeze time by time, don't panic, just wait. It happens because light effects have hundreads of parameters and expressions, AE needs some time to process them.  
