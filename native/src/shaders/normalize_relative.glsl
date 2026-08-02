#version 450 core
layout(local_size_x=256,local_size_y=1,local_size_z=1) in;
layout(std430,binding=0) buffer Depth { float data[]; } depth_data;
layout(std430,binding=1) readonly buffer Range { float data[]; } range_data;
layout(push_constant) uniform Parameters { uint count; } p;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=p.count)return;float lo=range_data.data[0],span=range_data.data[1]-lo;depth_data.data[i]=span>1.0e-12?clamp((depth_data.data[i]-lo)/span,0.0,1.0):0.0;}
