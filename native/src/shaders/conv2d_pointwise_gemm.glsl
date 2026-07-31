#version 450 core
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{float d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint iw;uint ih;uint ic;uint ow;uint oh;uint oc;
 uint kernel;uint stride;int padding;uint has_bias;}p;
#define KT 32
shared float it[32*KT];shared float wt[32*KT];
void main(){
 uint count=p.iw*p.ih,sb=gl_WorkGroupID.x*32+gl_LocalInvocationID.x*4;
 uint cb=gl_WorkGroupID.y*32+gl_LocalInvocationID.y*4,lane=gl_LocalInvocationID.y*8+gl_LocalInvocationID.x;
 float sums[4][4];for(uint c=0;c<4;++c)for(uint s=0;s<4;++s)sums[c][s]=0;
 for(uint kb=0;kb<p.ic;kb+=KT){
  for(uint n=lane;n<32*KT;n+=64){uint so=n/KT,k=kb+n%KT,sp=gl_WorkGroupID.x*32+so;
   it[n]=sp<count&&k<p.ic?i.d[k*count+sp]:0;}
  for(uint n=lane;n<32*KT;n+=64){uint co=n/KT,k=kb+n%KT,ch=gl_WorkGroupID.y*32+co;
   wt[n]=ch<p.oc&&k<p.ic?w.d[ch*p.ic+k]:0;}
  barrier();uint kc=min(KT,p.ic-kb);
  for(uint k=0;k<kc;++k){float iv[4],wv[4];
   for(uint s=0;s<4;++s)iv[s]=it[(gl_LocalInvocationID.x*4+s)*KT+k];
   for(uint c=0;c<4;++c)wv[c]=wt[(gl_LocalInvocationID.y*4+c)*KT+k];
   for(uint c=0;c<4;++c)for(uint s=0;s<4;++s)sums[c][s]+=wv[c]*iv[s];}
  barrier();}
 for(uint c=0;c<4;++c){uint ch=cb+c;if(ch>=p.oc)continue;float bias=p.has_bias!=0?b.d[ch]:0;
  for(uint s=0;s<4;++s){uint sp=sb+s;if(sp<count)o.d[ch*count+sp]=sums[c][s]+bias;}}
}
