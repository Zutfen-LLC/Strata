// Strata metal/kernels/gdn.metal:90-92, compensated positive log1p only.
// Do not copy Metal's surrounding gate exp: CUDA gdn_ab_multi returns log gate.
// Caller validates x in the declared positive domain before dispatch.
float strata_gdn_log1p_positive(float x,out float rounded_u,out float logarithm,out float numerator,out float denominator){
 precise float u=1.0+x;rounded_u=u;logarithm=0.0;numerator=0.0;denominator=0.0;
 if(u==1.0)return x;
 precise float l=log(u);precise float product=l*x;precise float d=u-1.0;
 precise float result=product/d;logarithm=l;numerator=product;denominator=d;return result;
}
