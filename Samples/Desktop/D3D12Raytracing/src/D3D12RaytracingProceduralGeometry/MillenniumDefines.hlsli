#ifndef MILLENNIUM_DEFINES_H
#define MILLENNIUM_DEFINES_H

#include "RayTracingHlslCompat.h"

// Flags set by caster
#define MC_RAYFLAG_SHADOW       (1<<0)
// #define MC_RAYFLAG_TRACER    (1<<1)
#define MC_RAYFLAG_PRIMARY      (1<<2)
#define MC_RAYFLAG_EMISSIVEONLY (1<<3)

// Representation of a ray that works across RT and non-RT shaders
struct RayInfo
{
	// In: How much does this ray contribute to the final pixel?
	// Out: This ray's color
    float3 color;
	// Parameter value at ray end
    float t;
	
	// Flags
    uint flags; // from #defines MC_RAYFLAG_*
	// Recursion depth for reflections and refraction
    int remainingDepth;
	// total number of calls to TraceRay() for this and child rays
    // int cost;

	// TODO split these three out!
	// World-space origin of the ray
    float3 originWS;
	// World-space direction of the ray. Not required to be normalised!
    float3 directionWS;
	// Parameter value at ray start
    float tMin;
};

RayInfo RayInfoFromRayIntersection(RayPayload rayPayload)
{
    RayInfo rayInfo;
    
    rayInfo.color = rayPayload.color.rgb;
    rayInfo.t = rayPayload.t; // allow for this to have been modified. Equals maximum t for a miss
    
    rayInfo.flags = rayPayload.flags;
    rayInfo.remainingDepth = rayPayload.remainingDepth;
    
    rayInfo.originWS = WorldRayOrigin();
    rayInfo.directionWS = WorldRayDirection();
    rayInfo.tMin = RayTMin();

    return rayInfo;
}

// Ray-sphere intersection
float HitTestSphere(float3 spherePos, float sphereRadius, float3 rayOrigin, float3 rayDir) {
  float3 d = rayOrigin - spherePos;
  float p1 = -dot(rayDir, d);
  float p2sqr = p1 * p1 - dot(d, d) + sphereRadius * sphereRadius;
  if (p2sqr < 0) {
    return 0;
  }
  float p2 = sqrt(p2sqr);
  float t = p1 - p2 > 0 ? p1 - p2 : p1 + p2;
  return t;
}

// Inverse of lerp()
float invlerp(float a, float b, float v)
{
    return (v - a) / (b - a);
}

// hash based 3d value noise
// function taken from https://www.shadertoy.com/view/XslGRr
// Created by inigo quilez - iq/2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// ported from GLSL to HLSL

float iq_hash(float n)
{
    return frac(sin(n) * 43758.5453);
}

float iq_noise(float3 x)
{
    // HLSL noise() is apparently not very good
    // The noise function returns a value in the range -1.0f -> 1.0f

    float3 p = floor(x);
    float3 f = frac(x);

    // Interpolation between noise points
    f = f * f * (3.0 - 2.0 * f);
    // Noise coordinate
    float n = p.x + p.y * 57.0 + 113.0 * p.z;

    return lerp(
        lerp(
            lerp(iq_hash(n + 0.0), iq_hash(n + 1.0), f.x),
            lerp(iq_hash(n + 57.0), iq_hash(n + 58.0), f.x),
            f.y),
        lerp(
            lerp(iq_hash(n + 113.0), iq_hash(n + 114.0), f.x),
            lerp(iq_hash(n + 170.0), iq_hash(n + 171.0), f.x),
            f.y),
        f.z);
}

// For LOD, num of octaves is fractional, and fraction
// is the interpolation value between octaves
float iq_fbm(float3 x, float H, float numOctaves = 5)
{
    float lastWholeOctave = floor(numOctaves);
    float fraction = frac(numOctaves);
    float G = exp2(-H);
    float f = 1.0; // frequency
    float a = 1.0; // amplitude
    float t = 0.0;
    for (int i = 0; i < lastWholeOctave; ++i)
    {
        t += a * iq_noise(f * x);
        f *= 2.0;
        a *= G;
    }

    float t_next = t + a * iq_noise(f * x);

    return lerp(t, t_next, fraction);
}

// Poorly named, these are scale factors for the optical depth for each
// colour channel.
float3 BryceHazeScaleForSunAlt(float sunAltitudeRadians) {
    float c = abs(sin(sunAltitudeRadians));
    return max(float3(4 - 8 * c, 2, 8 * c), float3(1, 1, 1));
}

// Exponential haze for atmosphere, using sky params
// returns the alpha for each channel
float3 ExponentialHaze(RayInfo rayInfo)
{
    // tau is the base optical depth
    // Y is up
    // Density
    float A = 0.005;
    // Falloff
    float B = 0.2;
    // Offset
    float Y = 0;

    float3 dir = rayInfo.directionWS;
    float3 rayDirectionWS = normalize(dir);
    
    float dist = rayInfo.t * length(dir);
    float o_y = rayInfo.originWS.y + Y;
    float d_y = rayDirectionWS.y;

    // Finally hit the small d_y edge case in practice
    if (B < 0.00001f) { B = 0.00001f; }
    float tau = (abs(d_y) < 0.00001f)
        ? A * dist * exp(-o_y * B)
        : (A/B) * exp(-o_y * B) * (1.0 - exp(-dist * d_y * B)) / d_y;

    float sunAltitudeRadians = radians(30.f);
    float3 tau3 = BryceHazeScaleForSunAlt(sunAltitudeRadians) * tau;

    // Presumed Bryce version / physically accurate
    return saturate(1-exp(-tau3));
}

void CalcWorldHaze(RayInfo rayInfo, out float3 fogColor, out float3 fogAlpha)
{
    fogColor = float3(0.839, 0.945, 1);
    fogAlpha = ExponentialHaze(rayInfo);
}

float3 ApplyWorldHaze(RayInfo rayInfo, float3 rayCol)
{
    float3 fogColor = { 0, 0, 0 };
    float3 fogAlpha = { 0, 0, 0 };
    
    CalcWorldHaze(rayInfo, fogColor, fogAlpha);
    
    return lerp(rayCol.rgb, fogColor, fogAlpha);
}

// Returns an alpha value for clouds at a point
// lo and hi are the range in noise values that correspond to 0 and 1 alpha
float BryceCloud(float2 cloud_uv, float lo, float hi, float z = 1.0f)
{
    float f = iq_fbm(float3(100.0f * cloud_uv, z), 1.0f, 5);
    return saturate(invlerp(lo, hi, f));
}

inline float3 SkyColor(float3 origin, float3 direction, float elapsedTime)
{
    float3 skyCol = float3(0,0,0);
    float4 cloudCol = float4(1, 1, 1, 0.4f);
    float cloudPlaneHeight = 1000.0f;
    float skyRadius = 10000;
    float3 windSpeed = float3(0.015f, 0, 0);
    

#if 0
    // Sun/Moon
#endif

    // Cloud plane
    
    // How far along ray until we hit sky height?
    float t = (cloudPlaneHeight - origin.y) / direction.y;
    float2 skyPos = origin.xz + t * direction.xz;
    // We don't have area or ddx+ddy so will look noisy
    //return float3(frac(sky_pos.x),frac(sky_pos.y),1.0f);
  
    // range: ...depends on H and numOctaves. Here, -2 to +2 maybe.
    // sky plane
    // float2 sky_coord = sky_pos.xy;
    // Unity world coords: Y is UP, Z is FORWARD, X is RIGHT
    
    float3 skyOrigin = float3(0, cloudPlaneHeight - skyRadius, 0);
    float sky_t = HitTestSphere(skyOrigin, skyRadius, origin, direction);
    float3 sky_p = origin + sky_t * direction - skyOrigin;
    // could scale these to taste
    float3 skyCoord = { asin(sky_p.x / skyRadius), asin(sky_p.z / skyRadius), 1.0f };
    
    skyCoord += windSpeed * elapsedTime;

    float f = BryceCloud(skyCoord.xy, 1.269, 1.654, skyCoord.z);
    return lerp(skyCol, cloudCol.rgb, f * cloudCol.a);
}

// Materials

#define MC_BRDF_RECEIVES_SHADOW     (1<<0)
#define MC_BRDF_RECEIVES_HAZE       (1<<1)
#define MC_BRDF_HAS_REFLECTION      (1<<2)
#define MC_BRDF_HAS_REFRACTION      (1<<3)

struct BRDF
{
    float4 baseColor;
    
    float3 shadingNormalWS;
    float3 geometricNormalWS;

    // Trying to keep to 8 bits for
    // eventual G-buffer packing
    uint flags8;
    // _ReceivesShadow;
    // _ReceivesHaze;
    // _HasReflection;
    // _HasRefraction;
    
    float luminosity;
    float diffuse;
    float specular;
    float roughness;
    
    float reflectivity;
    float ior;
    float metallicity;
};

float3 ProcWaterNormal(float3 positionWS, float3 normalWS, float elapsedTime) {
    float3 waveSpeed = float3(0.625, 0.0, 0.1875);

    float3 p_t = positionWS + waveSpeed * elapsedTime;
    // TODO HACK - derive from normalWS
    float3 tangentWS = { 1, 0, 0 };
    float3 bitangentWS = { 0, 0, 1 };

    float Km = 0.1;
    float H = 1.7;
    float octaves = 3;
    float3 scale = float3(0.4, 1.0, 0.8);

    //float e = 1e-5f;
    float e = 0.001;
    float3 p_du = p_t + e * tangentWS;
    float3 p_dv = p_t + e * bitangentWS;

    // Because we don't have higher-order fns, can't have a generic
    // 'apply displacement' function
    // Could have a macro #define.
    // Could have the caller provide the result of the 3 invocations of the
    // fn, but there's not much left after that point...
    float3 dp = Km * iq_fbm(scale * p_t, H, octaves) * normalWS;
    float3 np = p_t + dp;
    float3 du = Km * iq_fbm(scale * p_du, H, octaves) * normalWS;
    float3 dv = Km * iq_fbm(scale * p_dv, H, octaves) * normalWS;

    float3 dis_u = (e * tangentWS + du - dp) / e;
    float3 dis_v = (e * bitangentWS + dv - dp) / e;
    // Should be normal by construction...if I got things right
    // Problem: result is -ve, facing down - but fixing order/sign
    // gives garbage-looking results!
    float3 shadingNormalWS = normalize(cross(dis_v, dis_u));
    return shadingNormalWS;
}


// NOTE: world space origin and direction!
BRDF ProcMatWater(float3 positionWS, float3 normalWS, float elapsedTime) {
    // Have to copy-paste because _HasReflection etc are tags on the material in the non-proc version.
    // Can I refactor the shader to make it work better?
    BRDF surface;
    surface.baseColor = float4(1., 1., 1., 0.);
    
    surface.shadingNormalWS = ProcWaterNormal(positionWS, normalWS, elapsedTime);
    surface.geometricNormalWS = normalWS;

    // No refraction; ProcMatWater only used in skybox/miss shader, there's nothing underneath
    surface.flags8 = MC_BRDF_HAS_REFLECTION | MC_BRDF_RECEIVES_HAZE;
        
    surface.luminosity = 0;
    surface.diffuse = 0;
    surface.specular = 0;
    surface.roughness = 0;
    
    surface.reflectivity = 0.3;
    surface.ior = 1.33;
    surface.metallicity = 0;
    return surface;
}

// Unity URP non PBR functions
float3 LightingLambert(float3 lightColor, float3 lightDir, float3 normal)
{
    half NdotL = saturate(dot(normal, lightDir));
    return lightColor * NdotL;
}

float3 LightingSpecular(float3 lightColor, float3 lightDir, float3 normal, float3 viewDir, float smoothness)
{
    // From SampleSpecularSmoothness() in the URP SimpleLit shader
    // perceptual 0-1 -> exponent
    // PBR path instead has roughness being the square of the perceptual roughness.
    smoothness = exp2(10 * smoothness + 1);
    float3 halfVec = normalize(lightDir + viewDir);
    float NdotH = saturate(dot(normal, halfVec));
    float modifier = pow(NdotH, smoothness);
    return lightColor * modifier;
}

// Made-up value. But minimum of 0.01 is a bit high for testing and minimum of 0 can cause Too Many Rays
#define MC_MIN_PERCEPTUAL_DELTA 0.001f

// Estimate of the perceptual contribution based on the rgb contrib factor
float PerceptualDelta(float3 from, float3 to)
{
    // Weighted Euclidian
    // Values from https://stackoverflow.com/questions/6242114/how-much-worse-is-srgb-than-lab-when-computing-the-eucleidan-distance-between
    // comparable to REDMEAN, something like 8% within 'true' perceptual diff,
    // and cheap
    // also http://godsnotwheregodsnot.blogspot.com/2012/09/color-space-comparisons.html

    // I think my problem is the weighting makes the values tiny.
    
    // Scaling is correct:
    // Maximum is black to white, d = 1,1,1, weights sum to 1 => max is 1.
    // But makes most diffs tiny!
    // Scaling the weights by 100 is equiv to scaling the result by 10.
    // And that at least makes the numbers a bit more manageable.

    const float3 weights = {
      0.22216091748149788f,
      0.4288860259783791f,
      0.34895305654012304f
    };
    float3 d = from - to;
    return 10.0f * length(weights * d * d);
}

// From Ray Tracing Gems Ch 6. "A Fast and Robust Method for Avoiding Self-Intersection"
// When tracing a ray that starts at a point on a surface, this will return an appropriate
// start point for the ray.
// Normal n should point outward for rays exiting the surface, inwards for rays entering the surface.
// Normal n should be normalised.
float3 OffsetRay(const float3 p, float3 n)
{
    n = normalize(n);
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);
    
    // For optimisation this does number magic on the bitwise representation of the floating-point coordinates!
    float3 p_i = float3(
            asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
            asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
            asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    
    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                  abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                  abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

// Amount of reflection for cos(I) and IOR
inline float schlick(float cosine, float IOR)
{
    float r0 = (1.0f - IOR) / (1.0f + IOR);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

#endif  // MILLENNIUM_DEFINES_H
