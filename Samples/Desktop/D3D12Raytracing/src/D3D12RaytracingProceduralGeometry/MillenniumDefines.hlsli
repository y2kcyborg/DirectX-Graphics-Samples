#ifndef MILLENNIUM_DEFINES_H
#define MILLENNIUM_DEFINES_H

#include "RayTracingHlslCompat.h"

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
    float A = 0.00065;
    // Falloff
    float B = 0.013;
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

float3 ApplyWorldHaze(RayInfo rayInfo, float3 rayCol)
{
    float3 fogColor = float3(0.839, 0.945, 1);
    float3 fogAlpha = ExponentialHaze(rayInfo);
    
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
                    
float4 SkyBox(inout RayInfo rayInfo, float elapsedTime)
{
    float4 backgroundColor = float4(BackgroundColor);
    
    float3 originWS = rayInfo.originWS;
    float3 directionWS = rayInfo.directionWS;

    float3 skyCol = SkyColor(originWS, directionWS, elapsedTime);

    return float4(ApplyWorldHaze(rayInfo, skyCol), 1.f);
}

#endif  // MILLENNIUM_DEFINES_H
