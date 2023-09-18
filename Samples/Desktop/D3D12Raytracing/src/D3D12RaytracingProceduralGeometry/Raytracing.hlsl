//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#define HLSL
#include "RaytracingHlslCompat.h"
#include "ProceduralPrimitivesLibrary.hlsli"
#include "RaytracingShaderHelper.hlsli"
#include "MillenniumDefines.hlsli"

//***************************************************************************
//*****------ Shader resources bound via root signatures -------*************
//***************************************************************************

// Scene wide resources.
//  g_* - bound via a global root signature.
//  l_* - bound via a local root signature.
RaytracingAccelerationStructure g_scene : register(t0, space0);
RWTexture2D<float4> g_renderTarget : register(u0);
ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);

// Triangle resources
ByteAddressBuffer g_indices : register(t1, space0);
StructuredBuffer<Vertex> g_vertices : register(t2, space0);

// Procedural geometry resources
StructuredBuffer<PrimitiveInstancePerFrameBuffer> g_AABBPrimitiveAttributes : register(t3, space0);
ConstantBuffer<PrimitiveConstantBuffer> l_materialCB : register(b1);
ConstantBuffer<PrimitiveInstanceConstantBuffer> l_aabbCB: register(b2);


//***************************************************************************
//****************------ Utility functions -------***************************
//***************************************************************************

// Diffuse lighting calculation.
float CalculateDiffuseCoefficient(in float3 hitPosition, in float3 incidentLightRay, in float3 normal)
{
    float fNDotL = saturate(dot(-incidentLightRay, normal));
    return fNDotL;
}

// Phong lighting specular component
float4 CalculateSpecularCoefficient(in float3 hitPosition, in float3 incidentLightRay, in float3 normal, in float specularPower)
{
    float3 reflectedLightRay = normalize(reflect(incidentLightRay, normal));
    return pow(saturate(dot(reflectedLightRay, normalize (-WorldRayDirection()))), specularPower);
}


// Phong lighting model = ambient + diffuse + specular components.
float4 CalculatePhongLighting(in float4 albedo, in float3 normal, in bool isInShadow, in float diffuseCoef = 1.0, in float specularCoef = 1.0, in float specularPower = 50)
{
    float3 hitPosition = HitWorldPosition();
    float3 lightPosition = g_sceneCB.lightPosition.xyz;
    float shadowFactor = isInShadow ? InShadowRadiance : 1.0;
    float3 incidentLightRay = normalize(hitPosition - lightPosition);

    // Diffuse component.
    float4 lightDiffuseColor = g_sceneCB.lightDiffuseColor;
    float Kd = CalculateDiffuseCoefficient(hitPosition, incidentLightRay, normal);
    float4 diffuseColor = shadowFactor * diffuseCoef * Kd * lightDiffuseColor * albedo;

    // Specular component.
    float4 specularColor = float4(0, 0, 0, 0);
    if (!isInShadow)
    {
        float4 lightSpecularColor = float4(1, 1, 1, 1);
        float4 Ks = CalculateSpecularCoefficient(hitPosition, incidentLightRay, normal, specularPower);
        specularColor = specularCoef * Ks * lightSpecularColor;
    }

    // Ambient component.
    // Fake AO: Darken faces with normal facing downwards/away from the sky a little bit.
    float4 ambientColor = g_sceneCB.lightAmbientColor;
    float4 ambientColorMin = g_sceneCB.lightAmbientColor - 0.1;
    float4 ambientColorMax = g_sceneCB.lightAmbientColor;
    float a = 1 - saturate(dot(normal, float3(0, -1, 0)));
    ambientColor = albedo * lerp(ambientColorMin, ambientColorMax, a);
    
    return ambientColor + diffuseColor + specularColor;
}

//***************************************************************************
//*****------ TraceRay wrappers for radiance and shadow rays. -------********
//***************************************************************************

// Trace a radiance ray into the scene and returns a shaded color.
float4 TraceRadianceRay(in Ray ray, in UINT remainingRayRecursionDepth, bool primary)
{
    if (remainingRayRecursionDepth == 0U)
    {
        return float4(0, 0, 0, 0);
    }

    // Set the ray's extents.
    RayDesc rayDesc;
    rayDesc.Origin = ray.origin;
    rayDesc.Direction = ray.direction;
    // Set TMin to a zero value to avoid aliasing artifacts along contact areas.
    // Note: make sure to enable face culling so as to avoid surface face fighting.
    rayDesc.TMin = 0;
    rayDesc.TMax = 10000;
    uint flags = 0U;
    if (primary) { flags += MC_RAYFLAG_PRIMARY; }
    RayPayload rayPayload = { float4(1, 1, 1, 1), remainingRayRecursionDepth - 1, -1.f, flags };
    TraceRay(g_scene,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        TraceRayParameters::InstanceMask,
        TraceRayParameters::HitGroup::Offset[RayType::Radiance],
        TraceRayParameters::HitGroup::GeometryStride,
        TraceRayParameters::MissShader::Offset[RayType::Radiance],
        rayDesc, rayPayload);

    return rayPayload.color;
}

// Trace a shadow ray and return true if it hits any geometry.
bool TraceShadowRayAndReportIfHit(in Ray ray, in UINT remainingRayRecursionDepth)
{
    if (remainingRayRecursionDepth == 0U)
    {
        return false;
    }

    // Set the ray's extents.
    RayDesc rayDesc;
    rayDesc.Origin = ray.origin;
    rayDesc.Direction = ray.direction;
    // Set TMin to a zero value to avoid aliasing artifcats along contact areas.
    // Note: make sure to enable back-face culling so as to avoid surface face fighting.
    rayDesc.TMin = 0;
    rayDesc.TMax = 10000;

    // Initialize shadow ray payload.
    // Set the initial value to true since closest and any hit shaders are skipped. 
    // Shadow miss shader, if called, will set it to false.
    ShadowRayPayload shadowPayload = { true };
    TraceRay(g_scene,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES
        | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
        | RAY_FLAG_FORCE_OPAQUE             // ~skip any hit shaders
        | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // ~skip closest hit shaders,
        TraceRayParameters::InstanceMask,
        TraceRayParameters::HitGroup::Offset[RayType::Shadow],
        TraceRayParameters::HitGroup::GeometryStride,
        TraceRayParameters::MissShader::Offset[RayType::Shadow],
        rayDesc, shadowPayload);

    return shadowPayload.hit;
}

// "The rest of the owl"
float4 ShadeSurface(float3 positionWS, BRDF surface, inout RayInfo rayInfo, float3 lightPos)
{
    float3 _AmbientLightColor = { 1, 1, 1 };
    // Everything is organised as a sum-of-products for the final result.
    // This allows us to use perceptual thresholds to decide whether to cast
    // further rays.

    float3 returnCol = { 0, 0, 0 };
    
    float3 fogCol = { 0, 0, 0 };
    // Bryce fog model has alpha for each channel
    float3 fogAlpha = { 0, 0, 0 };

    // we can calculate the haze *early* with float3 col and float3 alpha for the haze.
    CalcWorldHaze(rayInfo, fogCol, fogAlpha);
    
    const float3 fogMul = 1.0f - fogAlpha;

    /////////////////
    // Light: Fog emission
    returnCol += fogCol * fogAlpha;

    /////////////////
    // Light: Emission
    {
        // This version would makes thrust lighting not additive, looks weird
        // const float3 light_emitted_mul = fogMul * surface.baseColor.rgb * surface.baseColor.a;
        const float3 light_emitted_mul = fogMul * surface.baseColor.rgb;
        returnCol += light_emitted_mul * surface.luminosity;
    }

    /////////////////
    // Light: Ambient
    // emissive/ambient contribution
    {
        const float3 light_ambient_diffuse_mul = fogMul * surface.baseColor.rgb * surface.baseColor.a * surface.diffuse;
        returnCol += light_ambient_diffuse_mul * _AmbientLightColor.rgb;
    }
    
    
    // Consider inlining into both places that use this to reduce live state
    const float3 light_reflection_common_mul = fogMul * lerp(float3(1, 1, 1), surface.baseColor.rgb, surface.metallicity);
    
    /////////////////
    // Light: Specular, Diffuse
    // Lighting/shadow
    
    // Is the surface affected by lights at all?
    // TODO replace this check with full contrib cutoff test?
    if (surface.diffuse + surface.specular > 0) {
        const float3 light_diffuse_mul = fogMul * surface.baseColor.rgb * surface.baseColor.a * surface.diffuse;
        const float3 light_specular_mul = light_reflection_common_mul * surface.specular;

        // Directional Lights
        {
            const float3 lightDir = normalize(positionWS - lightPos);
            const float3 lightCol = float3(1,1,1);
            const float shadowIntensity = 1;

            // For each light, compute shadow ray

            const float3 viewDir = -rayInfo.directionWS;
            const float3 totalLight =
                light_diffuse_mul * LightingLambert(lightCol, lightDir, surface.shadingNormalWS)
                + light_specular_mul * LightingSpecular(lightCol, lightDir, surface.shadingNormalWS, viewDir, 1 - surface.roughness);
            
            // Make shadow ray for directional light.
            bool hit = false;

            float3 shadingHit = (1 - shadowIntensity) * totalLight;
            float3 shadingNoHit = totalLight;
            float3 shading = shadingNoHit;
            
            if (surface.flags8 & MC_BRDF_RECEIVES_SHADOW) {
                // Also consider check for rayInfo.flags & MC_RAYFLAG_PRIMARY which makes us always cast
                float cutoff = MC_MIN_PERCEPTUAL_DELTA;
                if ((rayInfo.flags & MC_RAYFLAG_PRIMARY) == 0) { cutoff = max(cutoff, 0.03); }
                if (PerceptualDelta(rayInfo.color * shadingHit, rayInfo.color * shadingNoHit) >= cutoff) {
                    Ray shadowRay = { OffsetRay(positionWS, surface.geometricNormalWS), lightDir };
                    hit = TraceShadowRayAndReportIfHit(shadowRay, rayInfo.remainingDepth);
                }
            }

            if (hit) { shading = shadingHit; }
            
            returnCol += shading;
        }
    }
    
    /////////////////
    // Reflections and Transmission
    if ((surface.flags8 & (MC_BRDF_HAS_REFLECTION | MC_BRDF_HAS_REFRACTION)) &&
        (surface.reflectivity > 0 || surface.baseColor.a < 1))
    {
        float3 outwardNormal;
        float ior; // used for refracted dir
        float cosine; // used for schlick approximation
        const float IdotN = dot(rayInfo.directionWS, surface.shadingNormalWS);
        if (-IdotN > 0.0f)
        {
            // Ray exiting the surface
            outwardNormal = surface.shadingNormalWS;
            ior = 1.0f / surface.ior;
            cosine = surface.ior * -IdotN;
        }
        else
        {
            // Ray entering the surface
            outwardNormal = -surface.shadingNormalWS;
            ior = surface.ior;
            cosine = IdotN;
        }

        float fresnelReflection = saturate(schlick(cosine, surface.ior));

        // As a stylistic choice, opaque objects have no fresnel reflection
        const float reflectivity = (1 - surface.baseColor.a) * fresnelReflection + surface.reflectivity;

        const float3 reflectedDir = reflect(rayInfo.directionWS, surface.shadingNormalWS);
        const float3 refractedDir = refract(rayInfo.directionWS, outwardNormal, ior);

        if (surface.flags8 & MC_BRDF_HAS_REFLECTION)
        {
            const float3 light_reflected_mul = light_reflection_common_mul * reflectivity;
            float3 reflection = fogCol;

            // Diff between reflecting pure white and pure black
            float cutoff = max(0.04, MC_MIN_PERCEPTUAL_DELTA);
            if (PerceptualDelta(rayInfo.color * light_reflected_mul, float3(0,0,0)) > cutoff)
            {
                // Trace a reflection ray.
                Ray reflectionRay = { OffsetRay(positionWS, surface.geometricNormalWS), reflectedDir };
                reflection = TraceRadianceRay(reflectionRay, rayInfo.remainingDepth, false).rgb;
            }

            returnCol += light_reflected_mul * reflection;

        }

        // Trace refraction
        if (surface.flags8 & MC_BRDF_HAS_REFRACTION)
        {
            float3 light_transmitted_mul = fogMul * (1 - surface.baseColor.a);

            // Diff between reflecting pure white and pure black
            float cutoff = max(0.125, MC_MIN_PERCEPTUAL_DELTA);
            // Diff between transmitting pure white and pure black
            if (PerceptualDelta(rayInfo.color * light_transmitted_mul, float3(0,0,0)) > cutoff)
            {
                Ray refractionRay = { OffsetRay(positionWS, surface.geometricNormalWS), refractedDir };
                float3 light_transmitted = TraceRadianceRay(refractionRay, rayInfo.remainingDepth, false).rgb;
                returnCol += light_transmitted_mul * light_transmitted;
            }
            else
            {
                // Contribution below perceptual threshold.
                // Maybe run some simpler shading here.
            }
        }   
    }
    
    // TODO modify the alpha value based on uhhh?? transmitance? if necessary
    return float4(returnCol, surface.baseColor.a);
}

float4 SkyBox(inout RayInfo rayInfo, float elapsedTime)
{
    float3 originWS = rayInfo.originWS;
    float3 directionWS = rayInfo.directionWS;
    
    const float waterY = -0.05f;

    // t where ray intersects water surface
    float t = (directionWS.y == 0) 
        ? -1
        : (waterY - originWS.y) / directionWS.y;

    // Elsewhere, we say "if we run out of recursions, let it go to the miss shader"
    // (calling TraceRay() but telling it to ignore triangles).
    // But the miss shader contains the water with reflections, so we need to handle that here
    // to ensure we don't recurse further!

    if (t < rayInfo.tMin || rayInfo.remainingDepth <= 1)
    {
        // Clouds/Sky
        rayInfo.t = 1.#INF; // value for haze calc

        float3 skyCol = SkyColor(originWS, directionWS, elapsedTime);

        return float4(ApplyWorldHaze(rayInfo, skyCol), 1.f);
    }
    else
    {
        // Water
        rayInfo.t = t;
        
        float3 normalWS = { 0, 1, 0 };
        float3 positionWS = originWS + t * directionWS;
        positionWS.y = waterY;

        BRDF surface = ProcMatWater(positionWS, normalWS, elapsedTime);
        return ShadeSurface(positionWS, surface, rayInfo, g_sceneCB.lightPosition.xyz);
    }
}

//***************************************************************************
//********************------ Ray gen shader.. -------************************
//***************************************************************************

[shader("raygeneration")]
void MyRaygenShader()
{
    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
    Ray ray = GenerateCameraRay(DispatchRaysIndex().xy, g_sceneCB.cameraPosition.xyz, g_sceneCB.projectionToWorld);
 
    // Cast a ray into the scene and retrieve a shaded color.
    UINT remainingRecursionDepth = MAX_RAY_RECURSION_DEPTH;
    float4 color = TraceRadianceRay(ray, remainingRecursionDepth, true);

    // Write the raytraced color to the output texture.
    g_renderTarget[DispatchRaysIndex().xy] = color;
}

//***************************************************************************
//******************------ Closest hit shaders -------***********************
//***************************************************************************

[shader("closesthit")]
void MyClosestHitShader_Triangle(inout RayPayload rayPayload, in BuiltInTriangleIntersectionAttributes attr)
{
    // Get the base index of the triangle's first 16 bit index.
    uint indexSizeInBytes = 2;
    uint indicesPerTriangle = 3;
    uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;
    uint baseIndex = PrimitiveIndex() * triangleIndexStride;

    // Load up three 16 bit indices for the triangle.
    const uint3 indices = Load3x16BitIndices(baseIndex, g_indices);

    // Retrieve corresponding vertex normals for the triangle vertices.
    float3 triangleNormal = g_vertices[indices[0]].normal;

    // PERFORMANCE TIP: it is recommended to avoid values carry over across TraceRay() calls. 
    // Therefore, in cases like retrieving HitWorldPosition(), it is recomputed every time.

    // Shadow component.
    // Trace a shadow ray.
    float3 hitPosition = HitWorldPosition();
    Ray shadowRay = { hitPosition, normalize(g_sceneCB.lightPosition.xyz - hitPosition) };
    bool shadowRayHit = TraceShadowRayAndReportIfHit(shadowRay, rayPayload.remainingDepth);

    float checkers = AnalyticalCheckersTexture(HitWorldPosition(), triangleNormal, g_sceneCB.cameraPosition.xyz, g_sceneCB.projectionToWorld);

    // Reflected component.
    float4 reflectedColor = float4(0, 0, 0, 0);
    if (l_materialCB.reflectanceCoef > 0.001 )
    {
        // Trace a reflection ray.
        Ray reflectionRay = { HitWorldPosition(), reflect(WorldRayDirection(), triangleNormal) };
        float4 reflectionColor = TraceRadianceRay(reflectionRay, rayPayload.remainingDepth, false);

        float3 fresnelR = FresnelReflectanceSchlick(WorldRayDirection(), triangleNormal, l_materialCB.albedo.xyz);
        reflectedColor = l_materialCB.reflectanceCoef * float4(fresnelR, 1) * reflectionColor;
    }

    // Calculate final color.
    float4 phongColor = CalculatePhongLighting(l_materialCB.albedo, triangleNormal, shadowRayHit, l_materialCB.diffuseCoef, l_materialCB.specularCoef, l_materialCB.specularPower);
    float4 color = checkers * (phongColor + reflectedColor);

    rayPayload.t = RayTCurrent();
    RayInfo rayInfo = RayInfoFromRayIntersection(rayPayload);

    rayPayload.color = float4(ApplyWorldHaze(rayInfo, color.rgb), 1);
}

[shader("closesthit")]
void MyClosestHitShader_AABB(inout RayPayload rayPayload, in ProceduralPrimitiveAttributes attr)
{
    // PERFORMANCE TIP: it is recommended to minimize values carry over across TraceRay() calls. 
    // Therefore, in cases like retrieving HitWorldPosition(), it is recomputed every time.

    // Shadow component.
    // Trace a shadow ray.
    float3 hitPosition = HitWorldPosition();
    Ray shadowRay = { hitPosition, normalize(g_sceneCB.lightPosition.xyz - hitPosition) };
    bool shadowRayHit = TraceShadowRayAndReportIfHit(shadowRay, rayPayload.remainingDepth);

    // Reflected component.
    float4 reflectedColor = float4(0, 0, 0, 0);
    if (l_materialCB.reflectanceCoef > 0.001)
    {
        // Trace a reflection ray.
        Ray reflectionRay = { HitWorldPosition(), reflect(WorldRayDirection(), attr.normal) };
        float4 reflectionColor = TraceRadianceRay(reflectionRay, rayPayload.remainingDepth, false);

        float3 fresnelR = FresnelReflectanceSchlick(WorldRayDirection(), attr.normal, l_materialCB.albedo.xyz);
        reflectedColor = l_materialCB.reflectanceCoef * float4(fresnelR, 1) * reflectionColor;
    }

    // Calculate final color.
    float4 phongColor = CalculatePhongLighting(l_materialCB.albedo, attr.normal, shadowRayHit, l_materialCB.diffuseCoef, l_materialCB.specularCoef, l_materialCB.specularPower);
    float4 color = phongColor + reflectedColor;

    rayPayload.t = RayTCurrent();
    RayInfo rayInfo = RayInfoFromRayIntersection(rayPayload);

    rayPayload.color = float4(ApplyWorldHaze(rayInfo, color.rgb), 1);
}

//***************************************************************************
//**********************------ Miss shaders -------**************************
//***************************************************************************

[shader("miss")]
void MyMissShader(inout RayPayload rayPayload)
{
    RayInfo rayInfo = RayInfoFromRayIntersection(rayPayload);
    rayPayload.color = SkyBox(rayInfo, g_sceneCB.elapsedTime);
}

[shader("miss")]
void MyMissShader_ShadowRay(inout ShadowRayPayload rayPayload)
{
    rayPayload.hit = false;
}

//***************************************************************************
//*****************------ Intersection shaders-------************************
//***************************************************************************

// Get ray in AABB's local space.
Ray GetRayInAABBPrimitiveLocalSpace()
{
    PrimitiveInstancePerFrameBuffer attr = g_AABBPrimitiveAttributes[l_aabbCB.instanceIndex];

    // Retrieve a ray origin position and direction in bottom level AS space 
    // and transform them into the AABB primitive's local space.
    Ray ray;
    ray.origin = mul(float4(ObjectRayOrigin(), 1), attr.bottomLevelASToLocalSpace).xyz;
    ray.direction = mul(ObjectRayDirection(), (float3x3) attr.bottomLevelASToLocalSpace);
    return ray;
}

[shader("intersection")]
void MyIntersectionShader_AnalyticPrimitive()
{
    Ray localRay = GetRayInAABBPrimitiveLocalSpace();
    AnalyticPrimitive::Enum primitiveType = (AnalyticPrimitive::Enum) l_aabbCB.primitiveType;

    float thit;
    ProceduralPrimitiveAttributes attr = (ProceduralPrimitiveAttributes)0;
    if (RayAnalyticGeometryIntersectionTest(localRay, primitiveType, thit, attr))
    {
        PrimitiveInstancePerFrameBuffer aabbAttribute = g_AABBPrimitiveAttributes[l_aabbCB.instanceIndex];
        attr.normal = mul(attr.normal, (float3x3) aabbAttribute.localSpaceToBottomLevelAS);
        attr.normal = normalize(mul((float3x3) ObjectToWorld3x4(), attr.normal));

        ReportHit(thit, /*hitKind*/ 0, attr);
    }
}

[shader("intersection")]
void MyIntersectionShader_VolumetricPrimitive()
{
    Ray localRay = GetRayInAABBPrimitiveLocalSpace();
    VolumetricPrimitive::Enum primitiveType = (VolumetricPrimitive::Enum) l_aabbCB.primitiveType;
    
    float thit;
    ProceduralPrimitiveAttributes attr = (ProceduralPrimitiveAttributes)0;
    if (RayVolumetricGeometryIntersectionTest(localRay, primitiveType, thit, attr, g_sceneCB.elapsedTime))
    {
        PrimitiveInstancePerFrameBuffer aabbAttribute = g_AABBPrimitiveAttributes[l_aabbCB.instanceIndex];
        attr.normal = mul(attr.normal, (float3x3) aabbAttribute.localSpaceToBottomLevelAS);
        attr.normal = normalize(mul((float3x3) ObjectToWorld3x4(), attr.normal));

        ReportHit(thit, /*hitKind*/ 0, attr);
    }
}

[shader("intersection")]
void MyIntersectionShader_SignedDistancePrimitive()
{
    Ray localRay = GetRayInAABBPrimitiveLocalSpace();
    SignedDistancePrimitive::Enum primitiveType = (SignedDistancePrimitive::Enum) l_aabbCB.primitiveType;

    float thit;
    ProceduralPrimitiveAttributes attr = (ProceduralPrimitiveAttributes)0;
    if (RaySignedDistancePrimitiveTest(localRay, primitiveType, thit, attr, l_materialCB.stepScale))
    {
        PrimitiveInstancePerFrameBuffer aabbAttribute = g_AABBPrimitiveAttributes[l_aabbCB.instanceIndex];
        attr.normal = mul(attr.normal, (float3x3) aabbAttribute.localSpaceToBottomLevelAS);
        attr.normal = normalize(mul((float3x3) ObjectToWorld3x4(), attr.normal));
        
        ReportHit(thit, /*hitKind*/ 0, attr);
    }
}

#endif // RAYTRACING_HLSL