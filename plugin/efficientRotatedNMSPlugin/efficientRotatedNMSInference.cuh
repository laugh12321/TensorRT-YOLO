/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_EFFICIENT_ROTATED_NMS_INFERENCE_CUH
#define TRT_EFFICIENT_ROTATED_NMS_INFERENCE_CUH

#include <cuda_fp16.h>

// FP32 Intrinsics

float __device__ __inline__ exp_mp(const float a)
{
    return __expf(a);
}
float __device__ __inline__ sigmoid_mp(const float a)
{
    return __frcp_rn(__fadd_rn(1.f, __expf(-a)));
}
float __device__ __inline__ add_mp(const float a, const float b)
{
    return __fadd_rn(a, b);
}
float __device__ __inline__ sub_mp(const float a, const float b)
{
    return __fsub_rn(a, b);
}
float __device__ __inline__ mul_mp(const float a, const float b)
{
    return __fmul_rn(a, b);
}
bool __device__ __inline__ gt_mp(const float a, const float b)
{
    return a > b;
}
bool __device__ __inline__ lt_mp(const float a, const float b)
{
    return a < b;
}
bool __device__ __inline__ lte_mp(const float a, const float b)
{
    return a <= b;
}
bool __device__ __inline__ gte_mp(const float a, const float b)
{
    return a >= b;
}

#if __CUDA_ARCH__ >= 530

// FP16 Intrinsics

__half __device__ __inline__ exp_mp(const __half a)
{
    return hexp(a);
}
__half __device__ __inline__ sigmoid_mp(const __half a)
{
    return hrcp(__hadd((__half) 1, hexp(__hneg(a))));
}
__half __device__ __inline__ add_mp(const __half a, const __half b)
{
    return __hadd(a, b);
}
__half __device__ __inline__ sub_mp(const __half a, const __half b)
{
    return __hsub(a, b);
}
__half __device__ __inline__ mul_mp(const __half a, const __half b)
{
    return __hmul(a, b);
}
bool __device__ __inline__ gt_mp(const __half a, const __half b)
{
    return __hgt(a, b);
}
bool __device__ __inline__ lt_mp(const __half a, const __half b)
{
    return __hlt(a, b);
}
bool __device__ __inline__ lte_mp(const __half a, const __half b)
{
    return __hle(a, b);
}
bool __device__ __inline__ gte_mp(const __half a, const __half b)
{
    return __hge(a, b);
}

#else

// FP16 Fallbacks on older architectures that lack support

__half __device__ __inline__ exp_mp(const __half a)
{
    return __float2half(exp_mp(__half2float(a)));
}
__half __device__ __inline__ sigmoid_mp(const __half a)
{
    return __float2half(sigmoid_mp(__half2float(a)));
}
__half __device__ __inline__ add_mp(const __half a, const __half b)
{
    return __float2half(add_mp(__half2float(a), __half2float(b)));
}
__half __device__ __inline__ sub_mp(const __half a, const __half b)
{
    return __float2half(sub_mp(__half2float(a), __half2float(b)));
}
__half __device__ __inline__ mul_mp(const __half a, const __half b)
{
    return __float2half(mul_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ gt_mp(const __half a, const __half b)
{
    return __float2half(gt_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ lt_mp(const __half a, const __half b)
{
    return __float2half(lt_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ lte_mp(const __half a, const __half b)
{
    return __float2half(lte_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ gte_mp(const __half a, const __half b)
{
    return __float2half(gte_mp(__half2float(a), __half2float(b)));
}

#endif

template <typename T>
struct __align__(1 * sizeof(T)) RotatedBoxCorner;

template <typename T>
struct __align__(1 * sizeof(T)) RotatedBoxCenterSize;

struct CovarianceMatrix{
    float a, b, c;
};

template <typename T>
__device__ __inline__ void get_covariance_matrix(const RotatedBoxCenterSize<T>& box, CovarianceMatrix &matrix) {
    float w = float(box.w);
    float h = float(box.h);
    float r = float(box.r);

    float a = w * w * 0.08333333333333333f;
    float b = h * h * 0.08333333333333333f;

    float cos = __cosf(r);
    float sin = __sinf(r);

    float cos2 = cos * cos;
    float sin2 = sin * sin;

    matrix.a = a * cos2 + b * sin2;
    matrix.b = a * sin2 + b * cos2;
    matrix.c = (a - b) * cos * sin;
}

template <typename T>
struct __align__(1 * sizeof(T)) RotatedBoxCorner
{
    // For NMS/IOU purposes, YXYX coding is identical to XYXY
    T y1, x1, y2, x2, r;

    __device__ void reorder()
    {
        if (gt_mp(y1, y2))
        {
            // Swap values, so y1 < y2
            y1 = sub_mp(y1, y2);
            y2 = add_mp(y1, y2);
            y1 = sub_mp(y2, y1);
        }
        if (gt_mp(x1, x2))
        {
            // Swap values, so x1 < x2
            x1 = sub_mp(x1, x2);
            x2 = add_mp(x1, x2);
            x1 = sub_mp(x2, x1);
        }
    }

    __device__ RotatedBoxCorner<T> clip(T low, T high) const
    {
        return {lt_mp(y1, low) ? low : (gt_mp(y1, high) ? high : y1),
                lt_mp(x1, low) ? low : (gt_mp(x1, high) ? high : x1), 
                lt_mp(y2, low) ? low : (gt_mp(y2, high) ? high : y2),
                lt_mp(x2, low) ? low : (gt_mp(x2, high) ? high : x2),
                r};
    }

    __device__ RotatedBoxCorner<T> decode(RotatedBoxCorner<T> anchor) const
    {
        return {add_mp(y1, anchor.y1), add_mp(x1, anchor.x1), add_mp(y2, anchor.y2), add_mp(x2, anchor.x2), r};
    }

    __device__ float area() const
    {
        T w = sub_mp(x2, x1);
        T h = sub_mp(y2, y1);
        if (lte_mp(h, (T) 0))
        {
            return 0;
        }
        if (lte_mp(w, (T) 0))
        {
            return 0;
        }
        return (float) h * (float) w;
    }

    __device__ operator RotatedBoxCenterSize<T>() const
    {
        T w = sub_mp(x2, x1);
        T h = sub_mp(y2, y1);
        return RotatedBoxCenterSize<T>{add_mp(y1, mul_mp((T) 0.5, h)), add_mp(x1, mul_mp((T) 0.5, w)), h, w, r};
    }

    // Calculate probabilistic IoU between oriented bounding boxes.
    // Implements the algorithm from https://arxiv.org/pdf/2106.06072v1.pdf.
    __device__ static float probiou(RotatedBoxCorner<T> a, RotatedBoxCorner<T> b)
    {
        RotatedBoxCenterSize<T> box1(a), box2(b);

        return RotatedBoxCenterSize<T>::probiou(box1, box2);
    }
};

template <typename T>
struct __align__(1 * sizeof(T)) RotatedBoxCenterSize
{
    // For NMS/IOU purposes, YXHW coding is identical to XYWH
    T y, x, h, w, r;

    __device__ void reorder() {}

    __device__ RotatedBoxCenterSize<T> clip(T low, T high) const
    {
        return RotatedBoxCenterSize<T>(RotatedBoxCorner<T>(*this).clip(low, high));
    }

    __device__ RotatedBoxCenterSize<T> decode(RotatedBoxCenterSize<T> anchor) const
    {
        return {add_mp(mul_mp(y, anchor.h), anchor.y), add_mp(mul_mp(x, anchor.w), anchor.x),
            mul_mp(anchor.h, exp_mp(h)), mul_mp(anchor.w, exp_mp(w)), r};
    }

    __device__ float area() const
    {
        if (h <= (T) 0)
        {
            return 0;
        }
        if (w <= (T) 0)
        {
            return 0;
        }
        return (float) h * (float) w;
    }

    __device__ operator RotatedBoxCorner<T>() const
    {
        T h2 = mul_mp(h, (T) 0.5);
        T w2 = mul_mp(w, (T) 0.5);
        return RotatedBoxCorner<T>{sub_mp(y, h2), sub_mp(x, w2), add_mp(y, h2), add_mp(x, w2), r};
    }

    // Calculate probabilistic IoU between oriented bounding boxes.
    // Implements the algorithm from https://arxiv.org/pdf/2106.06072v1.pdf.
    __device__ static float probiou(RotatedBoxCenterSize < T > & a, RotatedBoxCenterSize < T > & b) {
        CovarianceMatrix matrix1, matrix2;

        get_covariance_matrix < T > (a, matrix1);
        get_covariance_matrix < T > (b, matrix2);

        float add_a1_a2 = matrix1.a + matrix2.a;
        float add_b1_b2 = matrix1.b + matrix2.b;
        float add_c1_c2 = matrix1.c + matrix2.c;
        float sub_x1_x2 = a.x - b.x;
        float sub_y1_y2 = a.y - b.y;
        float sub_data = (add_a1_a2 * add_b1_b2) - (add_c1_c2 * add_c1_c2);
        sub_data = fmaxf(sub_data, 1e-7f);

        float t1 = 0.25f * (
            (add_a1_a2 * sub_y1_y2 * sub_y1_y2) +
            (add_b1_b2 * sub_x1_x2 * sub_x1_x2)) / sub_data;

        float t2 = 0.25f *
            (add_c1_c2 * sub_x1_x2 * sub_y1_y2) / sub_data;

        float t3 = 0.5f *
            logf((sub_data / (4.0f * (fmaxf(0.0f, matrix1.a * matrix1.b - matrix1.c * matrix1.c) *
                fmaxf(0.0f, matrix2.a * matrix2.b - matrix2.c * matrix2.c))))) / sub_data;

        float bd = fmaxf(1e-7f, fminf(t1 + t2 + t3, 100.0f));
        float hd = sqrtf(1.0f - expf(-bd));
        return 1.0f - hd;
    }
};

#endif
