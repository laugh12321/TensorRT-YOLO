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
float __device__ __inline__ abs_mp(const float a)
{
    return fabsf(a);
}
float __device__ __inline__ cos_mp(const float a)
{
    return __cosf(a);
}
float __device__ __inline__ sin_mp(const float a)
{
    return __sinf(a);
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
__half __device__ __inline__ abs_mp(const __half a)
{
    return __habs(a);
}
__half __device__ __inline__ cos_mp(const __half a)
{
    return hcos(a);
}
__half __device__ __inline__ sin_mp(const __half a)
{
    return hsin(a);
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
__half __device__ __inline__ abs_mp(const __half a)
{
    return __float2half(fabsf(__half2float(a)));
}
__half __device__ __inline__ cos_mp(const __half a)
{
    return __float2half(cos_mp(__half2float(a)));
}
__half __device__ __inline__ sin_mp(const __half a)
{
    return __float2half(sin_mp(__half2float(a)));
}

#endif

template <typename T>
struct __align__(1 * sizeof(T)) RotatedBoxCorner;

template <typename T>
struct __align__(1 * sizeof(T)) RotatedBoxCenterSize;

// modified from
// https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/csrc/mmdeploy/backend_ops/tensorrt/common_impl/nms/allClassRotatedNMS.cu
template <typename T>
struct Point 
{
    T x, y;
    __device__ __inline__ Point(const T &px = 0, const T &py = 0) : x(px), y(py) {}
    __device__ __inline__ Point operator+(const Point &p) const 
    {
        return Point(add_mp(x, p.x), add_mp(y, p.y));
    }
    __device__ __inline__ Point &operator+=(const Point &p) 
    {
        x = add_mp(x, p.x);
        y = add_mp(y, p.y);
        return *this;
    }
    __device__ __inline__ Point operator-(const Point &p) const 
    {
        return Point(sub_mp(x, p.x), sub_mp(y, p.y));
    }
    __device__ __inline__ Point operator*(const T coeff) const 
    {
        return Point(mul_mp(x, coeff), mul_mp(y, coeff));
    }
};

// modified from
// https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/csrc/mmdeploy/backend_ops/tensorrt/common_impl/nms/allClassRotatedNMS.cu
template <typename T>
__device__ __inline__ T dot_2d(const Point<T> &A, const Point<T> &B) 
{
    return add_mp(mul_mp(A.x, B.x), mul_mp(A.y, B.y));
}

// modified from
// https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/csrc/mmdeploy/backend_ops/tensorrt/common_impl/nms/allClassRotatedNMS.cu
template <typename T>
__device__ __inline__ T cross_2d(const Point<T> &A, const Point<T> &B) 
{
    return sub_mp(mul_mp(A.x, B.y), mul_mp(B.x, A.y));
}

// modified from
// https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/csrc/mmdeploy/backend_ops/tensorrt/common_impl/nms/allClassRotatedNMS.cu
template <typename T>
__device__ __inline__ void get_rotated_vertices(const RotatedBoxCenterSize<T> &box, Point<T> (&pts)[4]) 
{
    // M_PI / 180. == 0.01745329251
    // double theta = box.a * 0.01745329251;
    // MODIFIED
    T cosTheta2 = mul_mp(cos_mp(box.r), T(0.5f));
    T sinTheta2 = mul_mp(sin_mp(box.r), T(0.5f));

    // y: top --> down; x: left --> right
    pts[0].x = sub_mp(box.x, add_mp(mul_mp(sinTheta2, box.h), mul_mp(cosTheta2, box.w)));
    pts[0].y = add_mp(box.y, sub_mp(mul_mp(cosTheta2, box.h), mul_mp(sinTheta2, box.w)));
    pts[1].x = add_mp(box.x, sub_mp(mul_mp(sinTheta2, box.h), mul_mp(cosTheta2, box.w)));
    pts[1].y = sub_mp(box.y, add_mp(mul_mp(cosTheta2, box.h), mul_mp(sinTheta2, box.w)));
    pts[2].x = sub_mp(mul_mp(T(2), box.x), pts[0].x);
    pts[2].y = sub_mp(mul_mp(T(2), box.y), pts[0].y);
    pts[3].x = sub_mp(mul_mp(T(2), box.x), pts[1].x);
    pts[3].y = sub_mp(mul_mp(T(2), box.y), pts[1].y);
}

// modified from
// https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/csrc/mmdeploy/backend_ops/tensorrt/common_impl/nms/allClassRotatedNMS.cu
template <typename T>
__device__ __inline__ int get_intersection_points(const Point<T> (&pts1)[4], const Point<T> (&pts2)[4], Point<T> (&intersections)[24]) 
{
    // Line vector
    // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    Point<T> vec1[4], vec2[4];
    #pragma unroll 4
    for (int i = 0; i < 4; i++) 
    {
        vec1[i] = pts1[(i + 1) % 4] - pts1[i];
        vec2[i] = pts2[(i + 1) % 4] - pts2[i];
    }

    // Line test - test all line combos for intersection
    int num = 0;  // number of intersections
    #pragma unroll 4
    for (int i = 0; i < 4; i++) 
    {
        #pragma unroll 4
        for (int j = 0; j < 4; j++) 
        {
            // Solve for 2x2 Ax=b
            T det = cross_2d<T>(vec2[j], vec1[i]);

            // This takes care of parallel lines
            if (lte_mp(abs_mp(det), T(1e-14))) 
            {
                continue;
            }

            auto vec12 = pts2[j] - pts1[i];

            T t1 = cross_2d<T>(vec2[j], vec12) / det;
            T t2 = cross_2d<T>(vec1[i], vec12) / det;

            if (gte_mp(t1, T(0.0)) && lte_mp(t1, T(1.0)) && gte_mp(t2, T(0.0)) && lte_mp(t2, T(1.0))) 
            {
                intersections[num++] = pts1[i] + vec1[i] * t1;
            }
        }
    }

    // Check for vertices of rect1 inside rect2
    {
        const auto &AB = vec2[0];
        const auto &DA = vec2[3];
        auto ABdotAB = dot_2d<T>(AB, AB);
        auto ADdotAD = dot_2d<T>(DA, DA);
        #pragma unroll 4
        for (int i = 0; i < 4; i++) 
        {
            // assume ABCD is the rectangle, and P is the point to be judged
            // P is inside ABCD iff. P's projection on AB lies within AB
            // and P's projection on AD lies within AD

            auto AP = pts1[i] - pts2[0];

            auto APdotAB = dot_2d<T>(AP, AB);
            auto APdotAD = -dot_2d<T>(AP, DA);

            if (gte_mp(APdotAB, T(0.0)) && gte_mp(APdotAD, T(0.0)) && lte_mp(APdotAB, ABdotAB) && lte_mp(APdotAD, ADdotAD)) 
            {
                intersections[num++] = pts1[i];
            }
        }
    }

    // Reverse the check - check for vertices of rect2 inside rect1
    {
        const auto &AB = vec1[0];
        const auto &DA = vec1[3];
        auto ABdotAB = dot_2d<T>(AB, AB);
        auto ADdotAD = dot_2d<T>(DA, DA);
        #pragma unroll 4
        for (int i = 0; i < 4; i++) 
        {
            auto AP = pts2[i] - pts1[0];

            auto APdotAB = dot_2d<T>(AP, AB);
            auto APdotAD = -dot_2d<T>(AP, DA);

            if (gte_mp(APdotAB, T(0.0)) && gte_mp(APdotAD, T(0.0)) && lte_mp(APdotAB, ABdotAB) && lte_mp(APdotAD, ADdotAD)) 
            {
                intersections[num++] = pts2[i];
            }
        }
    }

    return num;
}

// modified from
// https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/csrc/mmdeploy/backend_ops/tensorrt/common_impl/nms/allClassRotatedNMS.cu
template <typename T>
__device__ __inline__ int convex_hull_graham(const Point<T> (&p)[24], const int &num_in, Point<T> (&q)[24], bool shift_to_zero = false) 
{
    assert(num_in >= 2);

    // Step 1:
    // Find point with minimum y
    // if more than 1 points have the same minimum y,
    // pick the one with the minimum x.
    int t = 0;
    for (int i = 1; i < num_in; i++) 
    {
        if (lt_mp(p[i].y, p[t].y) || (lte_mp(p[i].y, p[t].y) && lt_mp(p[i].x, p[t].x))) 
        {
            t = i;
        }
    }
    auto &start = p[t];  // starting point

    // Step 2:
    // Subtract starting point from every points (for sorting in the next step)
    for (int i = 0; i < num_in; i++) 
    {
        q[i] = p[i] - start;
    }

    // Swap the starting point to position 0
    auto tmp = q[0];
    q[0] = q[t];
    q[t] = tmp;

    // Step 3:
    // Sort point 1 ~ num_in according to their relative cross-product values
    // (essentially sorting according to angles)
    // If the angles are the same, sort according to their distance to origin
    T dist[24];
    for (int i = 0; i < num_in; i++) 
    {
        dist[i] = dot_2d<T>(q[i], q[i]);
    }

    for (int i = 1; i < num_in - 1; i++) 
    {
        for (int j = i + 1; j < num_in; j++) 
        {
            T crossProduct = cross_2d<T>(q[i], q[j]);
            if (lt_mp(crossProduct, T(-1e-6)) || (lt_mp(abs_mp(crossProduct), T(1e-6)) && gt_mp(dist[i], dist[j]))) 
            {
                auto q_tmp = q[i];
                q[i] = q[j];
                q[j] = q_tmp;
                auto dist_tmp = dist[i];
                dist[i] = dist[j];
                dist[j] = dist_tmp;
            }
        }
    }

    // Step 4:
    // Make sure there are at least 2 points (that don't overlap with each other)
    // in the stack
    int k;  // index of the non-overlapped second point
    for (k = 1; k < num_in; k++) 
    {
        if (gt_mp(dist[k], T(1e-8))) 
        {
            break;
        }
    }
    if (k == num_in) 
    {
        // We reach the end, which means the convex hull is just one point
        q[0] = p[t];
        return 1;
    }
    q[1] = q[k];
    int m = 2;  // 2 points in the stack
    // Step 5:
    // Finally we can start the scanning process.
    // When a non-convex relationship between the 3 points is found
    // (either concave shape or duplicated points),
    // we pop the previous point from the stack
    // until the 3-point relationship is convex again, or
    // until the stack only contains two points
    for (int i = k + 1; i < num_in; i++) 
    {
        while (m > 1 && gte_mp(cross_2d<T>(q[i] - q[m - 2], q[m - 1] - q[m - 2]), T(0))) 
        {
            m--;
        }
        q[m++] = q[i];
    }

    // Step 6 (Optional):
    // In general sense we need the original coordinates, so we
    // need to shift the points back (reverting Step 2)
    // But if we're only interested in getting the area/perimeter of the shape
    // We can simply return.
    if (!shift_to_zero) 
    {
        for (int i = 0; i < m; i++) 
        {
            q[i] += start;
        }
    }

    return m;
}

// modified from
// https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/csrc/mmdeploy/backend_ops/tensorrt/common_impl/nms/allClassRotatedNMS.cu
template <typename T>
__device__ __inline__ T polygon_area(const Point<T> (&q)[24], const int &m) 
{
    if (m <= 2) 
    {
        return 0;
    }

    T area = 0;
    for (int i = 1; i < m - 1; i++) 
    {
        area = add_mp(area, abs_mp(cross_2d<T>(q[i] - q[0], q[i + 1] - q[0])));
    }

    return mul_mp(area, T(0.5));
}

// modified from
// https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/csrc/mmdeploy/backend_ops/tensorrt/common_impl/nms/allClassRotatedNMS.cu
template <typename T>
__device__ __inline__ T rotated_boxes_intersection(const RotatedBoxCenterSize<T>& box1, const RotatedBoxCenterSize<T>& box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from rotated_rect_intersection_pts
  Point<T> intersectPts[24], orderedPts[24];

  Point<T> pts1[4];
  Point<T> pts2[4];
  get_rotated_vertices<T>(box1, pts1);
  get_rotated_vertices<T>(box2, pts2);

  int num = get_intersection_points<T>(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  // Convex Hull to order the intersection points in clockwise order and find
  // the contour area.
  int num_convex = convex_hull_graham<T>(intersectPts, num, orderedPts, true);
  return polygon_area<T>(orderedPts, num_convex);
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

    __device__ static float intersect_area(RotatedBoxCorner<T> a, RotatedBoxCorner<T> b)
    {
        RotatedBoxCenterSize<T> box1(a), box2(b);

        auto center_shift_x = mul_mp(add_mp(box1.x, box2.x), T(0.5));
        auto center_shift_y = mul_mp(add_mp(box1.y, box2.y), T(0.5));
        box1.x = sub_mp(box1.x, center_shift_x);
        box1.y = sub_mp(box1.y, center_shift_y);

        box2.x = sub_mp(box2.x, center_shift_x);
        box2.y = sub_mp(box2.y, center_shift_y);

        if (lt_mp(box1.area(), 1e-14) || lt_mp(box2.area(), 1e-14)) 
        {
            return 0.f;
        }

        const T intersection_area = rotated_boxes_intersection<T>(box1, box2);
        return float(intersection_area);
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
    __device__ static float intersect_area(RotatedBoxCenterSize<T> a, RotatedBoxCenterSize<T> b)
    {
        RotatedBoxCenterSize<T> box1(a), box2(b);

        auto center_shift_x = mul_mp(add_mp(box1.x, box2.x), T(0.5));
        auto center_shift_y = mul_mp(add_mp(box1.y, box2.y), T(0.5));
        box1.x = sub_mp(box1.x, center_shift_x);
        box1.y = sub_mp(box1.y, center_shift_y);

        box2.x = sub_mp(box2.x, center_shift_x);
        box2.y = sub_mp(box2.y, center_shift_y);

        if (lt_mp(box1.area(), 1e-14) || lt_mp(box2.area(), 1e-14)) 
        {
            return 0.f;
        }

        const T intersection_area = rotated_boxes_intersection<T>(box1, box2);
        return float(intersection_area);
    }
};

#endif
