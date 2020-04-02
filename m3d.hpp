#pragma once

#include <cmath>
#include <cstring>
#include <iostream>
#include <initializer_list>


#define ABS(x) (x > = 0) ? x : (-x)

namespace m3d
{
  struct matrix4;

  struct vector2
  {
    float x;
    float y;

    vector2() = default;
    vector2(float p, float q) : x(p), y(q){}

    vector2& operator+=(vector2 const &ot)
    {
      x += ot.x;
      y += ot.y;

      return *this;
    }
    vector2& operator-=(vector2 const &ot)
    {
      x -= ot.x;
      y -= ot.y;

      return *this;
    }
    vector2& operator*=(float val)
    {
      x *= val;
      y *= val;

      return *this;
    }
    vector2& operator/=(float val)
    {
      x /= val;
      y /= val;

      return *this;
    }
  };

  vector2 operator+(vector2 const &one, vector2 const &two){
    vector2 result(one);
    result += two;

    return result;
  }

  vector2 operator-(vector2 const &one, vector2 const &two){
    vector2 result(one);
    result -= two;

    return result;
  }

  vector2 operator*(vector2 const &one, float f){
    vector2 result(one);
    result *= f;

    return result;
  }

  vector2 operator*(float f, vector2 const &two){
    vector2 result(two);
    result *= f;

    return result;
  }

  vector2 operator/(vector2 const &one, float f){
    vector2 result(one);
    result /= f;

    return result;
  }

  vector2 operator/(float f, vector2 const &two){
    vector2 result(two);
    result /= f;

    return result;
  }

  float dot(vector2 const &one, vector2 const &two)
  {
    return (one.x*two.x + one.x*two.y);
  }

  vector2 normalize(vector2 const &v)
  {
    float norm = v.x*v.x + v.y*v.y;
    norm = sqrt(norm);

    return vector2(v.x / norm,  v.y / norm);
  }

  struct vector3
  {
    float x;
    float y;
    float z;

    vector3() = default;
    vector3(float p, float q) : x(p), y(q), z(0){}
    vector3(vector2 const &ot) : x(ot.x), y(ot.y), z(0){}
    vector3(float p, float q, float r) : x(p), y(q), z(r){}
    vector3(vector2 const &ot, float r) : x(ot.x), y(ot.y), z(r){}
    vector3(float p, vector2 const &ot) : x(p), y(ot.x), z(ot.y){}

    vector3& operator+=(vector3 const &ot)
    {
      x += ot.x;
      y += ot.y;
      z += ot.z;

      return *this;
    }
    vector3& operator-=(vector3 const &ot)
    {
      x -= ot.x;
      y -= ot.y;
      z -= ot.z;

      return *this;
    }
    vector3& operator*=(float s)
    {
      x *= s;
      y *= s;
      x *= s;

      return *this;
    }
    vector3& operator/=(float s)
    {
      x /= s;
      y /= s;
      x /= s;

      return *this;
    }

  } __attribute__ ((aligned(16)));

  vector3 operator+(vector3 const &one, vector3 const &two){
    vector3 result(one);
    result += two;

    return result;
  }

  vector3 operator-(vector3 const &one, vector3 const &two){
    vector3 result(one);
    result -= two;

    return result;
  }

  vector3 operator*(vector3 const &one, float f){
    vector3 result(one);
    result *= f;

    return result;
  }

  vector3 operator*(float f, vector3 const &two){
    vector3 result(two);
    result *= f;

    return result;
  }

  vector3 operator/(vector3 const &one, float f){
    vector3 result(one);
    result /= f;

    return result;
  }

  vector3 operator/(float f, vector3 const &two){
    vector3 result(two);
    result /= f;

    return result;
  }

  float dot(vector3 const &one, vector3 const &two)
  {
    return (one.x*two.x + one.y*two.y + one.z*two.z);
  }

  vector3 cross(vector3 const &one, vector3 const &two)
  {
    float i = one.y*two.z - one.z*two.y;
    float j = one.z*two.x - one.x*two.z;
    float k = one.x*two.y - one.y*two.x;

    return vector3(i, j, k);
  }

  vector3 normalize(vector3 const &v)
  {
    float norm = v.x*v.x + v.y*v.y  + v.z*v.z;
    norm = sqrt(norm);

    return vector3(v.x / norm, v.y / norm, v.z / norm);
  }

  struct vector4
  {
    float x;
    float y;
    float z;
    float w;

    vector4() = default;
    vector4(float f, float s, float t) : x(f), y(s), z(t), w(0){}
    vector4(vector2 const &f, float s) : x(f.x), y(f.y), z(s), w(0){}
    vector4(float f, vector2 const &s) : x(f), y(s.x), z(s.y), w(0){}
    vector4(vector3 const &ot) : x(ot.x), y(ot.y), z(ot.z), w(0){}
    vector4(vector2 const &ot, float r, float t) : x(ot.x), y(ot.y), z(r), w(t){}
    vector4(float p, vector2 const &ot, float t) : x(p), y(ot.x), z(ot.y), w(t){}
    vector4(float p, float q, vector2 const &ot) : x(p), y(q), z(ot.x), w(ot.y){}
    vector4(vector2 const &f, vector2 const &s) : x(f.x), y(f.y), z(s.x), w(s.y){}
    vector4(float p, float q, float r, float t) : x(p), y(q), z(r), w(t){}
    vector4(vector3 const &ot, float t) : x(ot.x), y(ot.y), z(ot.z), w(t){}
    vector4(float p, vector3 const &ot) : x(p), y(ot.x), z(ot.y), w(ot.z){}

    vector4& operator+=(vector4 const &ot)
    {
      x += ot.x;
      y += ot.y;
      z += ot.z;
      w += ot.w;

      return *this;
    }
    vector4& operator-=(vector4 const &ot)
    {
      x -= ot.x;
      y -= ot.y;
      z -= ot.z;
      w -= ot.w;

      return *this;
    }
    vector4& operator*=(float s)
    {
      x *= s;
      y *= s;
      x *= s;
      w *= s;

      return *this;
    }
    vector4& operator/=(float s)
    {
      x /= s;
      y /= s;
      x /= s;
      w /= s;

      return *this;
    }

  } __attribute__ ((aligned(16)));

  vector4 operator+(vector4 const &one, vector4 const &two){
    vector4 result(one);
    result += two;

    return result;
  }

  vector4 operator-(vector4 const &one, vector4 const &two){
    vector4 result(one);
    result -= two;

    return result;
  }

  vector4 operator*(vector4 const &one, float f){
    vector4 result(one);
    result *= f;

    return result;
  }

  vector4 operator*(float f, vector4 const &two){
    vector4 result(two);
    result *= f;

    return result;
  }

  vector4 operator/(vector4 const &one, float f){
    vector4 result(one);
    result /= f;

    return result;
  }

  vector4 operator/(float f, vector4 const &two){
    vector4 result(two);
    result /= f;

    return result;
  }

  float dot(vector4 const &one, vector4 const &two)
  {
    return (one.x*two.x + one.y*two.y + one.z*two.z + one.w*two.w);
  }

  vector4 normalize(vector4 const &v)
  {
    float norm = v.x*v.x + v.y*v.y  + v.z*v.z + v.w*v.w;
    norm = sqrt(norm);

    return vector4(v.x / norm, v.y / norm, v.z / norm, v.w / norm);
  }

  struct matrix2{

    matrix2() = default;
    matrix2(float val){

      memset(data, 0, sizeof(data));
      data[0][0] = val;
      data[1][1] = val;
    }
    matrix2(std::initializer_list<float> const &lst){

      auto it = lst.begin();
      for(int i = 0; i != 2; ++i)
      {
          for(int j = 0; j != 2; ++j)
          {
            data[i][j] = *it;
            ++it;
          }
      }
    }
    matrix2(vector2 const &f, vector2 const &s){

      vector2 vecs[] = {f, s};
      for(int i = 0; i != 2; ++i)
      {
        *(*(data + i) + 0) = vecs[i].x;
        *(*(data + i) + 1) = vecs[i].y;
      }
    }

    float const* operator[](int index) const{
      return *(data + index);
    }
    float* operator[](int index){
      return *(data + index);
    }

    matrix2& operator+=(matrix2 const &ot){

      float* mout = &data[0][0];
      const matrix2* min = &ot;

      __asm__ __volatile__(
        "mov %0, %%rdi;"
        "mov %1, %%rsi;"

        "movaps (%%rdi), %%xmm0;"
        "movaps (%%rsi), %%xmm1;"

        "addps %%xmm1, %%xmm0;"

        "movaps %%xmm0, (%%rdi)"

        : :"m"(mout), "m"(min) : "%rdi", "%rsi"
      );

      return *this;
    }

    matrix2& operator-=(matrix2 const &ot){

      float* mout = &data[0][0];
      const matrix2* min = &ot;

      __asm__ __volatile__(
        "mov %0, %%rdi;"
        "mov %1, %%rsi;"

        "movaps (%%rdi), %%xmm0;"
        "movaps (%%rsi), %%xmm1;"

        "subps %%xmm1, %%xmm0;"

        "movaps %%xmm0, (%%rdi)"

        : :"m"(mout), "m"(min) : "%rdi", "%rsi"
      );

      return *this;
    }

    matrix2& operator*=(float f){

      float* mout = &data[0][0];
      float* fin = &f;

      __asm__ __volatile__(
        "mov %0, %%rdi;"
        "mov %1, %%rsi;"

        "movaps (%%rdi), %%xmm0;"
        "movss (%%rsi), %%xmm1;"
        "shufps $0x00, %%xmm1, %%xmm1;"

        "mulps %%xmm1, %%xmm0;"

        "movaps %%xmm0, (%%rdi);"

        : :"m"(mout), "m"(fin) : "%rdi", "%rsi"
      );
      return *this;
    }

    matrix2& operator/=(float f){

      float* mout = &data[0][0];
      float* fin = &f;

      __asm__ __volatile__(
        "mov %0, %%rdi;"
        "mov %1, %%rsi;"

        "movaps (%%rdi), %%xmm0;"
        "movss (%%rsi), %%xmm1;"
        "shufps $0x00, %%xmm1, %%xmm1;"

        "divps %%xmm1, %%xmm0;"

        "movaps %%xmm0, (%%rdi);"

        : :"m"(mout), "m"(fin) : "%rdi", "%rsi"
      );

      return *this;
    }

    float data[2][2] __attribute__ ((aligned(16)));
  };

  matrix2 operator+(matrix2 const &one, matrix2 const &two){
    matrix2 mout(one);
    mout+=two;

    return mout;
  }

  matrix2 operator-(matrix2 const &one, matrix2 const &two){
    matrix2 mout(one);
    mout-=two;

    return mout;
  }

  matrix2 operator*(matrix2 const &one, float f){
    matrix2 mout(one);
    mout*=f;

    return mout;
  }

  matrix2 operator*(float f, matrix2 const &two){
    matrix2 mout(two);
    mout*=f;

    return mout;
  }

  matrix2 operator/(matrix2 const &one, float f){
    matrix2 mout(one);
    mout/=f;

    return mout;
  }

  matrix2 operator/(float f, matrix2 const &two){
    matrix2 mout(two);
    mout/=f;

    return mout;
  }

  vector2 operator*(matrix2 const &mat, vector2 const &vi)
  {
    vector2 vout;

    vector2* vo = &vout;
    const matrix2* min = &mat;
    const vector2* vin = &vi;

    __asm__ __volatile__(

      "mov %0, %%rdi;"
      "mov %1, %%rsi;"
      "mov %2, %%rdx;"

      "movaps (%%rsi), %%xmm0;"
      "movss (%%rdx), %%xmm1;"
      "movss 4(%%rdx), %%xmm2;"

      "shufps $0x00, %%xmm1, %%xmm2;"
      "shufps $0xD8, %%xmm0, %%xmm0;"

      "mulps %%xmm2, %%xmm0;"

      "movhlps %%xmm0, %%xmm1;"

      "addps %%xmm0, %%xmm1;"

      "movlps %%xmm1, (%%rdi);"

      : : "m"(vo), "m"(min), "m"(vin):
    );

    return vout;
  }

  matrix2& transpose(matrix2& mat)
  {
    matrix2 mout(mat);
    mat[1][0] = mout[0][1];
    mat[0][1] = mout[1][0];

    return mat;
  }

  float determinant(matrix2 const &ot)
  {
    return (ot[0][0] * ot[1][1]) - (ot[1][0] * ot[0][1]);
  }

  matrix2 inverse(matrix2& mat)
  {
    float det = determinant(mat);
    matrix2 mout(mat);
    mout[0][0] = mat[1][1]/det;
    mout[1][0] = -mat[0][1]/det;
    mout[0][1] = -mat[1][0]/det;
    mout[1][1] = mat[0][0]/det;
    return mout;
  }

  struct matrix3{

    matrix3() = default;
    matrix3(float val)
    {
      memset(data, 0, sizeof(data));
      data[0][0] = val; data[1][1] = val; data[2][2] = val;
    }
    matrix3(std::initializer_list<float> const &lst)
    {
      auto it = lst.begin();
      for(int i = 0; i != 3; ++i)
      {
          for(int j = 0; j != 3; ++j)
          {
            data[i][j] = *it;
            ++it;
          }
      }
    }
    matrix3(vector3 const &f, vector3 const &s, vector3 const &t)
    {
      vector3 vecs[] = {f, s, t};
      for(int i = 0; i != 3; ++i)
      {
        *(*(data + i) + 0) = vecs[i].x;
        *(*(data + i) + 1) = vecs[i].y;
        *(*(data + i) + 2) = vecs[i].z;
      }
    }
    matrix3(matrix4 const &);

    float const* operator[](int index) const
    {
      return *(data + index);
    }
    float* operator[](int index)
    {
      return *(data + index);
    }

    float data[3][3] __attribute__ ((aligned(16)));
  };

  matrix3 operator+(matrix3 const &one, matrix3 const &two){
    matrix3 mout(one);
    //mout+=two;

    return mout;
  }

  matrix3 operator-(matrix3 const &one, matrix3 const &two){
    matrix3 mout(one);
    //mout-=two;

    return mout;
  }

  matrix3 operator*(matrix3 const &one, float f){
    matrix3 mout(one);
    //mout*=f;

    return mout;
  }

  matrix3 operator*(float f, matrix3 const &two){
    matrix3 mout(two);
    //mout*=f;

    return mout;
  }

  matrix3 operator/(matrix3 const &one, float f){
    matrix3 mout(one);
    //mout/=f;

    return mout;
  }

  matrix3 operator/(float f, matrix3 const &two){
    matrix3 mout(two);
    //mout/=f;

    return mout;
  }

  vector3 operator*(matrix3 const &m, vector3 const& vin)
  {
    vector3 vout;
    const matrix3 *mat = &m;
    const vector3 *vi = &vin;
    vector3 *vo = &vout;
    __asm__ __volatile__
    (
      "mov %0, %%rdi;"
      "mov %1, %%rsi;"
      "mov %2, %%rdx;"

      "vzeroall;"

      "movss (%%rdx), %%xmm4;"
      "movss 4(%%rdx), %%xmm5;"
      "movss 8(%%rdx), %%xmm6;"
      "shufps $0x51, %%xmm5, %%xmm5;"
      "shufps $0x45, %%xmm6, %%xmm6;"
      "addps %%xmm6, %%xmm5;"
      "addps %%xmm5, %%xmm4;"

      "mov $3, %%ecx;"
      "xor %%rax, %%rax;"
      "MMUL:;"
      "movss (%%rsi, %%rax), %%xmm1;"
      "add $4, %%rax;"
      "movss (%%rsi, %%rax), %%xmm2;"
      "add $4, %%rax;"
      "movss (%%rsi, %%rax), %%xmm3;"
      "shufps $0x51, %%xmm2, %%xmm2;"
      "shufps $0x45, %%xmm3, %%xmm3;"
      "addps %%xmm3, %%xmm2;"
      "addps %%xmm2, %%xmm1;"
      "mulps %%xmm4, %%xmm1;"
      "add $4 , %%rax;"
      "addps %%xmm1, %%xmm0;"
      "sub $1, %%ecx;"
      "jnz MMUL;"

      "movaps %%xmm0, %%xmm1;"
      "movaps %%xmm0, %%xmm2;"
      "shufps $0x00, %%xmm0, %%xmm0;"
      "shufps $0x55, %%xmm1, %%xmm1;"
      "shufps $0xAA, %%xmm2, %%xmm2;"

      "movss %%xmm0, (%%rdi);"
      "movss %%xmm1, 4(%%rdi);"
      "movss %%xmm2, 8(%%rdi);"

      : : "m"(vo), "m"(mat), "m"(vi):
    );
    return vout;
  }

  matrix3& transpose(matrix3& mat)
  {
    matrix3 mout(mat);
    for(int i = 0 ; i != 3; ++i){
      for(int j = 0; j != 3; ++j){
        if(i != j){
          mat[j][i] = mout[i][j];
        }
      }
    }
    return mat;
  }

  float determinant(matrix3 const &ot)
  {
    float t1 = ot[0][0]*(ot[1][1]*ot[2][2] - ot[1][2]*ot[2][1]);
    float t2 = ot[1][0]*(ot[0][2]*ot[2][1] - ot[0][1]*ot[2][2]);
    float t3 = ot[2][0]*(ot[0][1]*ot[1][2] - ot[0][2]*ot[1][1]);
    return t1 + t2 + t3;
  }

  matrix3 inverse(matrix3& mat)
  {
    return matrix3();
  }

  struct matrix4{

    matrix4() = default;
    matrix4(float val)
    {
      memset(data, 0, sizeof(data));
      data[0][0] = val; data[1][1] = val; data[2][2] = val;
      data[3][3] = val;
    }
    matrix4(std::initializer_list<float> const &lst)
    {
      auto it = lst.begin();
      for(int i = 0; i != 4; ++i)
      {
          for(int j = 0; j != 4; ++j)
          {
            data[i][j] = *it;
            ++it;
          }
      }
    }
    matrix4(vector3 const &f, vector3 const &s, vector3 const &t)
    {
      vector3 vecs[] = {f, s, t};
      for(int i = 0; i != 3; ++i)
      {
        *(*(data + i) + 0) = vecs[i].x;
        *(*(data + i) + 1) = vecs[i].y;
        *(*(data + i) + 2) = vecs[i].z;
        *(*(data + i) + 3) = 0;
      }
      *(*(data + 3) + 0) = 0;
      *(*(data + 3) + 1) = 0;
      *(*(data + 3) + 2) = 0;
      *(*(data + 3) + 3) = 1;
    }
    matrix4(vector3 const &f, vector3 const &s, vector3 const &t, vector3 const &l)
    {
      vector3 vecs[] = {f, s, t, l};
      for(int i = 0; i != 3; ++i)
      {
        *(*(data + i) + 0) = vecs[i].x;
        *(*(data + i) + 1) = vecs[i].y;
        *(*(data + i) + 2) = vecs[i].z;
        *(*(data + i) + 3) = 0;
      }
      data[3][3] = 1;
    }
    matrix4(vector4 const &f, vector4 const &s, vector4 const &t, vector4 const &l)
    {
      vector4 vecs[] = {f, s, t, l};
      for(int i = 0; i != 4; ++i)
      {
        *(*(data + i) + 0) = vecs[i].x;
        *(*(data + i) + 1) = vecs[i].y;
        *(*(data + i) + 2) = vecs[i].z;
        *(*(data + i) + 3) = vecs[i].w;
      }
    }

    float const* operator[](int index) const
    {
      return *(data + index);
    }
    float* operator[](int index)
    {
      return *(data + index);
    }

    matrix4& operator+=(matrix4 const &ot){

      float* mout = &data[0][0];
      const matrix4* min = &ot;

      __asm__ __volatile__(
        "mov %0, %%rdi;"
        "mov %1, %%rsi;"

        "movaps 0x00(%%rdi), %%xmm0;"
        "movaps 0x10(%%rdi), %%xmm1;"
        "movaps 0x00(%%rdi), %%xmm2;"
        "movaps 0x10(%%rdi), %%xmm3;"

        "movaps 0x00(%%rsi), %%xmm4;"
        "movaps 0x10(%%rsi), %%xmm5;"
        "movaps 0x00(%%rsi), %%xmm6;"
        "movaps 0x10(%%rsi), %%xmm7;"

        "addps %%xmm4, %%xmm0;"
        "addps %%xmm5, %%xmm1;"
        "addps %%xmm6, %%xmm2;"
        "addps %%xmm7, %%xmm3;"

        "movaps %%xmm0, 0x00(%%rdi);"
        "movaps %%xmm1, 0x10(%%rdi);"
        "movaps %%xmm2, 0x20(%%rdi);"
        "movaps %%xmm3, 0x30(%%rdi);"

        : :"m"(mout), "m"(min) : "%rdi", "%rsi"
      );

      return *this;
    }

    matrix4& operator-=(matrix4 const &ot){

      float* mout = &data[0][0];
      const matrix4* min = &ot;

      __asm__ __volatile__(
        "mov %0, %%rdi;"
        "mov %1, %%rsi;"

        "movaps 0x00(%%rdi), %%xmm0;"
        "movaps 0x10(%%rdi), %%xmm1;"
        "movaps 0x00(%%rdi), %%xmm2;"
        "movaps 0x10(%%rdi), %%xmm3;"

        "movaps 0x00(%%rsi), %%xmm4;"
        "movaps 0x10(%%rsi), %%xmm5;"
        "movaps 0x00(%%rsi), %%xmm6;"
        "movaps 0x10(%%rsi), %%xmm7;"

        "subps %%xmm4, %%xmm0;"
        "subps %%xmm5, %%xmm1;"
        "subps %%xmm6, %%xmm2;"
        "subps %%xmm7, %%xmm3;"

        "movaps %%xmm0, 0x00(%%rdi);"
        "movaps %%xmm1, 0x10(%%rdi);"
        "movaps %%xmm2, 0x20(%%rdi);"
        "movaps %%xmm3, 0x30(%%rdi);"

        : :"m"(mout), "m"(min) : "%rdi", "%rsi"
      );

      return *this;
    }

    matrix4& operator*=(float f){

      float* mout = &data[0][0];
      float* fin = &f;

      __asm__ __volatile__(
        "mov %0, %%rdi;"
        "mov %1, %%rsi;"

        "movaps 0x00(%%rdi), %%xmm0;"
        "movaps 0x10(%%rdi), %%xmm1;"
        "movaps 0x20(%%rdi), %%xmm2;"
        "movaps 0x30(%%rdi), %%xmm3;"

        "movss 0x00(%%rsi), %%xmm4;"
        "shufps $0x00, %%xmm4, %%xmm4;"

        "mulps %%xmm4, %%xmm0;"
        "mulps %%xmm4, %%xmm1;"
        "mulps %%xmm4, %%xmm2;"
        "mulps %%xmm4, %%xmm3;"

        "movaps %%xmm0, 0x00(%%rdi);"
        "movaps %%xmm1, 0x10(%%rdi);"
        "movaps %%xmm2, 0x20(%%rdi);"
        "movaps %%xmm2, 0x30(%%rdi);"

        : :"m"(mout), "m"(fin) : "%rdi", "%rsi"
      );

      return *this;
    }

    matrix4& operator/=(float f){

      float* mout = &data[0][0];
      float* fin = &f;

      __asm__ __volatile__(
        "mov %0, %%rdi;"
        "mov %1, %%rsi;"

        "movaps 0x00(%%rdi), %%xmm0;"
        "movaps 0x10(%%rdi), %%xmm1;"
        "movaps 0x20(%%rdi), %%xmm2;"
        "movaps 0x30(%%rdi), %%xmm3;"

        "movss 0x00(%%rsi), %%xmm4;"
        "shufps $0x00, %%xmm4, %%xmm4;"

        "divps %%xmm4, %%xmm0;"
        "divps %%xmm4, %%xmm1;"
        "divps %%xmm4, %%xmm2;"
        "divps %%xmm4, %%xmm3;"

        "movaps %%xmm0, 0x00(%%rdi);"
        "movaps %%xmm1, 0x10(%%rdi);"
        "movaps %%xmm2, 0x20(%%rdi);"
        "movaps %%xmm2, 0x30(%%rdi);"

        : :"m"(mout), "m"(fin) : "%rdi", "%rsi"
      );

      return *this;
    }

    float data[4][4] __attribute__ ((aligned(16)));
  };

  matrix4 operator+(matrix4 const &one, matrix4 const &two){
    matrix4 mout(one);
    mout+=two;

    return mout;
  }

  matrix4 operator-(matrix4 const &one, matrix4 const &two){
    matrix4 mout(one);
    mout-=two;

    return mout;
  }

  matrix4 operator*(matrix4 const &one, float f){
    matrix4 mout(one);
    mout*=f;

    return mout;
  }

  matrix4 operator*(float f, matrix4 const &two){
    matrix4 mout(two);
    mout*=f;

    return mout;
  }

  matrix4 operator/(matrix4 const &one, float f){
    matrix4 mout(one);
    mout/=f;

    return mout;
  }

  matrix4 operator/(float f, matrix4 const &two){
    matrix4 mout(two);
    mout/=f;

    return mout;
  }

  vector4 operator*(matrix4 const &mat, vector4 const &vi)
  {
    vector4 v;
    const matrix4* m = &mat;
    const vector4* vin = &vi;
    vector4* vout = &v;
    __asm__ __volatile__
    (
      "mov %0, %%rdi;"
      "mov %1, %%rsi;"
      "mov %2, %%rdx;"

      "movaps (%%rsi), %%xmm4;"
      "movaps 16(%%rsi), %%xmm5;"
      "movaps 32(%%rsi), %%xmm6;"
      "movaps 48(%%rsi), %%xmm7;"

      "movaps (%%rdx), %%xmm0;"
      "movaps %%xmm0, %%xmm1;"
      "movaps %%xmm0, %%xmm2;"
      "movaps %%xmm0, %%xmm3;"

      "shufps $0x00, %%xmm0, %%xmm0;"
      "shufps $0x55, %%xmm1, %%xmm1;"
      "shufps $0xAA, %%xmm2, %%xmm2;"
      "shufps $0xFF, %%xmm3, %%xmm3;"

      "mulps %%xmm4, %%xmm0;"
      "mulps %%xmm5, %%xmm1;"
      "mulps %%xmm6, %%xmm2;"
      "mulps %%xmm7, %%xmm3;"

      "addps %%xmm3, %%xmm2;"
      "addps %%xmm2, %%xmm1;"
      "addps %%xmm1, %%xmm0;"

      "movaps %%xmm0, (%%rdi);"
      : : "m"(vout), "m"(m), "m"(vin) : "%rdi", "%rsi", "%rdx", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
    );
    return v;
  }

  matrix4& transpose(matrix4& mat)
  {
    matrix4* m = &mat;
    __asm__ __volatile__
    (
      "mov %0, %%rdi;"
      "movaps 00(%%rdi), %%xmm0;"
      "movaps 16(%%rdi), %%xmm1;"
      "movaps 32(%%rdi), %%xmm2;"
      "movaps 48(%%rdi), %%xmm3;"

      "vunpcklps %%xmm1, %%xmm0, %%xmm12;"
      "vunpckhps %%xmm1, %%xmm0, %%xmm13;"
      "vunpcklps %%xmm3, %%xmm2, %%xmm14;"
      "vunpckhps %%xmm3, %%xmm2, %%xmm15;"

      "vmovlhps %%xmm14, %%xmm12, %%xmm0;"
      "vmovhlps %%xmm12, %%xmm14, %%xmm1;"
      "vmovlhps %%xmm15, %%xmm13, %%xmm2;"
      "vmovhlps %%xmm13, %%xmm15, %%xmm3;"

      "movaps %%xmm0, (%%rdi);"
      "movaps %%xmm1, 16(%%rdi);"
      "movaps %%xmm2, 32(%%rdi);"
      "movaps %%xmm3, 48(%%rdi);"

      : : "m"(m):
    );
    return mat;
  }

  float determinant(matrix4 const &ot)
  {
    return 0.0f;
  }

  matrix4 inverse(matrix4& mat)
  {
    return matrix4();
  }

  matrix4 translate(matrix4 const &mat, vector3 const &position)
  {
    matrix4 n(mat);
    vector4 vin(position);
    matrix4 *out  = &n;
    vector4 *p = &vin;
    __asm__ __volatile__
    (
      "mov %0, %%rdi;"
      "mov %1, %%rsi;"

      "movaps (%%rsi), %%xmm0;"
      "movaps %%xmm0, 48(%%rdi);"
      : :"m"(out), "m"(p) : "%rdi", "%rsi"
    );
    return n;
  }

  matrix4 rotate(matrix4 const &matrix, vector3 const &axis, float theta)
  {
    return matrix4();
  }

  matrix3::matrix3(matrix4 const &mat4)
  {
    vector3 c1(mat4[0][0], mat4[0][1], mat4[0][2]);
    vector3 c2(mat4[1][0], mat4[1][1], mat4[1][2]);
    vector3 c3(mat4[2][0], mat4[2][1], mat4[2][2]);
    matrix3(c1, c2, c3);
  }

  void mulmatvecb(vector4* vout, matrix4* mat, vector4* vin, int n)
  {
    __asm__ __volatile__
    (
      "mov %0, %%rdi;"
      "mov %1, %%rsi;"
      "mov %2, %%rdx;"
      "mov %3, %%ecx;"

      "movaps 00(%%rsi), %%xmm4;"
      "movaps 16(%%rsi), %%xmm5;"
      "movaps 32(%%rsi), %%xmm6;"
      "movaps 48(%%rsi), %%xmm7;"

      "BM:;"
      "movaps (%%rdx), %%xmm0;"
      "prefetchnta 0x30(%%rdx);"
      "movaps %%xmm0, %%xmm1;"
      "add $0x10, %%rdx;"
      "movaps %%xmm0, %%xmm2;"
      "add $0x10, %%rdi;"
      "movaps %%xmm0, %%xmm3;"
      "prefetchnta 0x30(%%rdi);"

      "shufps $0x00, %%xmm0, %%xmm0;"
      "shufps $0x55, %%xmm1, %%xmm1;"
      "shufps $0xAA, %%xmm2, %%xmm2;"
      "shufps $0xFF, %%xmm3, %%xmm3;"

      "mulps %%xmm4, %%xmm0;"
      "mulps %%xmm5, %%xmm1;"
      "mulps %%xmm6, %%xmm2;"
      "mulps %%xmm7, %%xmm3;"

      "addps %%xmm3, %%xmm2;"
      "addps %%xmm2, %%xmm1;"
      "addps %%xmm1, %%xmm0;"

      "movaps %%xmm0, -0x10(%%rdi);"
      "dec %%ecx;"
      "jnz BM;"

      : : "m"(vout), "m"(mat), "m"(vin), "m"(n) :
    );
  }

};
