//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Macros.hpp>

// Suppress "'long double' is treated as 'double' in device code"
// The suppression needs to happen before Kokkos_Complex.hpp is included to be
// effective
#ifdef KOKKOS_COMPILER_NVCC
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20208
#else
#ifdef __CUDA_ARCH__
#pragma diagnostic push
#pragma diag_suppress 3245
#endif
#endif
#endif

#include <Kokkos_Core.hpp>
#include <sstream>

namespace {
template <typename... Ts>
KOKKOS_FUNCTION constexpr void maybe_unused(Ts &&...) noexcept {}
}  // namespace

namespace Test {

// Test construction and assignment

template <class ExecSpace>
struct TestComplexConstruction {
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<double> *, ExecSpace>::HostMirror
      h_results;

  void testit() {
    d_results = Kokkos::View<Kokkos::complex<double> *, ExecSpace>(
        "TestComplexConstruction", 10);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    ASSERT_FLOAT_EQ(h_results(0).real(), 1.5);
    ASSERT_FLOAT_EQ(h_results(0).imag(), 2.5);
    ASSERT_FLOAT_EQ(h_results(1).real(), 1.5);
    ASSERT_FLOAT_EQ(h_results(1).imag(), 2.5);
    ASSERT_FLOAT_EQ(h_results(2).real(), 0.0);
    ASSERT_FLOAT_EQ(h_results(2).imag(), 0.0);
    ASSERT_FLOAT_EQ(h_results(3).real(), 3.5);
    ASSERT_FLOAT_EQ(h_results(3).imag(), 0.0);
    ASSERT_FLOAT_EQ(h_results(4).real(), 4.5);
    ASSERT_FLOAT_EQ(h_results(4).imag(), 5.5);
    ASSERT_FLOAT_EQ(h_results(5).real(), 1.5);
    ASSERT_FLOAT_EQ(h_results(5).imag(), 2.5);
    ASSERT_FLOAT_EQ(h_results(6).real(), 4.5);
    ASSERT_FLOAT_EQ(h_results(6).imag(), 5.5);
    ASSERT_FLOAT_EQ(h_results(7).real(), 7.5);
    ASSERT_FLOAT_EQ(h_results(7).imag(), 0.0);
    ASSERT_FLOAT_EQ(h_results(8).real(), double(8));
    ASSERT_FLOAT_EQ(h_results(8).imag(), 0.0);

    // Copy construction conversion between
    // Kokkos::complex and std::complex doesn't compile
    Kokkos::complex<double> a(1.5, 2.5), b(3.25, 5.25), r_kk;
    std::complex<double> sa(a), sb(3.25, 5.25), r;
    r    = a;
    r_kk = a;
    ASSERT_FLOAT_EQ(r.real(), r_kk.real());
    ASSERT_FLOAT_EQ(r.imag(), r_kk.imag());
    r    = sb * a;
    r_kk = b * a;
    ASSERT_FLOAT_EQ(r.real(), r_kk.real());
    ASSERT_FLOAT_EQ(r.imag(), r_kk.imag());
    r    = sa;
    r_kk = a;
    ASSERT_FLOAT_EQ(r.real(), r_kk.real());
    ASSERT_FLOAT_EQ(r.imag(), r_kk.imag());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    Kokkos::complex<double> a(1.5, 2.5);
    d_results(0) = a;
    Kokkos::complex<double> b(a);
    d_results(1)              = b;
    Kokkos::complex<double> c = Kokkos::complex<double>();
    d_results(2)              = c;
    Kokkos::complex<double> d(3.5);
    d_results(3) = d;
    Kokkos::complex<double> a_v(4.5, 5.5);
    d_results(4) = a_v;
    Kokkos::complex<double> b_v(a);
    d_results(5) = b_v;
    Kokkos::complex<double> e(a_v);
    d_results(6) = e;

    d_results(7) = double(7.5);
    d_results(8) = int(8);
  }
};

TEST(TEST_CATEGORY, complex_construction) {
  TestComplexConstruction<TEST_EXECSPACE> test;
  test.testit();
}

// Test Math FUnction

template <class ExecSpace>
struct TestComplexBasicMath {
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<double> *, ExecSpace>::HostMirror
      h_results;

  void testit() {
    d_results = Kokkos::View<Kokkos::complex<double> *, ExecSpace>(
        "TestComplexBasicMath", 24);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    std::complex<double> a(1.5, 2.5);
    std::complex<double> b(3.25, 5.75);
    std::complex<double> d(1.0, 2.0);
    double c = 9.3;
    int e    = 2;

    std::complex<double> r;
    r = a + b;
    ASSERT_FLOAT_EQ(h_results(0).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(0).imag(), r.imag());
    r = a - b;
    ASSERT_FLOAT_EQ(h_results(1).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(1).imag(), r.imag());
    r = a * b;
    ASSERT_FLOAT_EQ(h_results(2).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(2).imag(), r.imag());
    r = a / b;
#ifndef KOKKOS_WORKAROUND_OPENMPTARGET_CLANG
    ASSERT_FLOAT_EQ(h_results(3).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(3).imag(), r.imag());
#endif
    r = d + a;
    ASSERT_FLOAT_EQ(h_results(4).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(4).imag(), r.imag());
    r = d - a;
    ASSERT_FLOAT_EQ(h_results(5).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(5).imag(), r.imag());
    r = d * a;
    ASSERT_FLOAT_EQ(h_results(6).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(6).imag(), r.imag());
    r = d / a;
    ASSERT_FLOAT_EQ(h_results(7).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(7).imag(), r.imag());
    r = a + c;
    ASSERT_FLOAT_EQ(h_results(8).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(8).imag(), r.imag());
    r = a - c;
    ASSERT_FLOAT_EQ(h_results(9).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(9).imag(), r.imag());
    r = a * c;
    ASSERT_FLOAT_EQ(h_results(10).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(10).imag(), r.imag());
    r = a / c;
    ASSERT_FLOAT_EQ(h_results(11).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(11).imag(), r.imag());
    r = d + c;
    ASSERT_FLOAT_EQ(h_results(12).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(12).imag(), r.imag());
    r = d - c;
    ASSERT_FLOAT_EQ(h_results(13).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(13).imag(), r.imag());
    r = d * c;
    ASSERT_FLOAT_EQ(h_results(14).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(14).imag(), r.imag());
    r = d / c;
    ASSERT_FLOAT_EQ(h_results(15).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(15).imag(), r.imag());
    r = c + a;
    ASSERT_FLOAT_EQ(h_results(16).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(16).imag(), r.imag());
    r = c - a;
    ASSERT_FLOAT_EQ(h_results(17).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(17).imag(), r.imag());
    r = c * a;
    ASSERT_FLOAT_EQ(h_results(18).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(18).imag(), r.imag());
    r = c / a;
#ifndef KOKKOS_WORKAROUND_OPENMPTARGET_CLANG
    ASSERT_FLOAT_EQ(h_results(19).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(19).imag(), r.imag());
#endif

    r = a;
    /* r = a+e; */ ASSERT_FLOAT_EQ(h_results(20).real(), r.real() + e);
    ASSERT_FLOAT_EQ(h_results(20).imag(), r.imag());
    /* r = a-e; */ ASSERT_FLOAT_EQ(h_results(21).real(), r.real() - e);
    ASSERT_FLOAT_EQ(h_results(21).imag(), r.imag());
    /* r = a*e; */ ASSERT_FLOAT_EQ(h_results(22).real(), r.real() * e);
    ASSERT_FLOAT_EQ(h_results(22).imag(), r.imag() * e);
    /* r = a/e; */ ASSERT_FLOAT_EQ(h_results(23).real(), r.real() / 2);
    ASSERT_FLOAT_EQ(h_results(23).imag(), r.imag() / e);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    Kokkos::complex<double> a(1.5, 2.5);
    Kokkos::complex<double> b(3.25, 5.75);
    // Basic math complex / complex
    d_results(0) = a + b;
    d_results(1) = a - b;
    d_results(2) = a * b;
    d_results(3) = a / b;
    d_results(4).real(1.0);
    d_results(4).imag(2.0);
    d_results(4) += a;
    d_results(5) = Kokkos::complex<double>(1.0, 2.0);
    d_results(5) -= a;
    d_results(6) = Kokkos::complex<double>(1.0, 2.0);
    d_results(6) *= a;
    d_results(7) = Kokkos::complex<double>(1.0, 2.0);
    d_results(7) /= a;

    // Basic math complex / scalar
    double c      = 9.3;
    d_results(8)  = a + c;
    d_results(9)  = a - c;
    d_results(10) = a * c;
    d_results(11) = a / c;
    d_results(12).real(1.0);
    d_results(12).imag(2.0);
    d_results(12) += c;
    d_results(13) = Kokkos::complex<double>(1.0, 2.0);
    d_results(13) -= c;
    d_results(14) = Kokkos::complex<double>(1.0, 2.0);
    d_results(14) *= c;
    d_results(15) = Kokkos::complex<double>(1.0, 2.0);
    d_results(15) /= c;

    // Basic math scalar / complex
    d_results(16) = c + a;
    d_results(17) = c - a;
    d_results(18) = c * a;
    d_results(19) = c / a;

    int e         = 2;
    d_results(20) = a + e;
    d_results(21) = a - e;
    d_results(22) = a * e;
    d_results(23) = a / e;
  }
};

TEST(TEST_CATEGORY, complex_basic_math) {
  TestComplexBasicMath<TEST_EXECSPACE> test;
  test.testit();
}

template <class ExecSpace>
struct TestComplexSpecialFunctions {
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<double> *, ExecSpace>::HostMirror
      h_results;

  void testit() {
    d_results = Kokkos::View<Kokkos::complex<double> *, ExecSpace>(
        "TestComplexSpecialFunctions", 20);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    std::complex<double> a(1.5, 2.5);
    double c = 9.3;

    std::complex<double> r;
    r = a;
    ASSERT_FLOAT_EQ(h_results(0).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(0).imag(), r.imag());
    r = std::sqrt(a);
    ASSERT_FLOAT_EQ(h_results(1).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(1).imag(), r.imag());
    r = std::pow(a, c);
    ASSERT_FLOAT_EQ(h_results(2).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(2).imag(), r.imag());
    r = std::abs(a);
    ASSERT_FLOAT_EQ(h_results(3).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(3).imag(), r.imag());
    r = std::exp(a);
    ASSERT_FLOAT_EQ(h_results(4).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(4).imag(), r.imag());
    r = Kokkos::exp(a);
    ASSERT_FLOAT_EQ(h_results(4).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(4).imag(), r.imag());
#ifndef KOKKOS_WORKAROUND_OPENMPTARGET_CLANG
    r = std::log(a);
    ASSERT_FLOAT_EQ(h_results(5).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(5).imag(), r.imag());
    r = std::sin(a);
    ASSERT_FLOAT_EQ(h_results(6).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(6).imag(), r.imag());
    r = std::cos(a);
    ASSERT_FLOAT_EQ(h_results(7).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(7).imag(), r.imag());
    r = std::tan(a);
    ASSERT_FLOAT_EQ(h_results(8).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(8).imag(), r.imag());
    r = std::sinh(a);
    ASSERT_FLOAT_EQ(h_results(9).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(9).imag(), r.imag());
    r = std::cosh(a);
    ASSERT_FLOAT_EQ(h_results(10).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(10).imag(), r.imag());
    r = std::tanh(a);
    ASSERT_FLOAT_EQ(h_results(11).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(11).imag(), r.imag());
    r = std::asinh(a);
    ASSERT_FLOAT_EQ(h_results(12).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(12).imag(), r.imag());
    r = std::acosh(a);
    ASSERT_FLOAT_EQ(h_results(13).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(13).imag(), r.imag());
    // atanh
    // Work around a bug in gcc 5.3.1 where the compiler cannot compute atanh
    r = {0.163481616851666003, 1.27679502502111284};
    ASSERT_FLOAT_EQ(h_results(14).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(14).imag(), r.imag());
    r = std::asin(a);
    ASSERT_FLOAT_EQ(h_results(15).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(15).imag(), r.imag());
    r = std::acos(a);
    ASSERT_FLOAT_EQ(h_results(16).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(16).imag(), r.imag());
    // atan
    // Work around a bug in gcc 5.3.1 where the compiler cannot compute atan
    r = {1.380543138238714, 0.2925178131625636};
    ASSERT_FLOAT_EQ(h_results(17).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(17).imag(), r.imag());
    // log10
    r = std::log10(a);
    ASSERT_FLOAT_EQ(h_results(18).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(18).imag(), r.imag());
#endif
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    Kokkos::complex<double> a(1.5, 2.5);
    double c = 9.3;

    d_results(0)  = Kokkos::complex<double>(Kokkos::real(a), Kokkos::imag(a));
    d_results(1)  = Kokkos::sqrt(a);
    d_results(2)  = Kokkos::pow(a, c);
    d_results(3)  = Kokkos::abs(a);
    d_results(4)  = Kokkos::exp(a);
    d_results(5)  = Kokkos::log(a);
    d_results(6)  = Kokkos::sin(a);
    d_results(7)  = Kokkos::cos(a);
    d_results(8)  = Kokkos::tan(a);
    d_results(9)  = Kokkos::sinh(a);
    d_results(10) = Kokkos::cosh(a);
    d_results(11) = Kokkos::tanh(a);
    d_results(12) = Kokkos::asinh(a);
    d_results(13) = Kokkos::acosh(a);
    d_results(14) = Kokkos::atanh(a);
    d_results(15) = Kokkos::asin(a);
    d_results(16) = Kokkos::acos(a);
    d_results(17) = Kokkos::atan(a);
    d_results(18) = Kokkos::log10(a);
  }
};

void testComplexIO() {
  Kokkos::complex<double> z = {3.14, 1.41};
  std::stringstream ss;
  ss << z;
  ASSERT_EQ(ss.str(), "(3.14,1.41)");

  ss.str("1 (2) (3,4)");
  ss.clear();
  ss >> z;
  ASSERT_EQ(z, (Kokkos::complex<double>{1, 0}));
  ss >> z;
  ASSERT_EQ(z, (Kokkos::complex<double>{2, 0}));
  ss >> z;
  ASSERT_EQ(z, (Kokkos::complex<double>{3, 4}));
}

TEST(TEST_CATEGORY, complex_special_funtions) {
  TestComplexSpecialFunctions<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, complex_io) { testComplexIO(); }

TEST(TEST_CATEGORY, complex_trivially_copyable) {
  // Kokkos::complex<RealType> is trivially copyable when RealType is
  // trivially copyable
  using RealType = double;
  // clang claims compatibility with gcc 4.2.1 but all versions tested know
  // about std::is_trivially_copyable.
  ASSERT_TRUE(std::is_trivially_copyable_v<Kokkos::complex<RealType>> ||
              !std::is_trivially_copyable_v<RealType>);
}

template <class ExecSpace>
struct TestBugPowAndLogComplex {
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_pow;
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_log;
  TestBugPowAndLogComplex() : d_pow("pow", 2), d_log("log", 2) { test(); }
  void test() {
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    auto h_pow =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_pow);
    ASSERT_FLOAT_EQ(h_pow(0).real(), 18);
    ASSERT_FLOAT_EQ(h_pow(0).imag(), 26);
    ASSERT_FLOAT_EQ(h_pow(1).real(), -18);
    ASSERT_FLOAT_EQ(h_pow(1).imag(), 26);
    auto h_log =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_log);
    ASSERT_FLOAT_EQ(h_log(0).real(), 1.151292546497023);
    ASSERT_FLOAT_EQ(h_log(0).imag(), 0.3217505543966422);
    ASSERT_FLOAT_EQ(h_log(1).real(), 1.151292546497023);
    ASSERT_FLOAT_EQ(h_log(1).imag(), 2.819842099193151);
  }
  KOKKOS_FUNCTION void operator()(int) const {
    d_pow(0) = Kokkos::pow(Kokkos::complex<double>(+3., 1.), 3.);
    d_pow(1) = Kokkos::pow(Kokkos::complex<double>(-3., 1.), 3.);
    d_log(0) = Kokkos::log(Kokkos::complex<double>(+3., 1.));
    d_log(1) = Kokkos::log(Kokkos::complex<double>(-3., 1.));
  }
};

TEST(TEST_CATEGORY, complex_issue_3865) {
  TestBugPowAndLogComplex<TEST_EXECSPACE>();
}

#ifdef KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET
TEST(TEST_CATEGORY, complex_issue_3867) {
  ASSERT_EQ(Kokkos::pow(Kokkos::complex<double>(2., 1.), 3.),
            Kokkos::pow(Kokkos::complex<double>(2., 1.), 3));
  ASSERT_EQ(
      Kokkos::pow(Kokkos::complex<double>(2., 1.), 3.),
      Kokkos::pow(Kokkos::complex<double>(2., 1.), Kokkos::complex<double>(3)));

  auto x = Kokkos::pow(Kokkos::complex<double>(2, 1),
                       Kokkos::complex<double>(-3, 4));
  auto y = Kokkos::complex<double>(
      std::pow(std::complex<double>(2, 1), std::complex<double>(-3, 4)));
  ASSERT_FLOAT_EQ(x.real(), y.real());
  ASSERT_FLOAT_EQ(x.imag(), y.imag());

#define CHECK_POW_COMPLEX_PROMOTION(ARGTYPE1, ARGTYPE2, RETURNTYPE)          \
  static_assert(                                                             \
      std::is_same<RETURNTYPE,                                               \
                   decltype(Kokkos::pow(std::declval<ARGTYPE1>(),            \
                                        std::declval<ARGTYPE2>()))>::value); \
  static_assert(                                                             \
      std::is_same<RETURNTYPE,                                               \
                   decltype(Kokkos::pow(std::declval<ARGTYPE2>(),            \
                                        std::declval<ARGTYPE1>()))>::value);

  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<long double>, long double,
                              Kokkos::complex<long double>);
  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<long double>, double,
                              Kokkos::complex<long double>);
  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<long double>, float,
                              Kokkos::complex<long double>);
  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<long double>, int,
                              Kokkos::complex<long double>);

  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<double>, long double,
                              Kokkos::complex<long double>);
  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<double>, double,
                              Kokkos::complex<double>);
  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<double>, float,
                              Kokkos::complex<double>);
  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<double>, int,
                              Kokkos::complex<double>);

  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<float>, long double,
                              Kokkos::complex<long double>);
  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<float>, double,
                              Kokkos::complex<double>);
  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<float>, float,
                              Kokkos::complex<float>);
  CHECK_POW_COMPLEX_PROMOTION(Kokkos::complex<float>, int,
                              Kokkos::complex<double>);

#undef CHECK_POW_COMPLEX_PROMOTION
}
#endif

TEST(TEST_CATEGORY, complex_operations_arithmetic_types_overloads) {
  static_assert(Kokkos::real(1) == 1.);
  static_assert(Kokkos::real(2.f) == 2.f);
  static_assert(Kokkos::real(3.) == 3.);
  static_assert(Kokkos::real(4.l) == 4.l);
  static_assert((std::is_same_v<decltype(Kokkos::real(1)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::real(2.f)), float>));
  static_assert((std::is_same_v<decltype(Kokkos::real(3.)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::real(4.l)), long double>));

  static_assert(Kokkos::imag(1) == 0.);
  static_assert(Kokkos::imag(2.f) == 0.f);
  static_assert(Kokkos::imag(3.) == 0.);
  static_assert(Kokkos::imag(4.l) == 0.l);
  static_assert((std::is_same_v<decltype(Kokkos::imag(1)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::imag(2.f)), float>));
  static_assert((std::is_same_v<decltype(Kokkos::imag(3.)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::real(4.l)), long double>));

  // FIXME in principle could be checked at compile time too
  ASSERT_EQ(Kokkos::conj(1), Kokkos::complex<double>(1));
  ASSERT_EQ(Kokkos::conj(2.f), Kokkos::complex<float>(2.f));
  ASSERT_EQ(Kokkos::conj(3.), Kokkos::complex<double>(3.));
// long double has size 12 but Kokkos::complex requires 2*sizeof(T) to be a
// power of two.
#ifndef KOKKOS_IMPL_32BIT
  ASSERT_EQ(Kokkos::conj(4.l), Kokkos::complex<long double>(4.l));
  static_assert(
      (std::is_same_v<decltype(Kokkos::conj(1)), Kokkos::complex<double>>));
#endif
  static_assert(
      (std::is_same_v<decltype(Kokkos::conj(2.f)), Kokkos::complex<float>>));
  static_assert(
      (std::is_same_v<decltype(Kokkos::conj(3.)), Kokkos::complex<double>>));
  static_assert((std::is_same_v<decltype(Kokkos::conj(4.l)),
                                Kokkos::complex<long double>>));
}

template <class ExecSpace>
struct TestComplexStructuredBindings {
  using exec_space       = ExecSpace;
  using value_type       = double;
  using complex_type     = Kokkos::complex<double>;
  using device_view_type = Kokkos::View<complex_type *, exec_space>;
  using host_view_type   = typename device_view_type::HostMirror;

  device_view_type d_results;
  host_view_type h_results;

  // tuple_size
  static_assert(std::is_same_v<std::tuple_size<complex_type>::type,
                               std::integral_constant<size_t, 2>>);

  // tuple_element
  static_assert(
      std::is_same_v<std::tuple_element_t<0, complex_type>, value_type>);
  static_assert(
      std::is_same_v<std::tuple_element_t<1, complex_type>, value_type>);

  static void testgetreturnreferencetypes() {
    complex_type m;
    const complex_type c;

    // get lvalue
    complex_type &ml = m;
    static_assert(std::is_same_v<decltype(Kokkos::get<0>(ml)), value_type &>);
    static_assert(std::is_same_v<decltype(Kokkos::get<1>(ml)), value_type &>);

    // get rvalue
    complex_type &&mr = std::move(m);
    static_assert(
        std::is_same_v<decltype(Kokkos::get<0>(std::move(mr))), value_type &&>);
    static_assert(
        std::is_same_v<decltype(Kokkos::get<1>(std::move(mr))), value_type &&>);

    // get const lvalue
    const complex_type &cl = c;
    static_assert(
        std::is_same_v<decltype(Kokkos::get<0>(cl)), value_type const &>);
    static_assert(
        std::is_same_v<decltype(Kokkos::get<1>(cl)), value_type const &>);

    // get const rvalue
    complex_type const &&cr = std::move(c);
    static_assert(std::is_same_v<decltype(Kokkos::get<0>(std::move(cr))),
                                 value_type const &&>);
    static_assert(std::is_same_v<decltype(Kokkos::get<1>(std::move(cr))),
                                 value_type const &&>);

    maybe_unused(m, c, ml, mr, cl, cr);
  }

  void testit() {
    testgetreturnreferencetypes();

    d_results = device_view_type("TestComplexStructuredBindings", 6);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    // get lvalue
    ASSERT_FLOAT_EQ(h_results[0].real(), 2.);
    ASSERT_FLOAT_EQ(h_results[0].imag(), 3.);

    // get rvalue
    ASSERT_FLOAT_EQ(h_results[1].real(), 2.);
    ASSERT_FLOAT_EQ(h_results[1].imag(), 3.);

    // get const lvalue
    ASSERT_FLOAT_EQ(h_results[2].real(), 5.);
    ASSERT_FLOAT_EQ(h_results[2].imag(), 7.);

    // get const rvalue
    ASSERT_FLOAT_EQ(h_results[3].real(), 5.);
    ASSERT_FLOAT_EQ(h_results[3].imag(), 7.);

    // swap real and imaginary
    ASSERT_FLOAT_EQ(h_results[4].real(), 11.);
    ASSERT_FLOAT_EQ(h_results[4].imag(), 13.);
    ASSERT_FLOAT_EQ(h_results[5].real(), 13.);
    ASSERT_FLOAT_EQ(h_results[5].imag(), 11.);
  }

  KOKKOS_FUNCTION
  void operator()(int) const {
    complex_type m(2., 3.);
    const complex_type c(5., 7.);

    // get lvalue
    {
      complex_type &ml = m;
      auto &[mlr, mli] = ml;
      d_results[0]     = complex_type(mlr, mli);
    }

    // get rvalue
    {
      complex_type &&mr = std::move(m);
      auto &&[mrr, mri] = std::move(mr);
      d_results[1]      = complex_type(mrr, mri);
    }

    // get const lvalue
    {
      const complex_type &cl = c;
      auto &[clr, cli]       = cl;
      d_results[2]           = complex_type(clr, cli);
    }

    // get const rvalue
    {
      complex_type const &&cr = std::move(c);
      auto &&[crr, cri]       = std::move(cr);
      d_results[3]            = complex_type(crr, cri);
    }

    // swap real and imaginary
    {
      complex_type z(11., 13.);
      d_results[4] = z;

      auto &[zr, zi] = z;
      Kokkos::kokkos_swap(zr, zi);
      d_results[5] = z;
    }
  }
};

TEST(TEST_CATEGORY, complex_structured_bindings) {
  TestComplexStructuredBindings<TEST_EXECSPACE> test;
  test.testit();
}

#define CHECK_COMPLEX(_value_, _real_, _imag_) \
  (void)_value_;                               \
  if (_value_.real() != _real_) return false;  \
  if (_value_.imag() != _imag_) return false;

constexpr bool can_appear_in_constant_expressions() {
  const Kokkos::complex<double> from_single{1.2};
  const Kokkos::complex<double> from_both{1.2, 3.4};
  const Kokkos::complex<double> from_none{};

  CHECK_COMPLEX(from_single, 1.2, 0.);
  CHECK_COMPLEX(from_both, 1.2, 3.4);
  CHECK_COMPLEX(from_none, 0., 0.);

  Kokkos::complex<double> from_copy_assign;
  from_copy_assign = from_both;
  const auto from_copy_constr(from_both);

  CHECK_COMPLEX(from_copy_assign, 1.2, 3.4);
  CHECK_COMPLEX(from_copy_constr, 1.2, 3.4);

  Kokkos::complex<double> from_move_assign;
  from_move_assign = std::move(from_both);
  const auto from_move_constr(std::move(from_copy_assign));

  CHECK_COMPLEX(from_move_assign, 1.2, 3.4);
  CHECK_COMPLEX(from_move_constr, 1.2, 3.4);

  Kokkos::complex<double> from_real;
  from_real = 4.;

  CHECK_COMPLEX(from_real, 4., 0.);

  return true;
}

#undef CHECK_COMPLEX

static_assert(can_appear_in_constant_expressions());

}  // namespace Test

#ifdef KOKKOS_COMPILER_NVCC
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diagnostic pop
#else
#ifdef __CUDA_ARCH__
#pragma diagnostic pop
#endif
#endif
#endif
