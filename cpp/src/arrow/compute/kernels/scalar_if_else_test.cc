// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <gtest/gtest.h>
#include "arrow/array.h"
#include "arrow/array/concatenate.h"
#include "arrow/compute/api_scalar.h"
#include "arrow/compute/kernels/test_util.h"
#include "arrow/compute/registry.h"
#include "arrow/testing/gtest_util.h"

namespace arrow {
namespace compute {

void CheckIfElseOutput(const Datum& cond, const Datum& left, const Datum& right,
                       const Datum& expected) {
  ASSERT_OK_AND_ASSIGN(Datum datum_out, IfElse(cond, left, right));
  if (datum_out.is_array()) {
    std::shared_ptr<Array> result = datum_out.make_array();
    ValidateOutput(*result);
    std::shared_ptr<Array> expected_ = expected.make_array();
    AssertArraysEqual(*expected_, *result, /*verbose=*/true);
  } else {  // expecting scalar
    const std::shared_ptr<Scalar>& result = datum_out.scalar();
    const std::shared_ptr<Scalar>& expected_ = expected.scalar();
    AssertScalarsEqual(*expected_, *result, /*verbose=*/true);
  }
}

class TestIfElseKernel : public ::testing::Test {};

template <typename Type>
class TestIfElsePrimitive : public ::testing::Test {};

using NumericBasedTypes =
    ::testing::Types<UInt8Type, UInt16Type, UInt32Type, UInt64Type, Int8Type, Int16Type,
                     Int32Type, Int64Type, FloatType, DoubleType, Date32Type, Date64Type,
                     Time32Type, Time64Type, TimestampType, MonthIntervalType>;

TYPED_TEST_SUITE(TestIfElsePrimitive, NumericBasedTypes);

TYPED_TEST(TestIfElsePrimitive, IfElseFixedSizeRand) {
  using ArrayType = typename TypeTraits<TypeParam>::ArrayType;
  auto type = default_type_instance<TypeParam>();

  random::RandomArrayGenerator rand(/*seed=*/0);
  int64_t len = 1000;

  // adding 64 consecutive 1's and 0's in the cond array to test all-true/ all-false
  // word code paths
  ASSERT_OK_AND_ASSIGN(auto temp1, MakeArrayFromScalar(BooleanScalar(true), 64));
  ASSERT_OK_AND_ASSIGN(auto temp2, MakeArrayFromScalar(BooleanScalar(false), 64));
  auto temp3 = rand.ArrayOf(boolean(), len - 64 * 2, /*null_probability=*/0.01);

  ASSERT_OK_AND_ASSIGN(auto concat, Concatenate({temp1, temp2, temp3}));
  auto cond = std::static_pointer_cast<BooleanArray>(concat);
  auto left = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto right = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));

  typename TypeTraits<TypeParam>::BuilderType builder(type, default_memory_pool());

  for (int64_t i = 0; i < len; ++i) {
    if (!cond->IsValid(i) || (cond->Value(i) && !left->IsValid(i)) ||
        (!cond->Value(i) && !right->IsValid(i))) {
      ASSERT_OK(builder.AppendNull());
      continue;
    }

    if (cond->Value(i)) {
      ASSERT_OK(builder.Append(left->Value(i)));
    } else {
      ASSERT_OK(builder.Append(right->Value(i)));
    }
  }
  ASSERT_OK_AND_ASSIGN(auto expected_data, builder.Finish());

  CheckIfElseOutput(cond, left, right, expected_data);
}

void CheckWithDifferentShapes(const std::shared_ptr<Array>& cond,
                              const std::shared_ptr<Array>& left,
                              const std::shared_ptr<Array>& right,
                              const std::shared_ptr<Array>& expected) {
  // this will check for whole arrays, every scalar at i'th index and slicing (offset)
  CheckScalar("if_else", {cond, left, right}, expected);

  auto len = left->length();

  enum { COND_SCALAR = 1, LEFT_SCALAR = 2, RIGHT_SCALAR = 4 };
  for (int mask = 0; mask < (COND_SCALAR | LEFT_SCALAR | RIGHT_SCALAR); ++mask) {
    for (int64_t cond_idx = 0; cond_idx < len; ++cond_idx) {
      Datum cond_in, cond_bcast;
      std::string trace_cond = "Cond";
      if (mask & COND_SCALAR) {
        ASSERT_OK_AND_ASSIGN(cond_in, cond->GetScalar(cond_idx));
        ASSERT_OK_AND_ASSIGN(cond_bcast, MakeArrayFromScalar(*cond_in.scalar(), len));
        trace_cond += "@" + std::to_string(cond_idx) + "=" + cond_in.scalar()->ToString();
      } else {
        cond_in = cond_bcast = cond;
      }
      SCOPED_TRACE(trace_cond);

      for (int64_t left_idx = 0; left_idx < len; ++left_idx) {
        Datum left_in, left_bcast;
        std::string trace_left = "Left";
        if (mask & LEFT_SCALAR) {
          ASSERT_OK_AND_ASSIGN(left_in, left->GetScalar(left_idx).As<Datum>());
          ASSERT_OK_AND_ASSIGN(left_bcast, MakeArrayFromScalar(*left_in.scalar(), len));
          trace_cond +=
              "@" + std::to_string(left_idx) + "=" + left_in.scalar()->ToString();
        } else {
          left_in = left_bcast = left;
        }
        SCOPED_TRACE(trace_left);

        for (int64_t right_idx = 0; right_idx < len; ++right_idx) {
          Datum right_in, right_bcast;
          std::string trace_right = "Right";
          if (mask & RIGHT_SCALAR) {
            ASSERT_OK_AND_ASSIGN(right_in, right->GetScalar(right_idx));
            ASSERT_OK_AND_ASSIGN(right_bcast,
                                 MakeArrayFromScalar(*right_in.scalar(), len));
            trace_right +=
                "@" + std::to_string(right_idx) + "=" + right_in.scalar()->ToString();
          } else {
            right_in = right_bcast = right;
          }
          SCOPED_TRACE(trace_right);

          ASSERT_OK_AND_ASSIGN(auto exp, IfElse(cond_bcast, left_bcast, right_bcast));
          ASSERT_OK_AND_ASSIGN(auto actual, IfElse(cond_in, left_in, right_in));
          AssertDatumsEqual(exp, actual, /*verbose=*/true);

          if (right_in.is_array()) break;
        }
        if (left_in.is_array()) break;
      }
      if (cond_in.is_array()) break;
    }
  }  // for (mask)
}

TYPED_TEST(TestIfElsePrimitive, IfElseFixedSize) {
  auto type = default_type_instance<TypeParam>();

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, "[1, 2, 3, 4]"),
                           ArrayFromJSON(type, "[5, 6, 7, 8]"),
                           ArrayFromJSON(type, "[1, 2, 3, 8]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, "[1, 2, 3, 4]"),
                           ArrayFromJSON(type, "[5, 6, 7, null]"),
                           ArrayFromJSON(type, "[1, 2, 3, null]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, "[1, 2, null, 4]"),
                           ArrayFromJSON(type, "[5, 6, 7, null]"),
                           ArrayFromJSON(type, "[1, 2, null, null]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, "[1, 2, null, 4]"),
                           ArrayFromJSON(type, "[5, 6, 7, 8]"),
                           ArrayFromJSON(type, "[1, 2, null, 8]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[null, true, true, false]"),
                           ArrayFromJSON(type, "[1, 2, null, 4]"),
                           ArrayFromJSON(type, "[5, 6, 7, 8]"),
                           ArrayFromJSON(type, "[null, 2, null, 8]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[null, true, true, false]"),
                           ArrayFromJSON(type, "[1, 2, null, 4]"),
                           ArrayFromJSON(type, "[5, 6, 7, null]"),
                           ArrayFromJSON(type, "[null, 2, null, null]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[null, true, true, false]"),
                           ArrayFromJSON(type, "[1, 2, 3, 4]"),
                           ArrayFromJSON(type, "[5, 6, 7, null]"),
                           ArrayFromJSON(type, "[null, 2, 3, null]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[null, true, true, false]"),
                           ArrayFromJSON(type, "[1, 2, 3, 4]"),
                           ArrayFromJSON(type, "[5, 6, 7, 8]"),
                           ArrayFromJSON(type, "[null, 2, 3, 8]"));
}

TEST_F(TestIfElseKernel, IfElseBoolean) {
  auto type = boolean();

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, "[false, false, false, false]"),
                           ArrayFromJSON(type, "[true, true, true, true]"),
                           ArrayFromJSON(type, "[false, false, false, true]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, "[false, false, false, false]"),
                           ArrayFromJSON(type, "[true, true, true, null]"),
                           ArrayFromJSON(type, "[false, false, false, null]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, "[false, false, null, false]"),
                           ArrayFromJSON(type, "[true, true, true, null]"),
                           ArrayFromJSON(type, "[false, false, null, null]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, "[false, false, null, false]"),
                           ArrayFromJSON(type, "[true, true, true, true]"),
                           ArrayFromJSON(type, "[false, false, null, true]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[null, true, true, false]"),
                           ArrayFromJSON(type, "[false, false, null, false]"),
                           ArrayFromJSON(type, "[true, true, true, true]"),
                           ArrayFromJSON(type, "[null, false, null, true]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[null, true, true, false]"),
                           ArrayFromJSON(type, "[false, false, null, false]"),
                           ArrayFromJSON(type, "[true, true, true, null]"),
                           ArrayFromJSON(type, "[null, false, null, null]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[null, true, true, false]"),
                           ArrayFromJSON(type, "[false, false, false, false]"),
                           ArrayFromJSON(type, "[true, true, true, null]"),
                           ArrayFromJSON(type, "[null, false, false, null]"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[null, true, true, false]"),
                           ArrayFromJSON(type, "[false, false, false, false]"),
                           ArrayFromJSON(type, "[true, true, true, true]"),
                           ArrayFromJSON(type, "[null, false, false, true]"));
}

TEST_F(TestIfElseKernel, IfElseBooleanRand) {
  auto type = boolean();
  random::RandomArrayGenerator rand(/*seed=*/0);
  int64_t len = 1000;
  auto cond = std::static_pointer_cast<BooleanArray>(
      rand.ArrayOf(boolean(), len, /*null_probability=*/0.01));
  auto left = std::static_pointer_cast<BooleanArray>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto right = std::static_pointer_cast<BooleanArray>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));

  BooleanBuilder builder;
  for (int64_t i = 0; i < len; ++i) {
    if (!cond->IsValid(i) || (cond->Value(i) && !left->IsValid(i)) ||
        (!cond->Value(i) && !right->IsValid(i))) {
      ASSERT_OK(builder.AppendNull());
      continue;
    }

    if (cond->Value(i)) {
      ASSERT_OK(builder.Append(left->Value(i)));
    } else {
      ASSERT_OK(builder.Append(right->Value(i)));
    }
  }
  ASSERT_OK_AND_ASSIGN(auto expected_data, builder.Finish());

  CheckIfElseOutput(cond, left, right, expected_data);
}

TEST_F(TestIfElseKernel, IfElseNull) {
  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[null, null, null, null]"),
                           ArrayFromJSON(null(), "[null, null, null, null]"),
                           ArrayFromJSON(null(), "[null, null, null, null]"),
                           ArrayFromJSON(null(), "[null, null, null, null]"));
}

TEST_F(TestIfElseKernel, IfElseMultiType) {
  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(int32(), "[1, 2, 3, 4]"),
                           ArrayFromJSON(float32(), "[5, 6, 7, 8]"),
                           ArrayFromJSON(float32(), "[1, 2, 3, 8]"));
}

TEST_F(TestIfElseKernel, IfElseDispatchBest) {
  std::string name = "if_else";
  CheckDispatchBest(name, {boolean(), int32(), int32()}, {boolean(), int32(), int32()});
  CheckDispatchBest(name, {boolean(), int32(), null()}, {boolean(), int32(), int32()});
  CheckDispatchBest(name, {boolean(), null(), int32()}, {boolean(), int32(), int32()});

  CheckDispatchBest(name, {boolean(), int32(), int8()}, {boolean(), int32(), int32()});
  CheckDispatchBest(name, {boolean(), int32(), int16()}, {boolean(), int32(), int32()});
  CheckDispatchBest(name, {boolean(), int32(), int32()}, {boolean(), int32(), int32()});
  CheckDispatchBest(name, {boolean(), int32(), int64()}, {boolean(), int64(), int64()});

  CheckDispatchBest(name, {boolean(), int32(), uint8()}, {boolean(), int32(), int32()});
  CheckDispatchBest(name, {boolean(), int32(), uint16()}, {boolean(), int32(), int32()});
  CheckDispatchBest(name, {boolean(), int32(), uint32()}, {boolean(), int64(), int64()});
  CheckDispatchBest(name, {boolean(), int32(), uint64()}, {boolean(), int64(), int64()});

  CheckDispatchBest(name, {boolean(), uint8(), uint8()}, {boolean(), uint8(), uint8()});
  CheckDispatchBest(name, {boolean(), uint8(), uint16()},
                    {boolean(), uint16(), uint16()});

  CheckDispatchBest(name, {boolean(), int32(), float32()},
                    {boolean(), float32(), float32()});
  CheckDispatchBest(name, {boolean(), float32(), int64()},
                    {boolean(), float32(), float32()});
  CheckDispatchBest(name, {boolean(), float64(), int32()},
                    {boolean(), float64(), float64()});

  CheckDispatchBest(name, {null(), uint8(), int8()}, {boolean(), int16(), int16()});
}

template <typename Type>
class TestIfElseBaseBinary : public ::testing::Test {};

TYPED_TEST_SUITE(TestIfElseBaseBinary, BinaryArrowTypes);

TYPED_TEST(TestIfElseBaseBinary, IfElseBaseBinary) {
  auto type = TypeTraits<TypeParam>::type_singleton();

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, R"(["a", "ab", "abc", "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmn", "lm", "l"])"),
                           ArrayFromJSON(type, R"(["a", "ab", "abc", "l"])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([true, true, true, false])"),
                           ArrayFromJSON(type, R"(["a", "ab", "abc", "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmn", "lm", null])"),
                           ArrayFromJSON(type, R"(["a", "ab", "abc", null])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([true, true, true, false])"),
                           ArrayFromJSON(type, R"(["a", "ab", null, "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmn", "lm", null])"),
                           ArrayFromJSON(type, R"(["a", "ab", null, null])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([true, true, true, false])"),
                           ArrayFromJSON(type, R"(["a", "ab", null, "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmn", "lm", "l"])"),
                           ArrayFromJSON(type, R"(["a", "ab", null, "l"])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([null, true, true, false])"),
                           ArrayFromJSON(type, R"(["a", "ab", null, "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmn", "lm", "l"])"),
                           ArrayFromJSON(type, R"([null, "ab", null, "l"])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([null, true, true, false])"),
                           ArrayFromJSON(type, R"(["a", "ab", null, "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmn", "lm", null])"),
                           ArrayFromJSON(type, R"([null, "ab", null, null])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([null, true, true, false])"),
                           ArrayFromJSON(type, R"(["a", "ab", "abc", "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmn", "lm", null])"),
                           ArrayFromJSON(type, R"([null, "ab", "abc", null])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([null, true, true, false])"),
                           ArrayFromJSON(type, R"(["a", "ab", "abc", "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmn", "lm", "l"])"),
                           ArrayFromJSON(type, R"([null, "ab", "abc", "l"])"));
}

TYPED_TEST(TestIfElseBaseBinary, IfElseBaseBinaryRand) {
  using ArrayType = typename TypeTraits<TypeParam>::ArrayType;
  using OffsetType = typename TypeTraits<TypeParam>::OffsetType::c_type;
  auto type = TypeTraits<TypeParam>::type_singleton();

  random::RandomArrayGenerator rand(/*seed=*/0);
  int64_t len = 1000;

  //  this is to check the BitBlockCount::AllSet/ NoneSet code paths
  ASSERT_OK_AND_ASSIGN(auto temp1, MakeArrayFromScalar(BooleanScalar(true), 64));
  ASSERT_OK_AND_ASSIGN(auto temp2, MakeArrayFromScalar(BooleanScalar(false), 64));
  auto temp3 = rand.ArrayOf(boolean(), len - 64 * 2, /*null_probability=*/0.01);

  ASSERT_OK_AND_ASSIGN(auto concat, Concatenate({temp1, temp2, temp3}));
  auto cond = std::static_pointer_cast<BooleanArray>(concat);

  auto left = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto right = std::static_pointer_cast<ArrayType>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));

  typename TypeTraits<TypeParam>::BuilderType builder;

  for (int64_t i = 0; i < len; ++i) {
    if (!cond->IsValid(i) || (cond->Value(i) && !left->IsValid(i)) ||
        (!cond->Value(i) && !right->IsValid(i))) {
      ASSERT_OK(builder.AppendNull());
      continue;
    }

    OffsetType offset;
    const uint8_t* val;
    if (cond->Value(i)) {
      val = left->GetValue(i, &offset);
    } else {
      val = right->GetValue(i, &offset);
    }
    ASSERT_OK(builder.Append(val, offset));
  }
  ASSERT_OK_AND_ASSIGN(auto expected_data, builder.Finish());

  CheckIfElseOutput(cond, left, right, expected_data);
}

TEST_F(TestIfElseKernel, IfElseFSBinary) {
  auto type = std::make_shared<FixedSizeBinaryType>(4);

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), "[true, true, true, false]"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", "abca", "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmnl", "lmlm", "llll"])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", "abca", "llll"])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([true, true, true, false])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", "abca", "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmnl", "lmlm", null])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", "abca", null])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([true, true, true, false])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", null, "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmnl", "lmlm", null])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", null, null])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([true, true, true, false])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", null, "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmnl", "lmlm", "llll"])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", null, "llll"])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([null, true, true, false])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", null, "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmnl", "lmlm", "llll"])"),
                           ArrayFromJSON(type, R"([null, "abab", null, "llll"])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([null, true, true, false])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", null, "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmnl", "lmlm", null])"),
                           ArrayFromJSON(type, R"([null, "abab", null, null])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([null, true, true, false])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", "abca", "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmnl", "lmlm", null])"),
                           ArrayFromJSON(type, R"([null, "abab", "abca", null])"));

  CheckWithDifferentShapes(ArrayFromJSON(boolean(), R"([null, true, true, false])"),
                           ArrayFromJSON(type, R"(["aaaa", "abab", "abca", "abcd"])"),
                           ArrayFromJSON(type, R"(["lmno", "lmnl", "lmlm", "llll"])"),
                           ArrayFromJSON(type, R"([null, "abab", "abca", "llll"])"));

  // should fails for non-equal byte_widths
  auto type1 = std::make_shared<FixedSizeBinaryType>(5);
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, ::testing::HasSubstr("FixedSizeBinaryType byte_widths should be equal"),
      CallFunction("if_else", {ArrayFromJSON(boolean(), "[true]"),
                               ArrayFromJSON(type, R"(["aaaa"])"),
                               ArrayFromJSON(type1, R"(["aaaaa"])")}));
}

TEST_F(TestIfElseKernel, IfElseFSBinaryRand) {
  auto type = std::make_shared<FixedSizeBinaryType>(5);

  random::RandomArrayGenerator rand(/*seed=*/0);
  int64_t len = 1000;

  //  this is to check the BitBlockCount::AllSet/ NoneSet code paths
  ASSERT_OK_AND_ASSIGN(auto temp1, MakeArrayFromScalar(BooleanScalar(true), 64));
  ASSERT_OK_AND_ASSIGN(auto temp2, MakeArrayFromScalar(BooleanScalar(false), 64));
  auto temp3 = rand.ArrayOf(boolean(), len - 64 * 2, /*null_probability=*/0.01);

  ASSERT_OK_AND_ASSIGN(auto concat, Concatenate({temp1, temp2, temp3}));
  auto cond = std::static_pointer_cast<BooleanArray>(concat);

  auto left = std::static_pointer_cast<FixedSizeBinaryArray>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));
  auto right = std::static_pointer_cast<FixedSizeBinaryArray>(
      rand.ArrayOf(type, len, /*null_probability=*/0.01));

  FixedSizeBinaryBuilder builder(type);

  for (int64_t i = 0; i < len; ++i) {
    if (!cond->IsValid(i) || (cond->Value(i) && !left->IsValid(i)) ||
        (!cond->Value(i) && !right->IsValid(i))) {
      ASSERT_OK(builder.AppendNull());
      continue;
    }

    const uint8_t* val;
    if (cond->Value(i)) {
      val = left->GetValue(i);
    } else {
      val = right->GetValue(i);
    }
    ASSERT_OK(builder.Append(val));
  }
  ASSERT_OK_AND_ASSIGN(auto expected_data, builder.Finish());

  CheckIfElseOutput(cond, left, right, expected_data);
}

template <typename Type>
class TestCaseWhenNumeric : public ::testing::Test {};

TYPED_TEST_SUITE(TestCaseWhenNumeric, NumericBasedTypes);

Datum MakeStruct(const std::vector<Datum>& conds) {
  EXPECT_OK_AND_ASSIGN(auto result, CallFunction("make_struct", conds));
  return result;
}

TYPED_TEST(TestCaseWhenNumeric, FixedSize) {
  auto type = default_type_instance<TypeParam>();
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, "1");
  auto scalar2 = ScalarFromJSON(type, "2");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, "[3, null, 5, 6]");
  auto values2 = ArrayFromJSON(type, "[7, 8, null, 10]");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
              ArrayFromJSON(type, "[1, 1, 2, null]"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, "[null, null, 1, 1]"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
              ArrayFromJSON(type, "[1, 1, 2, 1]"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, "[3, null, null, null]"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, "[3, null, null, 6]"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, "[null, null, null, 6]"));

  CheckScalar(
      "case_when",
      {MakeStruct(
           {ArrayFromJSON(boolean(),
                          "[true, true, true, false, false, false, null, null, null]"),
            ArrayFromJSON(boolean(),
                          "[true, false, null, true, false, null, true, false, null]")}),
       ArrayFromJSON(type, "[10, 11, 12, 13, 14, 15, 16, 17, 18]"),
       ArrayFromJSON(type, "[20, 21, 22, 23, 24, 25, 26, 27, 28]")},
      ArrayFromJSON(type, "[10, 11, 12, 23, null, null, 26, null, null]"));
  CheckScalar(
      "case_when",
      {MakeStruct(
           {ArrayFromJSON(boolean(),
                          "[true, true, true, false, false, false, null, null, null]"),
            ArrayFromJSON(boolean(),
                          "[true, false, null, true, false, null, true, false, null]")}),
       ArrayFromJSON(type, "[10, 11, 12, 13, 14, 15, 16, 17, 18]"),

       ArrayFromJSON(type, "[20, 21, 22, 23, 24, 25, 26, 27, 28]"),
       ArrayFromJSON(type, "[30, 31, 32, 33, 34, null, 36, 37, null]")},
      ArrayFromJSON(type, "[10, 11, 12, 23, 34, null, 26, 37, null]"));

  // Error cases
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, ::testing::HasSubstr("cond struct must not be null"),
      CallFunction(
          "case_when",
          {Datum(std::make_shared<StructScalar>(struct_({field("", boolean())}))),
           Datum(scalar1)}));
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, ::testing::HasSubstr("cond struct must not have top-level nulls"),
      CallFunction(
          "case_when",
          {Datum(*MakeArrayOfNull(struct_({field("", boolean())}), 4)), Datum(values1)}));
}

TYPED_TEST(TestCaseWhenNumeric, ListOfType) {
  // More minimal test to check type coverage
  auto type = list(default_type_instance<TypeParam>());
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, R"([[1, 2], null, [3, 4, 5], [6, null]])");
  auto values2 = ArrayFromJSON(type, R"([[8, 9, 10], [11], null, [12]])");

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([[1, 2], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([[1, 2], null, null, [6, null]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [6, null]])"));
}

TEST(TestCaseWhen, Null) {
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_arr = ArrayFromJSON(boolean(), "[true, true, false, null]");
  auto scalar = ScalarFromJSON(null(), "null");
  auto array = ArrayFromJSON(null(), "[null, null, null, null]");
  CheckScalar("case_when", {MakeStruct({}), array}, array);
  CheckScalar("case_when", {MakeStruct({cond_false}), array}, array);
  CheckScalar("case_when", {MakeStruct({cond_true}), array, array}, array);
  CheckScalar("case_when", {MakeStruct({cond_arr, cond_true}), array, array}, array);
}

TEST(TestCaseWhen, Boolean) {
  auto type = boolean();
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, "true");
  auto scalar2 = ScalarFromJSON(type, "false");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, "[true, null, true, true]");
  auto values2 = ArrayFromJSON(type, "[false, false, null, false]");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
              ArrayFromJSON(type, "[true, true, false, null]"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, "[null, null, true, true]"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
              ArrayFromJSON(type, "[true, true, false, true]"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, "[true, null, null, null]"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, "[true, null, null, true]"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, "[null, null, null, true]"));
}

TEST(TestCaseWhen, DayTimeInterval) {
  auto type = day_time_interval();
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, "[1, 1]");
  auto scalar2 = ScalarFromJSON(type, "[2, 2]");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, "[[3, 3], null, [5, 5], [6, 6]]");
  auto values2 = ArrayFromJSON(type, "[[7, 7], [8, 8], null, [10, 10]]");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
              ArrayFromJSON(type, "[[1, 1], [1, 1], [2, 2], null]"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, "[null, null, [1, 1], [1, 1]]"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
              ArrayFromJSON(type, "[[1, 1], [1, 1], [2, 2], [1, 1]]"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, "[[3, 3], null, null, null]"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, "[[3, 3], null, null, [6, 6]]"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, "[null, null, null, [6, 6]]"));
}

TEST(TestCaseWhen, Decimal) {
  for (const auto& type :
       std::vector<std::shared_ptr<DataType>>{decimal128(3, 2), decimal256(3, 2)}) {
    auto cond_true = ScalarFromJSON(boolean(), "true");
    auto cond_false = ScalarFromJSON(boolean(), "false");
    auto cond_null = ScalarFromJSON(boolean(), "null");
    auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
    auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
    auto scalar_null = ScalarFromJSON(type, "null");
    auto scalar1 = ScalarFromJSON(type, R"("1.23")");
    auto scalar2 = ScalarFromJSON(type, R"("2.34")");
    auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
    auto values1 = ArrayFromJSON(type, R"(["3.45", null, "5.67", "6.78"])");
    auto values2 = ArrayFromJSON(type, R"(["7.89", "8.90", null, "1.01"])");

    CheckScalar("case_when", {MakeStruct({}), values1}, values1);
    CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

    CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
                *MakeArrayFromScalar(*scalar1, 4));
    CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

    CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
    CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
    CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
    CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
    CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
    CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

    CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
                values1);
    CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
                values_null);
    CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
                values1);
    CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
                values2);
    CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
                values2);
    CheckScalar("case_when",
                {MakeStruct({cond_false, cond_false}), values1, values2, values2},
                values2);

    CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
                ArrayFromJSON(type, R"(["1.23", "1.23", "2.34", null])"));
    CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
    CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
                ArrayFromJSON(type, R"([null, null, "1.23", "1.23"])"));
    CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
                ArrayFromJSON(type, R"(["1.23", "1.23", "2.34", "1.23"])"));

    CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
                ArrayFromJSON(type, R"(["3.45", null, null, null])"));
    CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
                ArrayFromJSON(type, R"(["3.45", null, null, "6.78"])"));
    CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
                ArrayFromJSON(type, R"([null, null, null, "6.78"])"));
  }
}

TEST(TestCaseWhen, FixedSizeBinary) {
  auto type = fixed_size_binary(3);
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"("abc")");
  auto scalar2 = ScalarFromJSON(type, R"("bcd")");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, R"(["cde", null, "def", "efg"])");
  auto values2 = ArrayFromJSON(type, R"(["fgh", "ghi", null, "hij"])");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
              ArrayFromJSON(type, R"(["abc", "abc", "bcd", null])"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, R"([null, null, "abc", "abc"])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
              ArrayFromJSON(type, R"(["abc", "abc", "bcd", "abc"])"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"(["cde", null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"(["cde", null, null, "efg"])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, "efg"])"));
}

template <typename Type>
class TestCaseWhenBinary : public ::testing::Test {};

TYPED_TEST_SUITE(TestCaseWhenBinary, BinaryArrowTypes);

TYPED_TEST(TestCaseWhenBinary, Basics) {
  auto type = default_type_instance<TypeParam>();
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"("aBxYz")");
  auto scalar2 = ScalarFromJSON(type, R"("b")");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, R"(["cDE", null, "degfhi", "efg"])");
  auto values2 = ArrayFromJSON(type, R"(["fghijk", "ghi", null, "hi"])");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
              ArrayFromJSON(type, R"(["aBxYz", "aBxYz", "b", null])"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, R"([null, null, "aBxYz", "aBxYz"])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
              ArrayFromJSON(type, R"(["aBxYz", "aBxYz", "b", "aBxYz"])"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"(["cDE", null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"(["cDE", null, null, "efg"])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, "efg"])"));
}

template <typename Type>
class TestCaseWhenList : public ::testing::Test {};

TYPED_TEST_SUITE(TestCaseWhenList, ListArrowTypes);

TYPED_TEST(TestCaseWhenList, ListOfString) {
  auto type = std::make_shared<TypeParam>(utf8());
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"(["aB", "xYz"])");
  auto scalar2 = ScalarFromJSON(type, R"(["b", null])");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 =
      ArrayFromJSON(type, R"([["cD", "E"], null, ["de", "gf", "hi"], ["ef", "g"]])");
  auto values2 = ArrayFromJSON(type, R"([["f", "ghi", "jk"], ["ghi"], null, ["hi"]])");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar(
      "case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
      ArrayFromJSON(type, R"([["aB", "xYz"], ["aB", "xYz"], ["b", null], null])"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, R"([null, null, ["aB", "xYz"], ["aB", "xYz"]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
              ArrayFromJSON(
                  type, R"([["aB", "xYz"], ["aB", "xYz"], ["b", null], ["aB", "xYz"]])"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([["cD", "E"], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([["cD", "E"], null, null, ["ef", "g"]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, ["ef", "g"]])"));
}

// More minimal tests to check type coverage
TYPED_TEST(TestCaseWhenList, ListOfBool) {
  auto type = std::make_shared<TypeParam>(boolean());
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, R"([[true], null, [false], [false, null]])");
  auto values2 = ArrayFromJSON(type, R"([[false], [false], null, [true]])");

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([[true], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([[true], null, null, [false, null]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [false, null]])"));
}

TYPED_TEST(TestCaseWhenList, ListOfInt) {
  auto type = std::make_shared<TypeParam>(int64());
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, R"([[1, 2], null, [3, 4, 5], [6, null]])");
  auto values2 = ArrayFromJSON(type, R"([[8, 9, 10], [11], null, [12]])");

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([[1, 2], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([[1, 2], null, null, [6, null]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [6, null]])"));
}

TYPED_TEST(TestCaseWhenList, ListOfDayTimeInterval) {
  auto type = std::make_shared<TypeParam>(day_time_interval());
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 =
      ArrayFromJSON(type, R"([[[1, 2]], null, [[3, 4], [5, 0]], [[6, 7], null]])");
  auto values2 = ArrayFromJSON(type, R"([[[8, 9], null], [[11, 12]], null, [[12, 1]]])");

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([[[1, 2]], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([[[1, 2]], null, null, [[6, 7], null]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [[6, 7], null]])"));
}

TYPED_TEST(TestCaseWhenList, ListOfDecimal) {
  for (const auto& decimal_ty :
       std::vector<std::shared_ptr<DataType>>{decimal128(3, 2), decimal256(3, 2)}) {
    auto type = std::make_shared<TypeParam>(decimal_ty);
    auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
    auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
    auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
    auto values1 = ArrayFromJSON(
        type, R"([["1.23", "2.34"], null, ["3.45", "4.56", "5.67"], ["6.78", null]])");
    auto values2 =
        ArrayFromJSON(type, R"([["8.90", "9.01", "1.02"], ["1.12"], null, ["1.23"]])");

    CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
                ArrayFromJSON(type, R"([["1.23", "2.34"], null, null, null])"));
    CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
                ArrayFromJSON(type, R"([["1.23", "2.34"], null, null, ["6.78", null]])"));
    CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
                ArrayFromJSON(type, R"([null, null, null, ["6.78", null]])"));
  }
}

TYPED_TEST(TestCaseWhenList, ListOfFixedSizeBinary) {
  auto type = std::make_shared<TypeParam>(fixed_size_binary(4));
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(
      type, R"([["1.23", "2.34"], null, ["3.45", "4.56", "5.67"], ["6.78", null]])");
  auto values2 =
      ArrayFromJSON(type, R"([["8.90", "9.01", "1.02"], ["1.12"], null, ["1.23"]])");

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([["1.23", "2.34"], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([["1.23", "2.34"], null, null, ["6.78", null]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, ["6.78", null]])"));
}

TYPED_TEST(TestCaseWhenList, ListOfListOfInt) {
  auto type = std::make_shared<TypeParam>(list(int64()));
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 =
      ArrayFromJSON(type, R"([[[1, 2], []], null, [[3, 4, 5]], [[6, null], null]])");
  auto values2 = ArrayFromJSON(type, R"([[[8, 9, 10]], [[11]], null, [[12]]])");

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([[[1, 2], []], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([[[1, 2], []], null, null, [[6, null], null]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [[6, null], null]])"));
}

TEST(TestCaseWhen, Map) {
  auto type = map(int64(), utf8());
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"([[1, "abc"], [2, "de"]])");
  auto scalar2 = ScalarFromJSON(type, R"([[3, "fghi"]])");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 =
      ArrayFromJSON(type, R"([[[4, "kl"]], null, [[5, "mn"]], [[6, "o"], [7, "pq"]]])");
  auto values2 = ArrayFromJSON(type, R"([[[8, "r"], [9, "st"]], [[10, "u"]], null, []])");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar(
      "case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
      ArrayFromJSON(
          type,
          R"([[[1, "abc"], [2, "de"]], [[1, "abc"], [2, "de"]], [[3, "fghi"]], null])"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar(
      "case_when", {MakeStruct({cond1}), scalar_null, scalar1},
      ArrayFromJSON(type,
                    R"([null, null, [[1, "abc"], [2, "de"]], [[1, "abc"], [2, "de"]]])"));
  CheckScalar(
      "case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
      ArrayFromJSON(
          type,
          R"([[[1, "abc"], [2, "de"]], [[1, "abc"], [2, "de"]], [[3, "fghi"]], [[1, "abc"], [2, "de"]]])"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([[[4, "kl"]], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([[[4, "kl"]], null, null, [[6, "o"], [7, "pq"]]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [[6, "o"], [7, "pq"]]])"));
}

TEST(TestCaseWhen, FixedSizeListOfInt) {
  auto type = fixed_size_list(int64(), 2);
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"([1, 2])");
  auto scalar2 = ScalarFromJSON(type, R"([3, null])");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, R"([[4, 5], null, [6, 7], [8,  9]])");
  auto values2 = ArrayFromJSON(type, R"([[10, 11], [12, null], null, [null, 13]])");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
              ArrayFromJSON(type, R"([[1, 2], [1, 2], [3, null], null])"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, R"([null, null, [1, 2], [1, 2]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
              ArrayFromJSON(type, R"([[1, 2], [1, 2], [3, null], [1, 2]])"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([[4, 5], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([[4, 5], null, null, [8, 9]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [8, 9]])"));
}

TEST(TestCaseWhen, FixedSizeListOfString) {
  auto type = fixed_size_list(utf8(), 2);
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"(["aB", "xYz"])");
  auto scalar2 = ScalarFromJSON(type, R"(["b", null])");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 =
      ArrayFromJSON(type, R"([["cD", "E"], null, ["de", "gfhi"], ["ef", "g"]])");
  auto values2 =
      ArrayFromJSON(type, R"([["fghi", "jk"], ["ghi", null], null, [null, "hi"]])");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar(
      "case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
      ArrayFromJSON(type, R"([["aB", "xYz"], ["aB", "xYz"], ["b", null], null])"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, R"([null, null, ["aB", "xYz"], ["aB", "xYz"]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
              ArrayFromJSON(
                  type, R"([["aB", "xYz"], ["aB", "xYz"], ["b", null], ["aB", "xYz"]])"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([["cD", "E"], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([["cD", "E"], null, null, ["ef", "g"]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, ["ef", "g"]])"));
}

TEST(TestCaseWhen, StructOfInt) {
  auto type = struct_({field("a", uint32()), field("b", int64())});
  auto cond_true = ScalarFromJSON(boolean(), "true");
  auto cond_false = ScalarFromJSON(boolean(), "false");
  auto cond_null = ScalarFromJSON(boolean(), "null");
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"([1, -2])");
  auto scalar2 = ScalarFromJSON(type, R"([null, 3])");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, R"([[4, null], null, [5, -6], [7, -8]])");
  auto values2 = ArrayFromJSON(type, R"([[9, 10], [11, -12], null, [null, null]])");

  CheckScalar("case_when", {MakeStruct({}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

  CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
              *MakeArrayFromScalar(*scalar1, 4));
  CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

  CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
  CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
  CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
  CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
              values_null);
  CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
              values1);
  CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
              values2);
  CheckScalar("case_when",
              {MakeStruct({cond_false, cond_false}), values1, values2, values2}, values2);

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
              ArrayFromJSON(type, R"([[1, -2], [1, -2], [null, 3], null])"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, R"([null, null, [1, -2], [1, -2]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
              ArrayFromJSON(type, R"([[1, -2], [1, -2], [null, 3], [1, -2]])"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([[4, null], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([[4, null], null, null, [7, -8]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [7, -8]])"));
}

TEST(TestCaseWhen, StructOfString) {
  // More minimal test to check type coverage
  auto type = struct_({field("a", utf8()), field("b", large_utf8())});
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"(["a", "bc"])");
  auto scalar2 = ScalarFromJSON(type, R"([null, "d"])");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 =
      ArrayFromJSON(type, R"([["efg", null], null, [null, null], [null, "hi"]])");
  auto values2 =
      ArrayFromJSON(type, R"([["j", "k"], [null, "lmnop"], null, ["qr", "stu"]])");
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
              ArrayFromJSON(type, R"([["a", "bc"], ["a", "bc"], [null, "d"], null])"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
              ArrayFromJSON(type, R"([null, null, ["a", "bc"], ["a", "bc"]])"));
  CheckScalar(
      "case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
      ArrayFromJSON(type, R"([["a", "bc"], ["a", "bc"], [null, "d"], ["a", "bc"]])"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([["efg", null], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([["efg", null], null, null, [null, "hi"]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [null, "hi"]])"));
}

TEST(TestCaseWhen, StructOfListOfInt) {
  // More minimal test to check type coverage
  auto type = struct_({field("a", utf8()), field("b", list(int64()))});
  auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
  auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"([null, [1, null]])");
  auto scalar2 = ScalarFromJSON(type, R"(["b", null])");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 =
      ArrayFromJSON(type, R"([["efg", null], null, [null, null], [null, [null, 1]]])");
  auto values2 =
      ArrayFromJSON(type, R"([["j", [2, 3]], [null, [4, 5, 6]], null, ["qr", [7]]])");
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
              ArrayFromJSON(
                  type, R"([[null, [1, null]], [null, [1, null]], ["b", null], null])"));
  CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
  CheckScalar(
      "case_when", {MakeStruct({cond1}), scalar_null, scalar1},
      ArrayFromJSON(type, R"([null, null, [null, [1, null]], [null, [1, null]]])"));
  CheckScalar(
      "case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
      ArrayFromJSON(
          type,
          R"([[null, [1, null]], [null, [1, null]], ["b", null], [null, [1, null]]])"));

  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
              ArrayFromJSON(type, R"([["efg", null], null, null, null])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
              ArrayFromJSON(type, R"([["efg", null], null, null, [null, [null, 1]]])"));
  CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
              ArrayFromJSON(type, R"([null, null, null, [null, [null, 1]]])"));
}

TEST(TestCaseWhen, UnionBoolString) {
  for (const auto& type : std::vector<std::shared_ptr<DataType>>{
           sparse_union({field("a", boolean()), field("b", utf8())}, {2, 7}),
           dense_union({field("a", boolean()), field("b", utf8())}, {2, 7})}) {
    ARROW_SCOPED_TRACE(type->ToString());
    auto cond_true = ScalarFromJSON(boolean(), "true");
    auto cond_false = ScalarFromJSON(boolean(), "false");
    auto cond_null = ScalarFromJSON(boolean(), "null");
    auto cond1 = ArrayFromJSON(boolean(), "[true, true, null, null]");
    auto cond2 = ArrayFromJSON(boolean(), "[true, false, true, null]");
    auto scalar_null = ScalarFromJSON(type, "null");
    auto scalar1 = ScalarFromJSON(type, R"([2, null])");
    auto scalar2 = ScalarFromJSON(type, R"([7, "foo"])");
    auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
    auto values1 = ArrayFromJSON(type, R"([[2, true], null, [7, "bar"], [7, "baz"]])");
    auto values2 = ArrayFromJSON(type, R"([[7, "spam"], [2, null], null, [7, null]])");

    CheckScalar("case_when", {MakeStruct({}), values1}, values1);
    CheckScalar("case_when", {MakeStruct({}), values_null}, values_null);

    CheckScalar("case_when", {MakeStruct({cond_true}), scalar1, values1},
                *MakeArrayFromScalar(*scalar1, 4));
    CheckScalar("case_when", {MakeStruct({cond_false}), scalar1, values1}, values1);

    CheckScalar("case_when", {MakeStruct({cond_true}), values1}, values1);
    CheckScalar("case_when", {MakeStruct({cond_false}), values1}, values_null);
    CheckScalar("case_when", {MakeStruct({cond_null}), values1}, values_null);
    CheckScalar("case_when", {MakeStruct({cond_true}), values1, values2}, values1);
    CheckScalar("case_when", {MakeStruct({cond_false}), values1, values2}, values2);
    CheckScalar("case_when", {MakeStruct({cond_null}), values1, values2}, values2);

    CheckScalar("case_when", {MakeStruct({cond_true, cond_true}), values1, values2},
                values1);
    CheckScalar("case_when", {MakeStruct({cond_false, cond_false}), values1, values2},
                values_null);
    CheckScalar("case_when", {MakeStruct({cond_true, cond_false}), values1, values2},
                values1);
    CheckScalar("case_when", {MakeStruct({cond_false, cond_true}), values1, values2},
                values2);
    CheckScalar("case_when", {MakeStruct({cond_null, cond_true}), values1, values2},
                values2);
    CheckScalar("case_when",
                {MakeStruct({cond_false, cond_false}), values1, values2, values2},
                values2);

    CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2},
                ArrayFromJSON(type, R"([[2, null], [2, null], [7, "foo"], null])"));
    CheckScalar("case_when", {MakeStruct({cond1}), scalar_null}, values_null);
    CheckScalar("case_when", {MakeStruct({cond1}), scalar_null, scalar1},
                ArrayFromJSON(type, R"([null, null, [2, null], [2, null]])"));
    CheckScalar("case_when", {MakeStruct({cond1, cond2}), scalar1, scalar2, scalar1},
                ArrayFromJSON(type, R"([[2, null], [2, null], [7, "foo"], [2, null]])"));

    CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2},
                ArrayFromJSON(type, R"([[2, true], null, null, null])"));
    CheckScalar("case_when", {MakeStruct({cond1, cond2}), values1, values2, values1},
                ArrayFromJSON(type, R"([[2, true], null, null, [7, "baz"]])"));
    CheckScalar("case_when", {MakeStruct({cond1, cond2}), values_null, values2, values1},
                ArrayFromJSON(type, R"([null, null, null, [7, "baz"]])"));
  }
}

TEST(TestCaseWhen, DispatchBest) {
  CheckDispatchBest("case_when", {struct_({field("", boolean())}), int64(), int32()},
                    {struct_({field("", boolean())}), int64(), int64()});

  ASSERT_RAISES(Invalid, CallFunction("case_when", {}));
  // Too many/too few conditions
  ASSERT_RAISES(
      Invalid, CallFunction("case_when", {MakeStruct({ArrayFromJSON(boolean(), "[]")})}));
  ASSERT_RAISES(Invalid,
                CallFunction("case_when", {MakeStruct({}), ArrayFromJSON(int64(), "[]"),
                                           ArrayFromJSON(int64(), "[]")}));
  // Conditions must be struct of boolean
  ASSERT_RAISES(TypeError,
                CallFunction("case_when", {MakeStruct({ArrayFromJSON(int64(), "[]")}),
                                           ArrayFromJSON(int64(), "[]")}));
  ASSERT_RAISES(TypeError, CallFunction("case_when", {ArrayFromJSON(boolean(), "[true]"),
                                                      ArrayFromJSON(int32(), "[0]")}));
  // Values must have compatible types
  ASSERT_RAISES(NotImplemented,
                CallFunction("case_when", {MakeStruct({ArrayFromJSON(boolean(), "[]")}),
                                           ArrayFromJSON(int64(), "[]"),
                                           ArrayFromJSON(utf8(), "[]")}));
}

template <typename Type>
class TestCoalesceNumeric : public ::testing::Test {};
template <typename Type>
class TestCoalesceBinary : public ::testing::Test {};

TYPED_TEST_SUITE(TestCoalesceNumeric, NumericBasedTypes);
TYPED_TEST_SUITE(TestCoalesceBinary, BinaryArrowTypes);

TYPED_TEST(TestCoalesceNumeric, FixedSize) {
  auto type = default_type_instance<TypeParam>();
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, "20");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, "[null, 10, 11, 12]");
  auto values2 = ArrayFromJSON(type, "[13, 14, 15, 16]");
  auto values3 = ArrayFromJSON(type, "[17, 18, 19, null]");
  // N.B. all-scalar cases are checked in CheckScalar
  CheckScalar("coalesce", {values_null}, values_null);
  CheckScalar("coalesce", {values_null, scalar1},
              ArrayFromJSON(type, "[20, 20, 20, 20]"));
  CheckScalar("coalesce", {values_null, values1}, values1);
  CheckScalar("coalesce", {values_null, values2}, values2);
  CheckScalar("coalesce", {values1, values_null}, values1);
  CheckScalar("coalesce", {values2, values_null}, values2);
  CheckScalar("coalesce", {scalar_null, values1}, values1);
  CheckScalar("coalesce", {values1, scalar_null}, values1);
  CheckScalar("coalesce", {values2, values1, values_null}, values2);
  CheckScalar("coalesce", {values1, scalar1}, ArrayFromJSON(type, "[20, 10, 11, 12]"));
  CheckScalar("coalesce", {values1, values2}, ArrayFromJSON(type, "[13, 10, 11, 12]"));
  CheckScalar("coalesce", {values1, values2, values3},
              ArrayFromJSON(type, "[13, 10, 11, 12]"));
  CheckScalar("coalesce", {scalar1, values1}, ArrayFromJSON(type, "[20, 20, 20, 20]"));
}

TYPED_TEST(TestCoalesceBinary, Basics) {
  auto type = default_type_instance<TypeParam>();
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"("a")");
  auto values_null = ArrayFromJSON(type, R"([null, null, null, null])");
  auto values1 = ArrayFromJSON(type, R"([null, "bc", "def", "ghij"])");
  auto values2 = ArrayFromJSON(type, R"(["klmno", "p", "qr", "stu"])");
  auto values3 = ArrayFromJSON(type, R"(["vwxy", "zabc", "d", null])");
  // N.B. all-scalar cases are checked in CheckScalar
  CheckScalar("coalesce", {values_null}, values_null);
  CheckScalar("coalesce", {values_null, scalar1},
              ArrayFromJSON(type, R"(["a", "a", "a", "a"])"));
  CheckScalar("coalesce", {values_null, values1}, values1);
  CheckScalar("coalesce", {values_null, values2}, values2);
  CheckScalar("coalesce", {values1, values_null}, values1);
  CheckScalar("coalesce", {values2, values_null}, values2);
  CheckScalar("coalesce", {scalar_null, values1}, values1);
  CheckScalar("coalesce", {values1, scalar_null}, values1);
  CheckScalar("coalesce", {values2, values1, values_null}, values2);
  CheckScalar("coalesce", {values1, scalar1},
              ArrayFromJSON(type, R"(["a", "bc", "def", "ghij"])"));
  CheckScalar("coalesce", {values1, values2},
              ArrayFromJSON(type, R"(["klmno", "bc", "def", "ghij"])"));
  CheckScalar("coalesce", {values1, values2, values3},
              ArrayFromJSON(type, R"(["klmno", "bc", "def", "ghij"])"));
  CheckScalar("coalesce", {scalar1, values1},
              ArrayFromJSON(type, R"(["a", "a", "a", "a"])"));
}

TEST(TestCoalesce, Null) {
  auto type = null();
  auto scalar_null = ScalarFromJSON(type, "null");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  CheckScalar("coalesce", {values_null}, values_null);
  CheckScalar("coalesce", {values_null, scalar_null}, values_null);
}

TEST(TestCoalesce, Boolean) {
  auto type = boolean();
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, "false");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, "[null, true, false, true]");
  auto values2 = ArrayFromJSON(type, "[true, false, true, false]");
  auto values3 = ArrayFromJSON(type, "[false, true, false, null]");
  CheckScalar("coalesce", {values_null}, values_null);
  CheckScalar("coalesce", {values_null, scalar1},
              ArrayFromJSON(type, "[false, false, false, false]"));
  CheckScalar("coalesce", {values_null, values1}, values1);
  CheckScalar("coalesce", {values_null, values2}, values2);
  CheckScalar("coalesce", {values1, values_null}, values1);
  CheckScalar("coalesce", {values2, values_null}, values2);
  CheckScalar("coalesce", {scalar_null, values1}, values1);
  CheckScalar("coalesce", {values1, scalar_null}, values1);
  CheckScalar("coalesce", {values2, values1, values_null}, values2);
  CheckScalar("coalesce", {values1, scalar1},
              ArrayFromJSON(type, "[false, true, false, true]"));
  CheckScalar("coalesce", {values1, values2},
              ArrayFromJSON(type, "[true, true, false, true]"));
  CheckScalar("coalesce", {values1, values2, values3},
              ArrayFromJSON(type, "[true, true, false, true]"));
  CheckScalar("coalesce", {scalar1, values1},
              ArrayFromJSON(type, "[false, false, false, false]"));
}

TEST(TestCoalesce, DayTimeInterval) {
  auto type = day_time_interval();
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, "[1, 2]");
  auto values_null = ArrayFromJSON(type, "[null, null, null, null]");
  auto values1 = ArrayFromJSON(type, "[null, [3, 4], [5, 6], [7, 8]]");
  auto values2 = ArrayFromJSON(type, "[[9, 10], [11, 12], [13, 14], [15, 16]]");
  auto values3 = ArrayFromJSON(type, "[[17, 18], [19, 20], [21, 22], null]");
  // N.B. all-scalar cases are checked in CheckScalar
  CheckScalar("coalesce", {values_null}, values_null);
  CheckScalar("coalesce", {values_null, scalar1},
              ArrayFromJSON(type, "[[1, 2], [1, 2], [1, 2], [1, 2]]"));
  CheckScalar("coalesce", {values_null, values1}, values1);
  CheckScalar("coalesce", {values_null, values2}, values2);
  CheckScalar("coalesce", {values1, values_null}, values1);
  CheckScalar("coalesce", {values2, values_null}, values2);
  CheckScalar("coalesce", {scalar_null, values1}, values1);
  CheckScalar("coalesce", {values1, scalar_null}, values1);
  CheckScalar("coalesce", {values2, values1, values_null}, values2);
  CheckScalar("coalesce", {values1, scalar1},
              ArrayFromJSON(type, "[[1, 2], [3, 4], [5, 6], [7, 8]]"));
  CheckScalar("coalesce", {values1, values2},
              ArrayFromJSON(type, "[[9, 10], [3, 4], [5, 6], [7, 8]]"));
  CheckScalar("coalesce", {values1, values2, values3},
              ArrayFromJSON(type, "[[9, 10], [3, 4], [5, 6], [7, 8]]"));
  CheckScalar("coalesce", {scalar1, values1},
              ArrayFromJSON(type, "[[1, 2], [1, 2], [1, 2], [1, 2]]"));
}

TEST(TestCoalesce, Decimal) {
  for (const auto& type :
       std::vector<std::shared_ptr<DataType>>{decimal128(3, 2), decimal256(3, 2)}) {
    auto scalar_null = ScalarFromJSON(type, "null");
    auto scalar1 = ScalarFromJSON(type, R"("1.23")");
    auto values_null = ArrayFromJSON(type, R"([null, null, null, null])");
    auto values1 = ArrayFromJSON(type, R"([null, "4.56", "7.89", "1.34"])");
    auto values2 = ArrayFromJSON(type, R"(["1.45", "2.34", "3.45", "4.56"])");
    auto values3 = ArrayFromJSON(type, R"(["5.67", "6.78", "7.91", null])");
    CheckScalar("coalesce", {values_null}, values_null);
    CheckScalar("coalesce", {values_null, scalar1},
                ArrayFromJSON(type, R"(["1.23", "1.23", "1.23", "1.23"])"));
    CheckScalar("coalesce", {values_null, values1}, values1);
    CheckScalar("coalesce", {values_null, values2}, values2);
    CheckScalar("coalesce", {values1, values_null}, values1);
    CheckScalar("coalesce", {values2, values_null}, values2);
    CheckScalar("coalesce", {scalar_null, values1}, values1);
    CheckScalar("coalesce", {values1, scalar_null}, values1);
    CheckScalar("coalesce", {values2, values1, values_null}, values2);
    CheckScalar("coalesce", {values1, scalar1},
                ArrayFromJSON(type, R"(["1.23", "4.56", "7.89", "1.34"])"));
    CheckScalar("coalesce", {values1, values2},
                ArrayFromJSON(type, R"(["1.45", "4.56", "7.89", "1.34"])"));
    CheckScalar("coalesce", {values1, values2, values3},
                ArrayFromJSON(type, R"(["1.45", "4.56", "7.89", "1.34"])"));
    CheckScalar("coalesce", {scalar1, values1},
                ArrayFromJSON(type, R"(["1.23", "1.23", "1.23", "1.23"])"));
  }
}

TEST(TestCoalesce, FixedSizeBinary) {
  auto type = fixed_size_binary(3);
  auto scalar_null = ScalarFromJSON(type, "null");
  auto scalar1 = ScalarFromJSON(type, R"("abc")");
  auto values_null = ArrayFromJSON(type, R"([null, null, null, null])");
  auto values1 = ArrayFromJSON(type, R"([null, "def", "ghi", "jkl"])");
  auto values2 = ArrayFromJSON(type, R"(["mno", "pqr", "stu", "vwx"])");
  auto values3 = ArrayFromJSON(type, R"(["yza", "bcd", "efg", null])");
  CheckScalar("coalesce", {values_null}, values_null);
  CheckScalar("coalesce", {values_null, scalar1},
              ArrayFromJSON(type, R"(["abc", "abc", "abc", "abc"])"));
  CheckScalar("coalesce", {values_null, values1}, values1);
  CheckScalar("coalesce", {values_null, values2}, values2);
  CheckScalar("coalesce", {values1, values_null}, values1);
  CheckScalar("coalesce", {values2, values_null}, values2);
  CheckScalar("coalesce", {scalar_null, values1}, values1);
  CheckScalar("coalesce", {values1, scalar_null}, values1);
  CheckScalar("coalesce", {values2, values1, values_null}, values2);
  CheckScalar("coalesce", {values1, scalar1},
              ArrayFromJSON(type, R"(["abc", "def", "ghi", "jkl"])"));
  CheckScalar("coalesce", {values1, values2},
              ArrayFromJSON(type, R"(["mno", "def", "ghi", "jkl"])"));
  CheckScalar("coalesce", {values1, values2, values3},
              ArrayFromJSON(type, R"(["mno", "def", "ghi", "jkl"])"));
  CheckScalar("coalesce", {scalar1, values1},
              ArrayFromJSON(type, R"(["abc", "abc", "abc", "abc"])"));
}

template <typename Type>
class TestChooseNumeric : public ::testing::Test {};
template <typename Type>
class TestChooseBinary : public ::testing::Test {};

TYPED_TEST_SUITE(TestChooseNumeric, NumericBasedTypes);
TYPED_TEST_SUITE(TestChooseBinary, BinaryArrowTypes);

TYPED_TEST(TestChooseNumeric, FixedSize) {
  auto type = default_type_instance<TypeParam>();
  auto indices1 = ArrayFromJSON(int64(), "[0, 1, 0, 1, null]");
  auto values1 = ArrayFromJSON(type, "[10, 11, null, null, 14]");
  auto values2 = ArrayFromJSON(type, "[20, 21, null, null, 24]");
  auto nulls = ArrayFromJSON(type, "[null, null, null, null, null]");
  CheckScalar("choose", {indices1, values1, values2},
              ArrayFromJSON(type, "[10, 21, null, null, null]"));
  CheckScalar("choose", {indices1, ScalarFromJSON(type, "1"), values1},
              ArrayFromJSON(type, "[1, 11, 1, null, null]"));
  // Mixed scalar and array (note CheckScalar checks all-scalar cases for us)
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), values1, values2}, values1);
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), values1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar1 = ScalarFromJSON(type, "42");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar1, values2},
              *MakeArrayFromScalar(*scalar1, 5));
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), scalar1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar_null = ScalarFromJSON(type, "null");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar_null, values2},
              *MakeArrayOfNull(type, 5));
}

TYPED_TEST(TestChooseBinary, Basics) {
  auto type = default_type_instance<TypeParam>();
  auto indices1 = ArrayFromJSON(int64(), "[0, 1, 0, 1, null]");
  auto values1 = ArrayFromJSON(type, R"(["a", "bc", null, null, "def"])");
  auto values2 = ArrayFromJSON(type, R"(["ghij", "klmno", null, null, "pqrstu"])");
  auto nulls = ArrayFromJSON(type, "[null, null, null, null, null]");
  CheckScalar("choose", {indices1, values1, values2},
              ArrayFromJSON(type, R"(["a", "klmno", null, null, null])"));
  CheckScalar("choose", {indices1, ScalarFromJSON(type, R"("foo")"), values1},
              ArrayFromJSON(type, R"(["foo", "bc", "foo", null, null])"));
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), values1, values2}, values1);
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), values1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar1 = ScalarFromJSON(type, R"("abcd")");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar1, values2},
              *MakeArrayFromScalar(*scalar1, 5));
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), scalar1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar_null = ScalarFromJSON(type, "null");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar_null, values2},
              *MakeArrayOfNull(type, 5));
}

TEST(TestChoose, Null) {
  auto type = null();
  auto indices1 = ArrayFromJSON(int64(), "[0, 1, 0, 1, null]");
  auto nulls = *MakeArrayOfNull(type, 5);
  CheckScalar("choose", {indices1, nulls, nulls}, nulls);
  CheckScalar("choose", {indices1, MakeNullScalar(type), nulls}, nulls);
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), nulls, nulls}, nulls);
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), nulls, nulls}, nulls);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), nulls, nulls}, nulls);
  auto scalar_null = ScalarFromJSON(type, "null");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar_null, nulls}, nulls);
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), scalar_null, nulls}, nulls);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), nulls, nulls}, nulls);
}

TEST(TestChoose, Boolean) {
  auto type = boolean();
  auto indices1 = ArrayFromJSON(int64(), "[0, 1, 0, 1, null]");
  auto values1 = ArrayFromJSON(type, "[true, true, null, null, true]");
  auto values2 = ArrayFromJSON(type, "[false, false, null, null, false]");
  auto nulls = ArrayFromJSON(type, "[null, null, null, null, null]");
  CheckScalar("choose", {indices1, values1, values2},
              ArrayFromJSON(type, "[true, false, null, null, null]"));
  CheckScalar("choose", {indices1, ScalarFromJSON(type, "false"), values1},
              ArrayFromJSON(type, "[false, true, false, null, null]"));
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), values1, values2}, values1);
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), values1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar1 = ScalarFromJSON(type, "true");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar1, values2},
              *MakeArrayFromScalar(*scalar1, 5));
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), scalar1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar_null = ScalarFromJSON(type, "null");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar_null, values2},
              *MakeArrayOfNull(type, 5));
}

TEST(TestChoose, DayTimeInterval) {
  auto type = day_time_interval();
  auto indices1 = ArrayFromJSON(int64(), "[0, 1, 0, 1, null]");
  auto values1 = ArrayFromJSON(type, "[[10, 1], [10, 1], null, null, [10, 1]]");
  auto values2 = ArrayFromJSON(type, "[[2, 20], [2, 20], null, null, [2, 20]]");
  auto nulls = ArrayFromJSON(type, "[null, null, null, null, null]");
  CheckScalar("choose", {indices1, values1, values2},
              ArrayFromJSON(type, "[[10, 1], [2, 20], null, null, null]"));
  CheckScalar("choose", {indices1, ScalarFromJSON(type, "[1, 2]"), values1},
              ArrayFromJSON(type, "[[1, 2], [10, 1], [1, 2], null, null]"));
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), values1, values2}, values1);
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), values1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar1 = ScalarFromJSON(type, "[10, 1]");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar1, values2},
              *MakeArrayFromScalar(*scalar1, 5));
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), scalar1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar_null = ScalarFromJSON(type, "null");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar_null, values2},
              *MakeArrayOfNull(type, 5));
}

TEST(TestChoose, Decimal) {
  for (const auto& type : {decimal128(3, 2), decimal256(3, 2)}) {
    auto indices1 = ArrayFromJSON(int64(), "[0, 1, 0, 1, null]");
    auto values1 = ArrayFromJSON(type, R"(["1.23", "1.24", null, null, "1.25"])");
    auto values2 = ArrayFromJSON(type, R"(["4.56", "4.57", null, null, "4.58"])");
    auto nulls = ArrayFromJSON(type, "[null, null, null, null, null]");
    CheckScalar("choose", {indices1, values1, values2},
                ArrayFromJSON(type, R"(["1.23", "4.57", null, null, null])"));
    CheckScalar("choose", {indices1, ScalarFromJSON(type, R"("2.34")"), values1},
                ArrayFromJSON(type, R"(["2.34", "1.24", "2.34", null, null])"));
    CheckScalar("choose", {ScalarFromJSON(int64(), "0"), values1, values2}, values1);
    CheckScalar("choose", {ScalarFromJSON(int64(), "1"), values1, values2}, values2);
    CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
    auto scalar1 = ScalarFromJSON(type, R"("1.23")");
    CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar1, values2},
                *MakeArrayFromScalar(*scalar1, 5));
    CheckScalar("choose", {ScalarFromJSON(int64(), "1"), scalar1, values2}, values2);
    CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
    auto scalar_null = ScalarFromJSON(type, "null");
    CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar_null, values2},
                *MakeArrayOfNull(type, 5));
  }
}

TEST(TestChoose, FixedSizeBinary) {
  auto type = fixed_size_binary(3);
  auto indices1 = ArrayFromJSON(int64(), "[0, 1, 0, 1, null]");
  auto values1 = ArrayFromJSON(type, R"(["abc", "abd", null, null, "abe"])");
  auto values2 = ArrayFromJSON(type, R"(["def", "deg", null, null, "deh"])");
  auto nulls = ArrayFromJSON(type, "[null, null, null, null, null]");
  CheckScalar("choose", {indices1, values1, values2},
              ArrayFromJSON(type, R"(["abc", "deg", null, null, null])"));
  CheckScalar("choose", {indices1, ScalarFromJSON(type, R"("xyz")"), values1},
              ArrayFromJSON(type, R"(["xyz", "abd", "xyz", null, null])"));
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), values1, values2}, values1);
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), values1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar1 = ScalarFromJSON(type, R"("abc")");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar1, values2},
              *MakeArrayFromScalar(*scalar1, 5));
  CheckScalar("choose", {ScalarFromJSON(int64(), "1"), scalar1, values2}, values2);
  CheckScalar("choose", {ScalarFromJSON(int64(), "null"), values1, values2}, nulls);
  auto scalar_null = ScalarFromJSON(type, "null");
  CheckScalar("choose", {ScalarFromJSON(int64(), "0"), scalar_null, values2},
              *MakeArrayOfNull(type, 5));
}

TEST(TestChooseKernel, DispatchBest) {
  ASSERT_OK_AND_ASSIGN(auto function, GetFunctionRegistry()->GetFunction("choose"));
  auto Check = [&](std::vector<ValueDescr> original_values) {
    auto values = original_values;
    ARROW_EXPECT_OK(function->DispatchBest(&values));
    return values;
  };

  // Since DispatchBest for this kernel pulls tricks, we can't compare it to DispatchExact
  // as CheckDispatchBest does
  for (auto ty :
       {int8(), int16(), int32(), int64(), uint8(), uint16(), uint32(), uint64()}) {
    // Index always promoted to int64
    EXPECT_EQ((std::vector<ValueDescr>{int64(), ty}), Check({ty, ty}));
    EXPECT_EQ((std::vector<ValueDescr>{int64(), int64(), int64()}),
              Check({ty, ty, int64()}));
  }
  // Other arguments promoted separately from index
  EXPECT_EQ((std::vector<ValueDescr>{int64(), int32(), int32()}),
            Check({int8(), int32(), uint8()}));
}

TEST(TestChooseKernel, Errors) {
  ASSERT_RAISES(Invalid, CallFunction("choose", {}));
  ASSERT_RAISES(Invalid, CallFunction("choose", {ArrayFromJSON(int64(), "[]")}));
  ASSERT_RAISES(Invalid, CallFunction("choose", {ArrayFromJSON(utf8(), "[\"a\"]"),
                                                 ArrayFromJSON(int64(), "[0]")}));
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      IndexError, ::testing::HasSubstr("choose: index 1 out of range"),
      CallFunction("choose",
                   {ArrayFromJSON(int64(), "[1]"), ArrayFromJSON(int32(), "[0]")}));
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      IndexError, ::testing::HasSubstr("choose: index -1 out of range"),
      CallFunction("choose",
                   {ArrayFromJSON(int64(), "[-1]"), ArrayFromJSON(int32(), "[0]")}));
}

}  // namespace compute
}  // namespace arrow
