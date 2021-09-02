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

#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "arrow/buffer_builder.h"
#include "arrow/compute/api_aggregate.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec/key_compare.h"
#include "arrow/compute/exec/key_encode.h"
#include "arrow/compute/exec/key_hash.h"
#include "arrow/compute/exec/key_map.h"
#include "arrow/compute/exec/util.h"
#include "arrow/compute/exec_internal.h"
#include "arrow/compute/kernel.h"
#include "arrow/compute/kernels/aggregate_internal.h"
#include "arrow/compute/kernels/aggregate_var_std_internal.h"
#include "arrow/compute/kernels/common.h"
#include "arrow/compute/kernels/util_internal.h"
#include "arrow/util/bit_run_reader.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/bitmap_writer.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/cpu_info.h"
#include "arrow/util/int128_internal.h"
#include "arrow/util/make_unique.h"
#include "arrow/util/task_group.h"
#include "arrow/util/tdigest.h"
#include "arrow/util/thread_pool.h"
#include "arrow/visitor_inline.h"

namespace arrow {

using internal::checked_cast;
using internal::FirstTimeBitmapWriter;

namespace compute {
namespace internal {
namespace {

struct KeyEncoder {
  // the first byte of an encoded key is used to indicate nullity
  static constexpr bool kExtraByteForNull = true;

  static constexpr uint8_t kNullByte = 1;
  static constexpr uint8_t kValidByte = 0;

  virtual ~KeyEncoder() = default;

  virtual void AddLength(const ArrayData&, int32_t* lengths) = 0;

  virtual Status Encode(const ArrayData&, uint8_t** encoded_bytes) = 0;

  virtual Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes,
                                                    int32_t length, MemoryPool*) = 0;

  // extract the null bitmap from the leading nullity bytes of encoded keys
  static Status DecodeNulls(MemoryPool* pool, int32_t length, uint8_t** encoded_bytes,
                            std::shared_ptr<Buffer>* null_bitmap, int32_t* null_count) {
    // first count nulls to determine if a null bitmap is necessary
    *null_count = 0;
    for (int32_t i = 0; i < length; ++i) {
      *null_count += (encoded_bytes[i][0] == kNullByte);
    }

    if (*null_count > 0) {
      ARROW_ASSIGN_OR_RAISE(*null_bitmap, AllocateBitmap(length, pool));
      uint8_t* validity = (*null_bitmap)->mutable_data();

      FirstTimeBitmapWriter writer(validity, 0, length);
      for (int32_t i = 0; i < length; ++i) {
        if (encoded_bytes[i][0] == kValidByte) {
          writer.Set();
        } else {
          writer.Clear();
        }
        writer.Next();
        encoded_bytes[i] += 1;
      }
      writer.Finish();
    } else {
      for (int32_t i = 0; i < length; ++i) {
        encoded_bytes[i] += 1;
      }
    }
    return Status ::OK();
  }
};

struct BooleanKeyEncoder : KeyEncoder {
  static constexpr int kByteWidth = 1;

  void AddLength(const ArrayData& data, int32_t* lengths) override {
    for (int64_t i = 0; i < data.length; ++i) {
      lengths[i] += kByteWidth + kExtraByteForNull;
    }
  }

  Status Encode(const ArrayData& data, uint8_t** encoded_bytes) override {
    VisitArrayDataInline<BooleanType>(
        data,
        [&](bool value) {
          auto& encoded_ptr = *encoded_bytes++;
          *encoded_ptr++ = kValidByte;
          *encoded_ptr++ = value;
        },
        [&] {
          auto& encoded_ptr = *encoded_bytes++;
          *encoded_ptr++ = kNullByte;
          *encoded_ptr++ = 0;
        });
    return Status::OK();
  }

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override {
    std::shared_ptr<Buffer> null_buf;
    int32_t null_count;
    RETURN_NOT_OK(DecodeNulls(pool, length, encoded_bytes, &null_buf, &null_count));

    ARROW_ASSIGN_OR_RAISE(auto key_buf, AllocateBitmap(length, pool));

    uint8_t* raw_output = key_buf->mutable_data();
    for (int32_t i = 0; i < length; ++i) {
      auto& encoded_ptr = encoded_bytes[i];
      BitUtil::SetBitTo(raw_output, i, encoded_ptr[0] != 0);
      encoded_ptr += 1;
    }

    return ArrayData::Make(boolean(), length, {std::move(null_buf), std::move(key_buf)},
                           null_count);
  }
};

struct FixedWidthKeyEncoder : KeyEncoder {
  explicit FixedWidthKeyEncoder(std::shared_ptr<DataType> type)
      : type_(std::move(type)),
        byte_width_(checked_cast<const FixedWidthType&>(*type_).bit_width() / 8) {}

  void AddLength(const ArrayData& data, int32_t* lengths) override {
    for (int64_t i = 0; i < data.length; ++i) {
      lengths[i] += byte_width_ + kExtraByteForNull;
    }
  }

  Status Encode(const ArrayData& data, uint8_t** encoded_bytes) override {
    ArrayData viewed(fixed_size_binary(byte_width_), data.length, data.buffers,
                     data.null_count, data.offset);

    VisitArrayDataInline<FixedSizeBinaryType>(
        viewed,
        [&](util::string_view bytes) {
          auto& encoded_ptr = *encoded_bytes++;
          *encoded_ptr++ = kValidByte;
          memcpy(encoded_ptr, bytes.data(), byte_width_);
          encoded_ptr += byte_width_;
        },
        [&] {
          auto& encoded_ptr = *encoded_bytes++;
          *encoded_ptr++ = kNullByte;
          memset(encoded_ptr, 0, byte_width_);
          encoded_ptr += byte_width_;
        });
    return Status::OK();
  }

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override {
    std::shared_ptr<Buffer> null_buf;
    int32_t null_count;
    RETURN_NOT_OK(DecodeNulls(pool, length, encoded_bytes, &null_buf, &null_count));

    ARROW_ASSIGN_OR_RAISE(auto key_buf, AllocateBuffer(length * byte_width_, pool));

    uint8_t* raw_output = key_buf->mutable_data();
    for (int32_t i = 0; i < length; ++i) {
      auto& encoded_ptr = encoded_bytes[i];
      std::memcpy(raw_output, encoded_ptr, byte_width_);
      encoded_ptr += byte_width_;
      raw_output += byte_width_;
    }

    return ArrayData::Make(type_, length, {std::move(null_buf), std::move(key_buf)},
                           null_count);
  }

  std::shared_ptr<DataType> type_;
  int byte_width_;
};

struct DictionaryKeyEncoder : FixedWidthKeyEncoder {
  DictionaryKeyEncoder(std::shared_ptr<DataType> type, MemoryPool* pool)
      : FixedWidthKeyEncoder(std::move(type)), pool_(pool) {}

  Status Encode(const ArrayData& data, uint8_t** encoded_bytes) override {
    auto dict = MakeArray(data.dictionary);
    if (dictionary_) {
      if (!dictionary_->Equals(dict)) {
        // TODO(bkietz) unify if necessary. For now, just error if any batch's dictionary
        // differs from the first we saw for this key
        return Status::NotImplemented("Unifying differing dictionaries");
      }
    } else {
      dictionary_ = std::move(dict);
    }
    return FixedWidthKeyEncoder::Encode(data, encoded_bytes);
  }

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override {
    ARROW_ASSIGN_OR_RAISE(auto data,
                          FixedWidthKeyEncoder::Decode(encoded_bytes, length, pool));

    if (dictionary_) {
      data->dictionary = dictionary_->data();
    } else {
      ARROW_ASSIGN_OR_RAISE(auto dict, MakeArrayOfNull(type_, 0));
      data->dictionary = dict->data();
    }

    data->type = type_;
    return data;
  }

  MemoryPool* pool_;
  std::shared_ptr<Array> dictionary_;
};

template <typename T>
struct VarLengthKeyEncoder : KeyEncoder {
  using Offset = typename T::offset_type;

  void AddLength(const ArrayData& data, int32_t* lengths) override {
    int64_t i = 0;
    VisitArrayDataInline<T>(
        data,
        [&](util::string_view bytes) {
          lengths[i++] +=
              kExtraByteForNull + sizeof(Offset) + static_cast<int32_t>(bytes.size());
        },
        [&] { lengths[i++] += kExtraByteForNull + sizeof(Offset); });
  }

  Status Encode(const ArrayData& data, uint8_t** encoded_bytes) override {
    VisitArrayDataInline<T>(
        data,
        [&](util::string_view bytes) {
          auto& encoded_ptr = *encoded_bytes++;
          *encoded_ptr++ = kValidByte;
          util::SafeStore(encoded_ptr, static_cast<Offset>(bytes.size()));
          encoded_ptr += sizeof(Offset);
          memcpy(encoded_ptr, bytes.data(), bytes.size());
          encoded_ptr += bytes.size();
        },
        [&] {
          auto& encoded_ptr = *encoded_bytes++;
          *encoded_ptr++ = kNullByte;
          util::SafeStore(encoded_ptr, static_cast<Offset>(0));
          encoded_ptr += sizeof(Offset);
        });
    return Status::OK();
  }

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override {
    std::shared_ptr<Buffer> null_buf;
    int32_t null_count;
    RETURN_NOT_OK(DecodeNulls(pool, length, encoded_bytes, &null_buf, &null_count));

    Offset length_sum = 0;
    for (int32_t i = 0; i < length; ++i) {
      length_sum += util::SafeLoadAs<Offset>(encoded_bytes[i]);
    }

    ARROW_ASSIGN_OR_RAISE(auto offset_buf,
                          AllocateBuffer(sizeof(Offset) * (1 + length), pool));
    ARROW_ASSIGN_OR_RAISE(auto key_buf, AllocateBuffer(length_sum));

    auto raw_offsets = reinterpret_cast<Offset*>(offset_buf->mutable_data());
    auto raw_keys = key_buf->mutable_data();

    Offset current_offset = 0;
    for (int32_t i = 0; i < length; ++i) {
      raw_offsets[i] = current_offset;

      auto key_length = util::SafeLoadAs<Offset>(encoded_bytes[i]);
      encoded_bytes[i] += sizeof(Offset);

      memcpy(raw_keys + current_offset, encoded_bytes[i], key_length);
      encoded_bytes[i] += key_length;

      current_offset += key_length;
    }
    raw_offsets[length] = current_offset;

    return ArrayData::Make(
        type_, length, {std::move(null_buf), std::move(offset_buf), std::move(key_buf)},
        null_count);
  }

  explicit VarLengthKeyEncoder(std::shared_ptr<DataType> type) : type_(std::move(type)) {}

  std::shared_ptr<DataType> type_;
};

struct GrouperImpl : Grouper {
  static Result<std::unique_ptr<GrouperImpl>> Make(const std::vector<ValueDescr>& keys,
                                                   ExecContext* ctx) {
    auto impl = ::arrow::internal::make_unique<GrouperImpl>();

    impl->encoders_.resize(keys.size());
    impl->ctx_ = ctx;

    for (size_t i = 0; i < keys.size(); ++i) {
      const auto& key = keys[i].type;

      if (key->id() == Type::BOOL) {
        impl->encoders_[i] = ::arrow::internal::make_unique<BooleanKeyEncoder>();
        continue;
      }

      if (key->id() == Type::DICTIONARY) {
        impl->encoders_[i] =
            ::arrow::internal::make_unique<DictionaryKeyEncoder>(key, ctx->memory_pool());
        continue;
      }

      if (is_fixed_width(key->id())) {
        impl->encoders_[i] = ::arrow::internal::make_unique<FixedWidthKeyEncoder>(key);
        continue;
      }

      if (is_binary_like(key->id())) {
        impl->encoders_[i] =
            ::arrow::internal::make_unique<VarLengthKeyEncoder<BinaryType>>(key);
        continue;
      }

      if (is_large_binary_like(key->id())) {
        impl->encoders_[i] =
            ::arrow::internal::make_unique<VarLengthKeyEncoder<LargeBinaryType>>(key);
        continue;
      }

      return Status::NotImplemented("Keys of type ", *key);
    }

    return std::move(impl);
  }

  Result<Datum> Consume(const ExecBatch& batch) override {
    std::vector<int32_t> offsets_batch(batch.length + 1);
    for (int i = 0; i < batch.num_values(); ++i) {
      encoders_[i]->AddLength(*batch[i].array(), offsets_batch.data());
    }

    int32_t total_length = 0;
    for (int64_t i = 0; i < batch.length; ++i) {
      auto total_length_before = total_length;
      total_length += offsets_batch[i];
      offsets_batch[i] = total_length_before;
    }
    offsets_batch[batch.length] = total_length;

    std::vector<uint8_t> key_bytes_batch(total_length);
    std::vector<uint8_t*> key_buf_ptrs(batch.length);
    for (int64_t i = 0; i < batch.length; ++i) {
      key_buf_ptrs[i] = key_bytes_batch.data() + offsets_batch[i];
    }

    for (int i = 0; i < batch.num_values(); ++i) {
      RETURN_NOT_OK(encoders_[i]->Encode(*batch[i].array(), key_buf_ptrs.data()));
    }

    TypedBufferBuilder<uint32_t> group_ids_batch(ctx_->memory_pool());
    RETURN_NOT_OK(group_ids_batch.Resize(batch.length));

    for (int64_t i = 0; i < batch.length; ++i) {
      int32_t key_length = offsets_batch[i + 1] - offsets_batch[i];
      std::string key(
          reinterpret_cast<const char*>(key_bytes_batch.data() + offsets_batch[i]),
          key_length);

      auto it_success = map_.emplace(key, num_groups_);
      auto group_id = it_success.first->second;

      if (it_success.second) {
        // new key; update offsets and key_bytes
        ++num_groups_;
        auto next_key_offset = static_cast<int32_t>(key_bytes_.size());
        key_bytes_.resize(next_key_offset + key_length);
        offsets_.push_back(next_key_offset + key_length);
        memcpy(key_bytes_.data() + next_key_offset, key.c_str(), key_length);
      }

      group_ids_batch.UnsafeAppend(group_id);
    }

    ARROW_ASSIGN_OR_RAISE(auto group_ids, group_ids_batch.Finish());
    return Datum(UInt32Array(batch.length, std::move(group_ids)));
  }

  uint32_t num_groups() const override { return num_groups_; }

  Result<ExecBatch> GetUniques() override {
    ExecBatch out({}, num_groups_);

    std::vector<uint8_t*> key_buf_ptrs(num_groups_);
    for (int64_t i = 0; i < num_groups_; ++i) {
      key_buf_ptrs[i] = key_bytes_.data() + offsets_[i];
    }

    out.values.resize(encoders_.size());
    for (size_t i = 0; i < encoders_.size(); ++i) {
      ARROW_ASSIGN_OR_RAISE(
          out.values[i],
          encoders_[i]->Decode(key_buf_ptrs.data(), static_cast<int32_t>(num_groups_),
                               ctx_->memory_pool()));
    }

    return out;
  }

  ExecContext* ctx_;
  std::unordered_map<std::string, uint32_t> map_;
  std::vector<int32_t> offsets_ = {0};
  std::vector<uint8_t> key_bytes_;
  uint32_t num_groups_ = 0;
  std::vector<std::unique_ptr<KeyEncoder>> encoders_;
};

struct GrouperFastImpl : Grouper {
  static constexpr int kBitmapPaddingForSIMD = 64;  // bits
  static constexpr int kPaddingForSIMD = 32;        // bytes

  static bool CanUse(const std::vector<ValueDescr>& keys) {
#if ARROW_LITTLE_ENDIAN
    for (size_t i = 0; i < keys.size(); ++i) {
      const auto& key = keys[i].type;
      if (is_large_binary_like(key->id())) {
        return false;
      }
    }
    return true;
#else
    return false;
#endif
  }

  static Result<std::unique_ptr<GrouperFastImpl>> Make(
      const std::vector<ValueDescr>& keys, ExecContext* ctx) {
    auto impl = ::arrow::internal::make_unique<GrouperFastImpl>();
    impl->ctx_ = ctx;

    RETURN_NOT_OK(impl->temp_stack_.Init(ctx->memory_pool(), 64 * minibatch_size_max_));
    impl->encode_ctx_.hardware_flags =
        arrow::internal::CpuInfo::GetInstance()->hardware_flags();
    impl->encode_ctx_.stack = &impl->temp_stack_;

    auto num_columns = keys.size();
    impl->col_metadata_.resize(num_columns);
    impl->key_types_.resize(num_columns);
    impl->dictionaries_.resize(num_columns);
    for (size_t icol = 0; icol < num_columns; ++icol) {
      const auto& key = keys[icol].type;
      if (key->id() == Type::DICTIONARY) {
        auto bit_width = checked_cast<const FixedWidthType&>(*key).bit_width();
        ARROW_DCHECK(bit_width % 8 == 0);
        impl->col_metadata_[icol] =
            arrow::compute::KeyEncoder::KeyColumnMetadata(true, bit_width / 8);
      } else if (key->id() == Type::BOOL) {
        impl->col_metadata_[icol] =
            arrow::compute::KeyEncoder::KeyColumnMetadata(true, 0);
      } else if (is_fixed_width(key->id())) {
        impl->col_metadata_[icol] = arrow::compute::KeyEncoder::KeyColumnMetadata(
            true, checked_cast<const FixedWidthType&>(*key).bit_width() / 8);
      } else if (is_binary_like(key->id())) {
        impl->col_metadata_[icol] =
            arrow::compute::KeyEncoder::KeyColumnMetadata(false, sizeof(uint32_t));
      } else {
        return Status::NotImplemented("Keys of type ", *key);
      }
      impl->key_types_[icol] = key;
    }

    impl->encoder_.Init(impl->col_metadata_, &impl->encode_ctx_,
                        /* row_alignment = */ sizeof(uint64_t),
                        /* string_alignment = */ sizeof(uint64_t));
    RETURN_NOT_OK(impl->rows_.Init(ctx->memory_pool(), impl->encoder_.row_metadata()));
    RETURN_NOT_OK(
        impl->rows_minibatch_.Init(ctx->memory_pool(), impl->encoder_.row_metadata()));
    impl->minibatch_size_ = impl->minibatch_size_min_;
    GrouperFastImpl* impl_ptr = impl.get();
    auto equal_func = [impl_ptr](
                          int num_keys_to_compare, const uint16_t* selection_may_be_null,
                          const uint32_t* group_ids, uint32_t* out_num_keys_mismatch,
                          uint16_t* out_selection_mismatch) {
      arrow::compute::KeyCompare::CompareRows(
          num_keys_to_compare, selection_may_be_null, group_ids, &impl_ptr->encode_ctx_,
          out_num_keys_mismatch, out_selection_mismatch, impl_ptr->rows_minibatch_,
          impl_ptr->rows_);
    };
    auto append_func = [impl_ptr](int num_keys, const uint16_t* selection) {
      return impl_ptr->rows_.AppendSelectionFrom(impl_ptr->rows_minibatch_, num_keys,
                                                 selection);
    };
    RETURN_NOT_OK(impl->map_.init(impl->encode_ctx_.hardware_flags, ctx->memory_pool(),
                                  impl->encode_ctx_.stack, impl->log_minibatch_max_,
                                  equal_func, append_func));
    impl->cols_.resize(num_columns);
    impl->minibatch_hashes_.resize(impl->minibatch_size_max_ +
                                   kPaddingForSIMD / sizeof(uint32_t));

    return std::move(impl);
  }

  ~GrouperFastImpl() { map_.cleanup(); }

  Result<Datum> Consume(const ExecBatch& batch) override {
    int64_t num_rows = batch.length;
    int num_columns = batch.num_values();

    // Process dictionaries
    for (int icol = 0; icol < num_columns; ++icol) {
      if (key_types_[icol]->id() == Type::DICTIONARY) {
        auto data = batch[icol].array();
        auto dict = MakeArray(data->dictionary);
        if (dictionaries_[icol]) {
          if (!dictionaries_[icol]->Equals(dict)) {
            // TODO(bkietz) unify if necessary. For now, just error if any batch's
            // dictionary differs from the first we saw for this key
            return Status::NotImplemented("Unifying differing dictionaries");
          }
        } else {
          dictionaries_[icol] = std::move(dict);
        }
      }
    }

    std::shared_ptr<arrow::Buffer> group_ids;
    ARROW_ASSIGN_OR_RAISE(
        group_ids, AllocateBuffer(sizeof(uint32_t) * num_rows, ctx_->memory_pool()));

    for (int icol = 0; icol < num_columns; ++icol) {
      const uint8_t* non_nulls = nullptr;
      if (batch[icol].array()->buffers[0] != NULLPTR) {
        non_nulls = batch[icol].array()->buffers[0]->data();
      }
      const uint8_t* fixedlen = batch[icol].array()->buffers[1]->data();
      const uint8_t* varlen = nullptr;
      if (!col_metadata_[icol].is_fixed_length) {
        varlen = batch[icol].array()->buffers[2]->data();
      }

      int64_t offset = batch[icol].array()->offset;

      auto col_base = arrow::compute::KeyEncoder::KeyColumnArray(
          col_metadata_[icol], offset + num_rows, non_nulls, fixedlen, varlen);

      cols_[icol] =
          arrow::compute::KeyEncoder::KeyColumnArray(col_base, offset, num_rows);
    }

    // Split into smaller mini-batches
    //
    for (uint32_t start_row = 0; start_row < num_rows;) {
      uint32_t batch_size_next = std::min(static_cast<uint32_t>(minibatch_size_),
                                          static_cast<uint32_t>(num_rows) - start_row);

      // Encode
      rows_minibatch_.Clean();
      RETURN_NOT_OK(encoder_.PrepareOutputForEncode(start_row, batch_size_next,
                                                    &rows_minibatch_, cols_));
      encoder_.Encode(start_row, batch_size_next, &rows_minibatch_, cols_);

      // Compute hash
      if (encoder_.row_metadata().is_fixed_length) {
        Hashing::hash_fixed(encode_ctx_.hardware_flags, batch_size_next,
                            encoder_.row_metadata().fixed_length, rows_minibatch_.data(1),
                            minibatch_hashes_.data());
      } else {
        auto hash_temp_buf =
            util::TempVectorHolder<uint32_t>(&temp_stack_, 4 * batch_size_next);
        Hashing::hash_varlen(encode_ctx_.hardware_flags, batch_size_next,
                             rows_minibatch_.offsets(), rows_minibatch_.data(2),
                             hash_temp_buf.mutable_data(), minibatch_hashes_.data());
      }

      // Map
      RETURN_NOT_OK(
          map_.map(batch_size_next, minibatch_hashes_.data(),
                   reinterpret_cast<uint32_t*>(group_ids->mutable_data()) + start_row));

      start_row += batch_size_next;

      if (minibatch_size_ * 2 <= minibatch_size_max_) {
        minibatch_size_ *= 2;
      }
    }

    return Datum(UInt32Array(batch.length, std::move(group_ids)));
  }

  uint32_t num_groups() const override { return static_cast<uint32_t>(rows_.length()); }

  // Make sure padded buffers end up with the right logical size

  Result<std::shared_ptr<Buffer>> AllocatePaddedBitmap(int64_t length) {
    ARROW_ASSIGN_OR_RAISE(
        std::shared_ptr<Buffer> buf,
        AllocateBitmap(length + kBitmapPaddingForSIMD, ctx_->memory_pool()));
    return SliceMutableBuffer(buf, 0, BitUtil::BytesForBits(length));
  }

  Result<std::shared_ptr<Buffer>> AllocatePaddedBuffer(int64_t size) {
    ARROW_ASSIGN_OR_RAISE(
        std::shared_ptr<Buffer> buf,
        AllocateBuffer(size + kBitmapPaddingForSIMD, ctx_->memory_pool()));
    return SliceMutableBuffer(buf, 0, size);
  }

  Result<ExecBatch> GetUniques() override {
    auto num_columns = static_cast<uint32_t>(col_metadata_.size());
    int64_t num_groups = rows_.length();

    std::vector<std::shared_ptr<Buffer>> non_null_bufs(num_columns);
    std::vector<std::shared_ptr<Buffer>> fixedlen_bufs(num_columns);
    std::vector<std::shared_ptr<Buffer>> varlen_bufs(num_columns);

    for (size_t i = 0; i < num_columns; ++i) {
      ARROW_ASSIGN_OR_RAISE(non_null_bufs[i], AllocatePaddedBitmap(num_groups));
      if (col_metadata_[i].is_fixed_length) {
        if (col_metadata_[i].fixed_length == 0) {
          ARROW_ASSIGN_OR_RAISE(fixedlen_bufs[i], AllocatePaddedBitmap(num_groups));
        } else {
          ARROW_ASSIGN_OR_RAISE(
              fixedlen_bufs[i],
              AllocatePaddedBuffer(num_groups * col_metadata_[i].fixed_length));
        }
      } else {
        ARROW_ASSIGN_OR_RAISE(fixedlen_bufs[i],
                              AllocatePaddedBuffer((num_groups + 1) * sizeof(uint32_t)));
      }
      cols_[i] = arrow::compute::KeyEncoder::KeyColumnArray(
          col_metadata_[i], num_groups, non_null_bufs[i]->mutable_data(),
          fixedlen_bufs[i]->mutable_data(), nullptr);
    }

    for (int64_t start_row = 0; start_row < num_groups;) {
      int64_t batch_size_next =
          std::min(num_groups - start_row, static_cast<int64_t>(minibatch_size_max_));
      encoder_.DecodeFixedLengthBuffers(start_row, start_row, batch_size_next, rows_,
                                        &cols_);
      start_row += batch_size_next;
    }

    if (!rows_.metadata().is_fixed_length) {
      for (size_t i = 0; i < num_columns; ++i) {
        if (!col_metadata_[i].is_fixed_length) {
          auto varlen_size =
              reinterpret_cast<const uint32_t*>(fixedlen_bufs[i]->data())[num_groups];
          ARROW_ASSIGN_OR_RAISE(varlen_bufs[i], AllocatePaddedBuffer(varlen_size));
          cols_[i] = arrow::compute::KeyEncoder::KeyColumnArray(
              col_metadata_[i], num_groups, non_null_bufs[i]->mutable_data(),
              fixedlen_bufs[i]->mutable_data(), varlen_bufs[i]->mutable_data());
        }
      }

      for (int64_t start_row = 0; start_row < num_groups;) {
        int64_t batch_size_next =
            std::min(num_groups - start_row, static_cast<int64_t>(minibatch_size_max_));
        encoder_.DecodeVaryingLengthBuffers(start_row, start_row, batch_size_next, rows_,
                                            &cols_);
        start_row += batch_size_next;
      }
    }

    ExecBatch out({}, num_groups);
    out.values.resize(num_columns);
    for (size_t i = 0; i < num_columns; ++i) {
      auto valid_count = arrow::internal::CountSetBits(
          non_null_bufs[i]->data(), /*offset=*/0, static_cast<int64_t>(num_groups));
      int null_count = static_cast<int>(num_groups) - static_cast<int>(valid_count);

      if (col_metadata_[i].is_fixed_length) {
        out.values[i] = ArrayData::Make(
            key_types_[i], num_groups,
            {std::move(non_null_bufs[i]), std::move(fixedlen_bufs[i])}, null_count);
      } else {
        out.values[i] =
            ArrayData::Make(key_types_[i], num_groups,
                            {std::move(non_null_bufs[i]), std::move(fixedlen_bufs[i]),
                             std::move(varlen_bufs[i])},
                            null_count);
      }
    }

    // Process dictionaries
    for (size_t icol = 0; icol < num_columns; ++icol) {
      if (key_types_[icol]->id() == Type::DICTIONARY) {
        if (dictionaries_[icol]) {
          out.values[icol].array()->dictionary = dictionaries_[icol]->data();
        } else {
          ARROW_ASSIGN_OR_RAISE(auto dict, MakeArrayOfNull(key_types_[icol], 0));
          out.values[icol].array()->dictionary = dict->data();
        }
      }
    }

    return out;
  }

  static constexpr int log_minibatch_max_ = 10;
  static constexpr int minibatch_size_max_ = 1 << log_minibatch_max_;
  static constexpr int minibatch_size_min_ = 128;
  int minibatch_size_;

  ExecContext* ctx_;
  arrow::util::TempVectorStack temp_stack_;
  arrow::compute::KeyEncoder::KeyEncoderContext encode_ctx_;

  std::vector<std::shared_ptr<arrow::DataType>> key_types_;
  std::vector<arrow::compute::KeyEncoder::KeyColumnMetadata> col_metadata_;
  std::vector<arrow::compute::KeyEncoder::KeyColumnArray> cols_;
  std::vector<uint32_t> minibatch_hashes_;

  std::vector<std::shared_ptr<Array>> dictionaries_;

  arrow::compute::KeyEncoder::KeyRowArray rows_;
  arrow::compute::KeyEncoder::KeyRowArray rows_minibatch_;
  arrow::compute::KeyEncoder encoder_;
  arrow::compute::SwissTable map_;
};

/// C++ abstract base class for the HashAggregateKernel interface.
/// Implementations should be default constructible and perform initialization in
/// Init().
struct GroupedAggregator : KernelState {
  virtual Status Init(ExecContext*, const FunctionOptions*) = 0;

  virtual Status Resize(int64_t new_num_groups) = 0;

  virtual Status Consume(const ExecBatch& batch) = 0;

  virtual Status Merge(GroupedAggregator&& other, const ArrayData& group_id_mapping) = 0;

  virtual Result<Datum> Finalize() = 0;

  virtual std::shared_ptr<DataType> out_type() const = 0;
};

template <typename Impl>
Result<std::unique_ptr<KernelState>> HashAggregateInit(KernelContext* ctx,
                                                       const KernelInitArgs& args) {
  auto impl = ::arrow::internal::make_unique<Impl>();
  RETURN_NOT_OK(impl->Init(ctx->exec_context(), args.options));
  return std::move(impl);
}

HashAggregateKernel MakeKernel(InputType argument_type, KernelInit init) {
  HashAggregateKernel kernel;

  kernel.init = std::move(init);

  kernel.signature = KernelSignature::Make(
      {std::move(argument_type), InputType::Array(Type::UINT32)},
      OutputType(
          [](KernelContext* ctx, const std::vector<ValueDescr>&) -> Result<ValueDescr> {
            return checked_cast<GroupedAggregator*>(ctx->state())->out_type();
          }));

  kernel.resize = [](KernelContext* ctx, int64_t num_groups) {
    return checked_cast<GroupedAggregator*>(ctx->state())->Resize(num_groups);
  };

  kernel.consume = [](KernelContext* ctx, const ExecBatch& batch) {
    return checked_cast<GroupedAggregator*>(ctx->state())->Consume(batch);
  };

  kernel.merge = [](KernelContext* ctx, KernelState&& other,
                    const ArrayData& group_id_mapping) {
    return checked_cast<GroupedAggregator*>(ctx->state())
        ->Merge(checked_cast<GroupedAggregator&&>(other), group_id_mapping);
  };

  kernel.finalize = [](KernelContext* ctx, Datum* out) {
    ARROW_ASSIGN_OR_RAISE(*out,
                          checked_cast<GroupedAggregator*>(ctx->state())->Finalize());
    return Status::OK();
  };

  return kernel;
}

Status AddHashAggKernels(
    const std::vector<std::shared_ptr<DataType>>& types,
    Result<HashAggregateKernel> make_kernel(const std::shared_ptr<DataType>&),
    HashAggregateFunction* function) {
  for (const auto& ty : types) {
    ARROW_ASSIGN_OR_RAISE(auto kernel, make_kernel(ty));
    RETURN_NOT_OK(function->AddKernel(std::move(kernel)));
  }
  return Status::OK();
}

// ----------------------------------------------------------------------
// Count implementation

struct GroupedCountImpl : public GroupedAggregator {
  Status Init(ExecContext* ctx, const FunctionOptions* options) override {
    options_ = checked_cast<const CountOptions&>(*options);
    counts_ = BufferBuilder(ctx->memory_pool());
    return Status::OK();
  }

  Status Resize(int64_t new_num_groups) override {
    auto added_groups = new_num_groups - num_groups_;
    num_groups_ = new_num_groups;
    return counts_.Append(added_groups * sizeof(int64_t), 0);
  }

  Status Merge(GroupedAggregator&& raw_other,
               const ArrayData& group_id_mapping) override {
    auto other = checked_cast<GroupedCountImpl*>(&raw_other);

    auto counts = reinterpret_cast<int64_t*>(counts_.mutable_data());
    auto other_counts = reinterpret_cast<const int64_t*>(other->counts_.mutable_data());

    auto g = group_id_mapping.GetValues<uint32_t>(1);
    for (int64_t other_g = 0; other_g < group_id_mapping.length; ++other_g, ++g) {
      counts[*g] += other_counts[other_g];
    }
    return Status::OK();
  }

  Status Consume(const ExecBatch& batch) override {
    auto counts = reinterpret_cast<int64_t*>(counts_.mutable_data());

    const auto& input = batch[0].array();

    auto g_begin = batch[1].array()->GetValues<uint32_t>(1);
    switch (options_.mode) {
      case CountOptions::ONLY_VALID: {
        arrow::internal::VisitSetBitRunsVoid(input->buffers[0], input->offset,
                                             input->length,
                                             [&](int64_t offset, int64_t length) {
                                               auto g = g_begin + offset;
                                               for (int64_t i = 0; i < length; ++i, ++g) {
                                                 counts[*g] += 1;
                                               }
                                             });
        break;
      }
      case CountOptions::ONLY_NULL: {
        if (input->MayHaveNulls()) {
          auto end = input->offset + input->length;
          for (int64_t i = input->offset; i < end; ++i, ++g_begin) {
            counts[*g_begin] += !BitUtil::GetBit(input->buffers[0]->data(), i);
          }
        }
        break;
      }
      case CountOptions::ALL: {
        for (int64_t i = 0; i < batch.length; ++i, ++g_begin) {
          counts[*g_begin] += 1;
        }
        break;
      }
      default:
        DCHECK(false) << "unreachable";
    }
    return Status::OK();
  }

  Result<Datum> Finalize() override {
    ARROW_ASSIGN_OR_RAISE(auto counts, counts_.Finish());
    return std::make_shared<Int64Array>(num_groups_, std::move(counts));
  }

  std::shared_ptr<DataType> out_type() const override { return int64(); }

  int64_t num_groups_ = 0;
  CountOptions options_;
  BufferBuilder counts_;
};

// ----------------------------------------------------------------------
// Sum/Mean/Product implementation

template <typename Type, typename Impl>
struct GroupedReducingAggregator : public GroupedAggregator {
  using AccType = typename FindAccumulatorType<Type>::Type;
  using c_type = typename TypeTraits<AccType>::CType;

  Status Init(ExecContext* ctx, const FunctionOptions* options) override {
    pool_ = ctx->memory_pool();
    options_ = checked_cast<const ScalarAggregateOptions&>(*options);
    reduced_ = TypedBufferBuilder<c_type>(pool_);
    counts_ = TypedBufferBuilder<int64_t>(pool_);
    no_nulls_ = TypedBufferBuilder<bool>(pool_);
    // out_type_ initialized by SumInit
    return Status::OK();
  }

  Status Resize(int64_t new_num_groups) override {
    auto added_groups = new_num_groups - num_groups_;
    num_groups_ = new_num_groups;
    RETURN_NOT_OK(reduced_.Append(added_groups, Impl::NullValue(*out_type_)));
    RETURN_NOT_OK(counts_.Append(added_groups, 0));
    RETURN_NOT_OK(no_nulls_.Append(added_groups, true));
    return Status::OK();
  }

  Status Consume(const ExecBatch& batch) override {
    c_type* reduced = reduced_.mutable_data();
    int64_t* counts = counts_.mutable_data();
    uint8_t* no_nulls = no_nulls_.mutable_data();

    auto g = batch[1].array()->GetValues<uint32_t>(1);

    return Impl::Consume(*batch[0].array(), reduced, counts, no_nulls, g);
  }

  Status Merge(GroupedAggregator&& raw_other,
               const ArrayData& group_id_mapping) override {
    auto other = checked_cast<GroupedReducingAggregator<Type, Impl>*>(&raw_other);

    c_type* reduced = reduced_.mutable_data();
    int64_t* counts = counts_.mutable_data();
    uint8_t* no_nulls = no_nulls_.mutable_data();

    const c_type* other_reduced = other->reduced_.data();
    const int64_t* other_counts = other->counts_.data();
    const uint8_t* other_no_nulls = no_nulls_.mutable_data();

    auto g = group_id_mapping.GetValues<uint32_t>(1);
    for (int64_t other_g = 0; other_g < group_id_mapping.length; ++other_g, ++g) {
      counts[*g] += other_counts[other_g];
      Impl::UpdateGroupWith(*out_type_, reduced, *g, other_reduced[other_g]);
      BitUtil::SetBitTo(
          no_nulls, *g,
          BitUtil::GetBit(no_nulls, *g) && BitUtil::GetBit(other_no_nulls, other_g));
    }
    return Status::OK();
  }

  // Generate the values/nulls buffers
  static Result<std::shared_ptr<Buffer>> Finish(MemoryPool* pool,
                                                const ScalarAggregateOptions& options,
                                                const int64_t* counts,
                                                TypedBufferBuilder<c_type>* reduced,
                                                int64_t num_groups, int64_t* null_count,
                                                std::shared_ptr<Buffer>* null_bitmap) {
    for (int64_t i = 0; i < num_groups; ++i) {
      if (counts[i] >= options.min_count) continue;

      if ((*null_bitmap) == nullptr) {
        ARROW_ASSIGN_OR_RAISE(*null_bitmap, AllocateBitmap(num_groups, pool));
        BitUtil::SetBitsTo((*null_bitmap)->mutable_data(), 0, num_groups, true);
      }

      (*null_count)++;
      BitUtil::SetBitTo((*null_bitmap)->mutable_data(), i, false);
    }
    return reduced->Finish();
  }

  Result<Datum> Finalize() override {
    std::shared_ptr<Buffer> null_bitmap = nullptr;
    const int64_t* counts = counts_.data();
    int64_t null_count = 0;

    ARROW_ASSIGN_OR_RAISE(auto values,
                          Impl::Finish(pool_, options_, counts, &reduced_, num_groups_,
                                       &null_count, &null_bitmap));

    if (!options_.skip_nulls) {
      null_count = kUnknownNullCount;
      if (null_bitmap) {
        arrow::internal::BitmapAnd(null_bitmap->data(), /*left_offset=*/0,
                                   no_nulls_.data(), /*right_offset=*/0, num_groups_,
                                   /*out_offset=*/0, null_bitmap->mutable_data());
      } else {
        ARROW_ASSIGN_OR_RAISE(null_bitmap, no_nulls_.Finish());
      }
    }

    return ArrayData::Make(out_type(), num_groups_,
                           {std::move(null_bitmap), std::move(values)}, null_count);
  }

  std::shared_ptr<DataType> out_type() const override { return out_type_; }

  int64_t num_groups_ = 0;
  ScalarAggregateOptions options_;
  TypedBufferBuilder<c_type> reduced_;
  TypedBufferBuilder<int64_t> counts_;
  TypedBufferBuilder<bool> no_nulls_;
  std::shared_ptr<DataType> out_type_;
  MemoryPool* pool_;
};

// ----------------------------------------------------------------------
// Sum implementation

template <typename Type>
struct GroupedSumImpl : public GroupedReducingAggregator<Type, GroupedSumImpl<Type>> {
  using Base = GroupedReducingAggregator<Type, GroupedSumImpl<Type>>;
  using c_type = typename Base::c_type;

  // Default value for a group
  static c_type NullValue(const DataType&) { return c_type(0); }

  // Update all groups
  static Status Consume(const ArrayData& values, c_type* reduced, int64_t* counts,
                        uint8_t* no_nulls, const uint32_t* g) {
    // XXX this uses naive summation; we should switch to pairwise summation as was
    // done for the scalar aggregate kernel in ARROW-11758
    internal::VisitArrayValuesInline<Type>(
        values,
        [&](typename TypeTraits<Type>::CType value) {
          reduced[*g] = static_cast<c_type>(to_unsigned(reduced[*g]) +
                                            to_unsigned(static_cast<c_type>(value)));
          counts[*g++] += 1;
        },
        [&] { BitUtil::SetBitTo(no_nulls, *g++, false); });
    return Status::OK();
  }

  // Update a single group during merge
  static void UpdateGroupWith(const DataType&, c_type* reduced, uint32_t g,
                              c_type value) {
    reduced[g] += value;
  }

  using Base::Finish;
};

template <template <typename T> class Impl, typename T>
Result<std::unique_ptr<KernelState>> SumInit(KernelContext* ctx,
                                             const KernelInitArgs& args) {
  ARROW_ASSIGN_OR_RAISE(auto impl, HashAggregateInit<Impl<T>>(ctx, args));
  static_cast<Impl<T>*>(impl.get())->out_type_ =
      TypeTraits<typename Impl<T>::AccType>::type_singleton();
  return std::move(impl);
}

template <typename Impl>
Result<std::unique_ptr<KernelState>> DecimalSumInit(KernelContext* ctx,
                                                    const KernelInitArgs& args) {
  ARROW_ASSIGN_OR_RAISE(auto impl, HashAggregateInit<Impl>(ctx, args));
  static_cast<Impl*>(impl.get())->out_type_ = args.inputs[0].type;
  return std::move(impl);
}

struct GroupedSumFactory {
  template <typename T, typename AccType = typename FindAccumulatorType<T>::Type>
  Status Visit(const T&) {
    kernel = MakeKernel(std::move(argument_type), SumInit<GroupedSumImpl, T>);
    return Status::OK();
  }

  Status Visit(const Decimal128Type&) {
    kernel = MakeKernel(std::move(argument_type),
                        DecimalSumInit<GroupedSumImpl<Decimal128Type>>);
    return Status::OK();
  }
  Status Visit(const Decimal256Type&) {
    kernel = MakeKernel(std::move(argument_type),
                        DecimalSumInit<GroupedSumImpl<Decimal256Type>>);
    return Status::OK();
  }

  Status Visit(const HalfFloatType& type) {
    return Status::NotImplemented("Summing data of type ", type);
  }

  Status Visit(const DataType& type) {
    return Status::NotImplemented("Summing data of type ", type);
  }

  static Result<HashAggregateKernel> Make(const std::shared_ptr<DataType>& type) {
    GroupedSumFactory factory;
    factory.argument_type = InputType::Array(type->id());
    RETURN_NOT_OK(VisitTypeInline(*type, &factory));
    return std::move(factory.kernel);
  }

  HashAggregateKernel kernel;
  InputType argument_type;
};

// ----------------------------------------------------------------------
// Product implementation

template <typename Type>
struct GroupedProductImpl final
    : public GroupedReducingAggregator<Type, GroupedProductImpl<Type>> {
  using Base = GroupedReducingAggregator<Type, GroupedProductImpl<Type>>;
  using AccType = typename Base::AccType;
  using c_type = typename Base::c_type;

  static c_type NullValue(const DataType& out_type) {
    return MultiplyTraits<AccType>::one(out_type);
  }

  static Status Consume(const ArrayData& values, c_type* reduced, int64_t* counts,
                        uint8_t* no_nulls, const uint32_t* g) {
    internal::VisitArrayValuesInline<Type>(
        values,
        [&](typename TypeTraits<Type>::CType value) {
          reduced[*g] = MultiplyTraits<AccType>::Multiply(*values.type, reduced[*g],
                                                          static_cast<c_type>(value));
          counts[*g++] += 1;
        },
        [&] { BitUtil::SetBitTo(no_nulls, *g++, false); });
    return Status::OK();
  }

  static void UpdateGroupWith(const DataType& out_type, c_type* reduced, uint32_t g,
                              c_type value) {
    reduced[g] = MultiplyTraits<AccType>::Multiply(out_type, reduced[g], value);
  }

  using Base::Finish;
};

struct GroupedProductFactory {
  template <typename T, typename AccType = typename FindAccumulatorType<T>::Type>
  Status Visit(const T&) {
    kernel = MakeKernel(std::move(argument_type), SumInit<GroupedProductImpl, T>);
    return Status::OK();
  }

  Status Visit(const Decimal128Type&) {
    kernel = MakeKernel(std::move(argument_type),
                        DecimalSumInit<GroupedProductImpl<Decimal128Type>>);
    return Status::OK();
  }

  Status Visit(const Decimal256Type&) {
    kernel = MakeKernel(std::move(argument_type),
                        DecimalSumInit<GroupedProductImpl<Decimal256Type>>);
    return Status::OK();
  }

  Status Visit(const HalfFloatType& type) {
    return Status::NotImplemented("Taking product of data of type ", type);
  }

  Status Visit(const DataType& type) {
    return Status::NotImplemented("Taking product of data of type ", type);
  }

  static Result<HashAggregateKernel> Make(const std::shared_ptr<DataType>& type) {
    GroupedProductFactory factory;
    factory.argument_type = InputType::Array(type->id());
    RETURN_NOT_OK(VisitTypeInline(*type, &factory));
    return std::move(factory.kernel);
  }

  HashAggregateKernel kernel;
  InputType argument_type;
};

// ----------------------------------------------------------------------
// Mean implementation

template <typename Type>
struct GroupedMeanImpl : public GroupedReducingAggregator<Type, GroupedMeanImpl<Type>> {
  using Base = GroupedReducingAggregator<Type, GroupedMeanImpl<Type>>;
  using c_type = typename Base::c_type;
  using MeanType =
      typename std::conditional<is_decimal_type<Type>::value, c_type, double>::type;

  static c_type NullValue(const DataType&) { return c_type(0); }

  static Status Consume(const ArrayData& values, c_type* reduced, int64_t* counts,
                        uint8_t* no_nulls, const uint32_t* g) {
    // XXX this uses naive summation; we should switch to pairwise summation as was
    // done for the scalar aggregate kernel in ARROW-11758
    internal::VisitArrayValuesInline<Type>(
        values,
        [&](typename TypeTraits<Type>::CType value) {
          reduced[*g] = static_cast<c_type>(to_unsigned(reduced[*g]) +
                                            to_unsigned(static_cast<c_type>(value)));
          counts[*g++] += 1;
        },
        [&] { BitUtil::SetBitTo(no_nulls, *g++, false); });
    return Status::OK();
  }

  static void UpdateGroupWith(const DataType&, c_type* reduced, uint32_t g,
                              c_type value) {
    reduced[g] += value;
  }

  static Result<std::shared_ptr<Buffer>> Finish(MemoryPool* pool,
                                                const ScalarAggregateOptions& options,
                                                const int64_t* counts,
                                                TypedBufferBuilder<c_type>* reduced_,
                                                int64_t num_groups, int64_t* null_count,
                                                std::shared_ptr<Buffer>* null_bitmap) {
    const c_type* reduced = reduced_->data();
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Buffer> values,
                          AllocateBuffer(num_groups * sizeof(MeanType), pool));
    MeanType* means = reinterpret_cast<MeanType*>(values->mutable_data());
    for (int64_t i = 0; i < num_groups; ++i) {
      if (counts[i] >= options.min_count) {
        means[i] = static_cast<MeanType>(reduced[i]) / counts[i];
        continue;
      }
      means[i] = MeanType(0);

      if ((*null_bitmap) == nullptr) {
        ARROW_ASSIGN_OR_RAISE(*null_bitmap, AllocateBitmap(num_groups, pool));
        BitUtil::SetBitsTo((*null_bitmap)->mutable_data(), 0, num_groups, true);
      }

      (*null_count)++;
      BitUtil::SetBitTo((*null_bitmap)->mutable_data(), i, false);
    }
    return std::move(values);
  }

  std::shared_ptr<DataType> out_type() const override {
    if (is_decimal_type<Type>::value) return this->out_type_;
    return float64();
  }
};

struct GroupedMeanFactory {
  template <typename T, typename AccType = typename FindAccumulatorType<T>::Type>
  Status Visit(const T&) {
    kernel = MakeKernel(std::move(argument_type), SumInit<GroupedMeanImpl, T>);
    return Status::OK();
  }

  Status Visit(const Decimal128Type&) {
    kernel = MakeKernel(std::move(argument_type),
                        DecimalSumInit<GroupedMeanImpl<Decimal128Type>>);
    return Status::OK();
  }

  Status Visit(const Decimal256Type&) {
    kernel = MakeKernel(std::move(argument_type),
                        DecimalSumInit<GroupedMeanImpl<Decimal256Type>>);
    return Status::OK();
  }

  Status Visit(const HalfFloatType& type) {
    return Status::NotImplemented("Computing mean of type ", type);
  }

  Status Visit(const DataType& type) {
    return Status::NotImplemented("Computing mean of type ", type);
  }

  static Result<HashAggregateKernel> Make(const std::shared_ptr<DataType>& type) {
    GroupedMeanFactory factory;
    factory.argument_type = InputType::Array(type->id());
    RETURN_NOT_OK(VisitTypeInline(*type, &factory));
    return std::move(factory.kernel);
  }

  HashAggregateKernel kernel;
  InputType argument_type;
};

// Variance/Stdev implementation

using arrow::internal::int128_t;

template <typename Type>
struct GroupedVarStdImpl : public GroupedAggregator {
  using CType = typename Type::c_type;

  Status Init(ExecContext* ctx, const FunctionOptions* options) override {
    options_ = *checked_cast<const VarianceOptions*>(options);
    ctx_ = ctx;
    pool_ = ctx->memory_pool();
    counts_ = BufferBuilder(pool_);
    means_ = BufferBuilder(pool_);
    m2s_ = BufferBuilder(pool_);
    return Status::OK();
  }

  Status Resize(int64_t new_num_groups) override {
    auto added_groups = new_num_groups - num_groups_;
    num_groups_ = new_num_groups;
    RETURN_NOT_OK(counts_.Append(added_groups * sizeof(int64_t), 0));
    RETURN_NOT_OK(means_.Append(added_groups * sizeof(double), 0));
    RETURN_NOT_OK(m2s_.Append(added_groups * sizeof(double), 0));
    return Status::OK();
  }

  Status Consume(const ExecBatch& batch) override { return ConsumeImpl(batch); }

  // float/double/int64: calculate `m2` (sum((X-mean)^2)) with `two pass algorithm`
  // (see aggregate_var_std.cc)
  template <typename T = Type>
  enable_if_t<is_floating_type<T>::value || (sizeof(CType) > 4), Status> ConsumeImpl(
      const ExecBatch& batch) {
    using SumType =
        typename std::conditional<is_floating_type<T>::value, double, int128_t>::type;

    int64_t* counts = reinterpret_cast<int64_t*>(counts_.mutable_data());
    double* means = reinterpret_cast<double*>(means_.mutable_data());
    double* m2s = reinterpret_cast<double*>(m2s_.mutable_data());

    // XXX this uses naive summation; we should switch to pairwise summation as was
    // done for the scalar aggregate kernel in ARROW-11567
    std::vector<SumType> sums(num_groups_);
    auto g = batch[1].array()->GetValues<uint32_t>(1);
    VisitArrayDataInline<Type>(
        *batch[0].array(),
        [&](typename TypeTraits<Type>::CType value) {
          sums[*g] += value;
          counts[*g] += 1;
          ++g;
        },
        [&] { ++g; });

    for (int64_t i = 0; i < num_groups_; i++) {
      means[i] = static_cast<double>(sums[i]) / counts[i];
    }

    g = batch[1].array()->GetValues<uint32_t>(1);
    VisitArrayDataInline<Type>(
        *batch[0].array(),
        [&](typename TypeTraits<Type>::CType value) {
          const double v = static_cast<double>(value);
          m2s[*g] += (v - means[*g]) * (v - means[*g]);
          ++g;
        },
        [&] { ++g; });

    return Status::OK();
  }

  // int32/16/8: textbook one pass algorithm with integer arithmetic (see
  // aggregate_var_std.cc)
  template <typename T = Type>
  enable_if_t<is_integer_type<T>::value && (sizeof(CType) <= 4), Status> ConsumeImpl(
      const ExecBatch& batch) {
    // max number of elements that sum will not overflow int64 (2Gi int32 elements)
    // for uint32:    0 <= sum < 2^63 (int64 >= 0)
    // for int32: -2^62 <= sum < 2^62
    constexpr int64_t max_length = 1ULL << (63 - sizeof(CType) * 8);

    const auto& array = *batch[0].array();
    const auto g = batch[1].array()->GetValues<uint32_t>(1);

    std::vector<IntegerVarStd<Type>> var_std(num_groups_);

    ARROW_ASSIGN_OR_RAISE(auto mapping,
                          AllocateBuffer(num_groups_ * sizeof(uint32_t), pool_));
    for (uint32_t i = 0; static_cast<int64_t>(i) < num_groups_; i++) {
      reinterpret_cast<uint32_t*>(mapping->mutable_data())[i] = i;
    }
    ArrayData group_id_mapping(uint32(), num_groups_, {nullptr, std::move(mapping)},
                               /*null_count=*/0);

    const CType* values = array.GetValues<CType>(1);

    for (int64_t start_index = 0; start_index < batch.length; start_index += max_length) {
      // process in chunks that overflow will never happen

      // reset state
      var_std.clear();
      var_std.resize(num_groups_);
      GroupedVarStdImpl<Type> state;
      RETURN_NOT_OK(state.Init(ctx_, &options_));
      RETURN_NOT_OK(state.Resize(num_groups_));
      int64_t* other_counts = reinterpret_cast<int64_t*>(state.counts_.mutable_data());
      double* other_means = reinterpret_cast<double*>(state.means_.mutable_data());
      double* other_m2s = reinterpret_cast<double*>(state.m2s_.mutable_data());

      arrow::internal::VisitSetBitRunsVoid(
          array.buffers[0], array.offset + start_index,
          std::min(max_length, batch.length - start_index),
          [&](int64_t pos, int64_t len) {
            for (int64_t i = 0; i < len; ++i) {
              const int64_t index = start_index + pos + i;
              const auto value = values[index];
              var_std[g[index]].ConsumeOne(value);
            }
          });

      for (int64_t i = 0; i < num_groups_; i++) {
        if (var_std[i].count == 0) continue;

        other_counts[i] = var_std[i].count;
        other_means[i] = var_std[i].mean();
        other_m2s[i] = var_std[i].m2();
      }
      RETURN_NOT_OK(this->Merge(std::move(state), group_id_mapping));
    }
    return Status::OK();
  }

  Status Merge(GroupedAggregator&& raw_other,
               const ArrayData& group_id_mapping) override {
    // Combine m2 from two chunks (see aggregate_var_std.cc)
    auto other = checked_cast<GroupedVarStdImpl*>(&raw_other);

    auto counts = reinterpret_cast<int64_t*>(counts_.mutable_data());
    auto means = reinterpret_cast<double*>(means_.mutable_data());
    auto m2s = reinterpret_cast<double*>(m2s_.mutable_data());

    const auto* other_counts = reinterpret_cast<const int64_t*>(other->counts_.data());
    const auto* other_means = reinterpret_cast<const double*>(other->means_.data());
    const auto* other_m2s = reinterpret_cast<const double*>(other->m2s_.data());

    auto g = group_id_mapping.GetValues<uint32_t>(1);
    for (int64_t other_g = 0; other_g < group_id_mapping.length; ++other_g, ++g) {
      if (other_counts[other_g] == 0) continue;
      MergeVarStd(counts[*g], means[*g], other_counts[other_g], other_means[other_g],
                  other_m2s[other_g], &counts[*g], &means[*g], &m2s[*g]);
    }
    return Status::OK();
  }

  Result<Datum> Finalize() override {
    std::shared_ptr<Buffer> null_bitmap;
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Buffer> values,
                          AllocateBuffer(num_groups_ * sizeof(double), pool_));
    int64_t null_count = 0;

    double* results = reinterpret_cast<double*>(values->mutable_data());
    const int64_t* counts = reinterpret_cast<const int64_t*>(counts_.data());
    const double* m2s = reinterpret_cast<const double*>(m2s_.data());
    for (int64_t i = 0; i < num_groups_; ++i) {
      if (counts[i] > options_.ddof) {
        const double variance = m2s[i] / (counts[i] - options_.ddof);
        results[i] = result_type_ == VarOrStd::Var ? variance : std::sqrt(variance);
        continue;
      }

      results[i] = 0;
      if (null_bitmap == nullptr) {
        ARROW_ASSIGN_OR_RAISE(null_bitmap, AllocateBitmap(num_groups_, pool_));
        BitUtil::SetBitsTo(null_bitmap->mutable_data(), 0, num_groups_, true);
      }

      null_count += 1;
      BitUtil::SetBitTo(null_bitmap->mutable_data(), i, false);
    }

    return ArrayData::Make(float64(), num_groups_,
                           {std::move(null_bitmap), std::move(values)}, null_count);
  }

  std::shared_ptr<DataType> out_type() const override { return float64(); }

  VarOrStd result_type_;
  VarianceOptions options_;
  int64_t num_groups_ = 0;
  // m2 = count * s2 = sum((X-mean)^2)
  BufferBuilder counts_, means_, m2s_;
  ExecContext* ctx_;
  MemoryPool* pool_;
};

template <typename T, VarOrStd result_type>
Result<std::unique_ptr<KernelState>> VarStdInit(KernelContext* ctx,
                                                const KernelInitArgs& args) {
  auto impl = ::arrow::internal::make_unique<GroupedVarStdImpl<T>>();
  impl->result_type_ = result_type;
  RETURN_NOT_OK(impl->Init(ctx->exec_context(), args.options));
  return std::move(impl);
}

template <VarOrStd result_type>
struct GroupedVarStdFactory {
  template <typename T, typename Enable = enable_if_t<is_integer_type<T>::value ||
                                                      is_floating_type<T>::value>>
  Status Visit(const T&) {
    kernel = MakeKernel(std::move(argument_type), VarStdInit<T, result_type>);
    return Status::OK();
  }

  Status Visit(const HalfFloatType& type) {
    return Status::NotImplemented("Computing variance/stddev of data of type ", type);
  }

  Status Visit(const DataType& type) {
    return Status::NotImplemented("Computing variance/stddev of data of type ", type);
  }

  static Result<HashAggregateKernel> Make(const std::shared_ptr<DataType>& type) {
    GroupedVarStdFactory factory;
    factory.argument_type = InputType::Array(type);
    RETURN_NOT_OK(VisitTypeInline(*type, &factory));
    return std::move(factory.kernel);
  }

  HashAggregateKernel kernel;
  InputType argument_type;
};

// ----------------------------------------------------------------------
// TDigest implementation

using arrow::internal::TDigest;

template <typename Type>
struct GroupedTDigestImpl : public GroupedAggregator {
  using CType = typename Type::c_type;

  Status Init(ExecContext* ctx, const FunctionOptions* options) override {
    options_ = *checked_cast<const TDigestOptions*>(options);
    ctx_ = ctx;
    pool_ = ctx->memory_pool();
    return Status::OK();
  }

  Status Resize(int64_t new_num_groups) override {
    const int64_t added_groups = new_num_groups - tdigests_.size();
    tdigests_.reserve(new_num_groups);
    for (int64_t i = 0; i < added_groups; i++) {
      tdigests_.emplace_back(options_.delta, options_.buffer_size);
    }
    return Status::OK();
  }

  Status Consume(const ExecBatch& batch) override {
    auto g = batch[1].array()->GetValues<uint32_t>(1);
    VisitArrayDataInline<Type>(
        *batch[0].array(),
        [&](typename TypeTraits<Type>::CType value) {
          this->tdigests_[*g].NanAdd(value);
          ++g;
        },
        [&] { ++g; });
    return Status::OK();
  }

  Status Merge(GroupedAggregator&& raw_other,
               const ArrayData& group_id_mapping) override {
    auto other = checked_cast<GroupedTDigestImpl*>(&raw_other);

    auto g = group_id_mapping.GetValues<uint32_t>(1);
    for (int64_t other_g = 0; other_g < group_id_mapping.length; ++other_g, ++g) {
      tdigests_[*g].Merge(other->tdigests_[other_g]);
    }

    return Status::OK();
  }

  Result<Datum> Finalize() override {
    std::shared_ptr<Buffer> null_bitmap;
    ARROW_ASSIGN_OR_RAISE(
        std::shared_ptr<Buffer> values,
        AllocateBuffer(tdigests_.size() * options_.q.size() * sizeof(double), pool_));
    int64_t null_count = 0;
    const int64_t slot_length = options_.q.size();

    double* results = reinterpret_cast<double*>(values->mutable_data());
    for (int64_t i = 0; static_cast<size_t>(i) < tdigests_.size(); ++i) {
      if (!tdigests_[i].is_empty()) {
        for (int64_t j = 0; j < slot_length; j++) {
          results[i * slot_length + j] = tdigests_[i].Quantile(options_.q[j]);
        }
        continue;
      }

      if (!null_bitmap) {
        ARROW_ASSIGN_OR_RAISE(null_bitmap, AllocateBitmap(tdigests_.size(), pool_));
        BitUtil::SetBitsTo(null_bitmap->mutable_data(), 0, tdigests_.size(), true);
      }
      null_count++;
      BitUtil::SetBitTo(null_bitmap->mutable_data(), i, false);
      std::fill(&results[i * slot_length], &results[(i + 1) * slot_length], 0.0);
    }

    auto child = ArrayData::Make(float64(), tdigests_.size() * options_.q.size(),
                                 {nullptr, std::move(values)}, /*null_count=*/0);
    return ArrayData::Make(out_type(), tdigests_.size(), {std::move(null_bitmap)},
                           {std::move(child)}, null_count);
  }

  std::shared_ptr<DataType> out_type() const override {
    return fixed_size_list(float64(), static_cast<int32_t>(options_.q.size()));
  }

  TDigestOptions options_;
  std::vector<TDigest> tdigests_;
  ExecContext* ctx_;
  MemoryPool* pool_;
};

struct GroupedTDigestFactory {
  template <typename T>
  enable_if_number<T, Status> Visit(const T&) {
    kernel =
        MakeKernel(std::move(argument_type), HashAggregateInit<GroupedTDigestImpl<T>>);
    return Status::OK();
  }

  Status Visit(const HalfFloatType& type) {
    return Status::NotImplemented("Computing t-digest of data of type ", type);
  }

  Status Visit(const DataType& type) {
    return Status::NotImplemented("Computing t-digest of data of type ", type);
  }

  static Result<HashAggregateKernel> Make(const std::shared_ptr<DataType>& type) {
    GroupedTDigestFactory factory;
    factory.argument_type = InputType::Array(type);
    RETURN_NOT_OK(VisitTypeInline(*type, &factory));
    return std::move(factory.kernel);
  }

  HashAggregateKernel kernel;
  InputType argument_type;
};

// ----------------------------------------------------------------------
// MinMax implementation

template <typename CType>
struct AntiExtrema {
  static constexpr CType anti_min() { return std::numeric_limits<CType>::max(); }
  static constexpr CType anti_max() { return std::numeric_limits<CType>::min(); }
};

template <>
struct AntiExtrema<float> {
  static constexpr float anti_min() { return std::numeric_limits<float>::infinity(); }
  static constexpr float anti_max() { return -std::numeric_limits<float>::infinity(); }
};

template <>
struct AntiExtrema<double> {
  static constexpr double anti_min() { return std::numeric_limits<double>::infinity(); }
  static constexpr double anti_max() { return -std::numeric_limits<double>::infinity(); }
};

template <>
struct AntiExtrema<Decimal128> {
  static constexpr Decimal128 anti_min() { return BasicDecimal128::GetMaxSentinel(); }
  static constexpr Decimal128 anti_max() { return BasicDecimal128::GetMinSentinel(); }
};

template <>
struct AntiExtrema<Decimal256> {
  static constexpr Decimal256 anti_min() { return BasicDecimal256::GetMaxSentinel(); }
  static constexpr Decimal256 anti_max() { return BasicDecimal256::GetMinSentinel(); }
};

template <typename Type>
struct GroupedMinMaxImpl : public GroupedAggregator {
  using CType = typename TypeTraits<Type>::CType;

  Status Init(ExecContext* ctx, const FunctionOptions* options) override {
    options_ = *checked_cast<const ScalarAggregateOptions*>(options);
    // type_ initialized by MinMaxInit
    mins_ = TypedBufferBuilder<CType>(ctx->memory_pool());
    maxes_ = TypedBufferBuilder<CType>(ctx->memory_pool());
    has_values_ = TypedBufferBuilder<bool>(ctx->memory_pool());
    has_nulls_ = TypedBufferBuilder<bool>(ctx->memory_pool());
    return Status::OK();
  }

  Status Resize(int64_t new_num_groups) override {
    auto added_groups = new_num_groups - num_groups_;
    num_groups_ = new_num_groups;
    RETURN_NOT_OK(mins_.Append(added_groups, AntiExtrema<CType>::anti_min()));
    RETURN_NOT_OK(maxes_.Append(added_groups, AntiExtrema<CType>::anti_max()));
    RETURN_NOT_OK(has_values_.Append(added_groups, false));
    RETURN_NOT_OK(has_nulls_.Append(added_groups, false));
    return Status::OK();
  }

  Status Consume(const ExecBatch& batch) override {
    auto g = batch[1].array()->GetValues<uint32_t>(1);
    auto raw_mins = reinterpret_cast<CType*>(mins_.mutable_data());
    auto raw_maxes = reinterpret_cast<CType*>(maxes_.mutable_data());

    VisitArrayValuesInline<Type>(
        *batch[0].array(),
        [&](CType val) {
          raw_maxes[*g] = std::max(raw_maxes[*g], val);
          raw_mins[*g] = std::min(raw_mins[*g], val);
          BitUtil::SetBit(has_values_.mutable_data(), *g++);
        },
        [&] { BitUtil::SetBit(has_nulls_.mutable_data(), *g++); });
    return Status::OK();
  }

  Status Merge(GroupedAggregator&& raw_other,
               const ArrayData& group_id_mapping) override {
    auto other = checked_cast<GroupedMinMaxImpl*>(&raw_other);

    auto raw_mins = reinterpret_cast<CType*>(mins_.mutable_data());
    auto raw_maxes = reinterpret_cast<CType*>(maxes_.mutable_data());

    auto other_raw_mins = reinterpret_cast<const CType*>(other->mins_.mutable_data());
    auto other_raw_maxes = reinterpret_cast<const CType*>(other->maxes_.mutable_data());

    auto g = group_id_mapping.GetValues<uint32_t>(1);
    for (int64_t other_g = 0; other_g < group_id_mapping.length; ++other_g, ++g) {
      raw_mins[*g] = std::min(raw_mins[*g], other_raw_mins[other_g]);
      raw_maxes[*g] = std::max(raw_maxes[*g], other_raw_maxes[other_g]);

      if (BitUtil::GetBit(other->has_values_.data(), other_g)) {
        BitUtil::SetBit(has_values_.mutable_data(), *g);
      }
      if (BitUtil::GetBit(other->has_nulls_.data(), other_g)) {
        BitUtil::SetBit(has_nulls_.mutable_data(), *g);
      }
    }
    return Status::OK();
  }

  Result<Datum> Finalize() override {
    // aggregation for group is valid if there was at least one value in that group
    ARROW_ASSIGN_OR_RAISE(auto null_bitmap, has_values_.Finish());

    if (!options_.skip_nulls) {
      // ... and there were no nulls in that group
      ARROW_ASSIGN_OR_RAISE(auto has_nulls, has_nulls_.Finish());
      arrow::internal::BitmapAndNot(null_bitmap->data(), 0, has_nulls->data(), 0,
                                    num_groups_, 0, null_bitmap->mutable_data());
    }

    auto mins = ArrayData::Make(type_, num_groups_, {null_bitmap, nullptr});
    auto maxes = ArrayData::Make(type_, num_groups_, {std::move(null_bitmap), nullptr});
    ARROW_ASSIGN_OR_RAISE(mins->buffers[1], mins_.Finish());
    ARROW_ASSIGN_OR_RAISE(maxes->buffers[1], maxes_.Finish());

    return ArrayData::Make(out_type(), num_groups_, {nullptr},
                           {std::move(mins), std::move(maxes)});
  }

  std::shared_ptr<DataType> out_type() const override {
    return struct_({field("min", type_), field("max", type_)});
  }

  int64_t num_groups_;
  TypedBufferBuilder<CType> mins_, maxes_;
  TypedBufferBuilder<bool> has_values_, has_nulls_;
  std::shared_ptr<DataType> type_;
  ScalarAggregateOptions options_;
};

template <typename T>
Result<std::unique_ptr<KernelState>> MinMaxInit(KernelContext* ctx,
                                                const KernelInitArgs& args) {
  ARROW_ASSIGN_OR_RAISE(auto impl, HashAggregateInit<GroupedMinMaxImpl<T>>(ctx, args));
  static_cast<GroupedMinMaxImpl<T>*>(impl.get())->type_ = args.inputs[0].type;
  return std::move(impl);
}

struct GroupedMinMaxFactory {
  template <typename T>
  enable_if_number<T, Status> Visit(const T&) {
    kernel = MakeKernel(std::move(argument_type), MinMaxInit<T>);
    return Status::OK();
  }

  template <typename T>
  enable_if_decimal<T, Status> Visit(const T&) {
    kernel = MakeKernel(std::move(argument_type), MinMaxInit<T>);
    return Status::OK();
  }

  Status Visit(const HalfFloatType& type) {
    return Status::NotImplemented("Computing min/max of data of type ", type);
  }

  Status Visit(const DataType& type) {
    return Status::NotImplemented("Computing min/max of data of type ", type);
  }

  static Result<HashAggregateKernel> Make(const std::shared_ptr<DataType>& type) {
    GroupedMinMaxFactory factory;
    factory.argument_type = InputType::Array(type->id());
    RETURN_NOT_OK(VisitTypeInline(*type, &factory));
    return std::move(factory.kernel);
  }

  HashAggregateKernel kernel;
  InputType argument_type;
};

// ----------------------------------------------------------------------
// Any/All implementation

template <typename Impl>
struct GroupedBooleanAggregator : public GroupedAggregator {
  Status Init(ExecContext* ctx, const FunctionOptions* options) override {
    options_ = checked_cast<const ScalarAggregateOptions&>(*options);
    pool_ = ctx->memory_pool();
    seen_ = TypedBufferBuilder<bool>(pool_);
    no_nulls_ = TypedBufferBuilder<bool>(pool_);
    counts_ = TypedBufferBuilder<int64_t>(pool_);
    return Status::OK();
  }

  Status Resize(int64_t new_num_groups) override {
    auto added_groups = new_num_groups - num_groups_;
    num_groups_ = new_num_groups;
    RETURN_NOT_OK(seen_.Append(added_groups, Impl::NullValue()));
    RETURN_NOT_OK(no_nulls_.Append(added_groups, true));
    return counts_.Append(added_groups, 0);
  }

  Status Consume(const ExecBatch& batch) override {
    uint8_t* seen = seen_.mutable_data();
    uint8_t* no_nulls = no_nulls_.mutable_data();
    int64_t* counts = counts_.mutable_data();
    const auto& input = *batch[0].array();
    auto g = batch[1].array()->GetValues<uint32_t>(1);

    if (input.MayHaveNulls()) {
      const uint8_t* bitmap = input.buffers[1]->data();
      arrow::internal::VisitBitBlocksVoid(
          input.buffers[0], input.offset, input.length,
          [&](int64_t position) {
            counts[*g]++;
            Impl::UpdateGroupWith(seen, *g, BitUtil::GetBit(bitmap, position));
            g++;
          },
          [&] { BitUtil::SetBitTo(no_nulls, *g++, false); });
    } else {
      arrow::internal::VisitBitBlocksVoid(
          input.buffers[1], input.offset, input.length,
          [&](int64_t) {
            Impl::UpdateGroupWith(seen, *g, true);
            counts[*g++]++;
          },
          [&]() {
            Impl::UpdateGroupWith(seen, *g, false);
            counts[*g++]++;
          });
    }
    return Status::OK();
  }

  Status Merge(GroupedAggregator&& raw_other,
               const ArrayData& group_id_mapping) override {
    auto other = checked_cast<GroupedBooleanAggregator<Impl>*>(&raw_other);

    uint8_t* seen = seen_.mutable_data();
    uint8_t* no_nulls = no_nulls_.mutable_data();
    int64_t* counts = counts_.mutable_data();

    const uint8_t* other_seen = other->seen_.mutable_data();
    const uint8_t* other_no_nulls = other->no_nulls_.mutable_data();
    const int64_t* other_counts = other->counts_.mutable_data();

    auto g = group_id_mapping.GetValues<uint32_t>(1);
    for (int64_t other_g = 0; other_g < group_id_mapping.length; ++other_g, ++g) {
      counts[*g] += other_counts[other_g];
      Impl::UpdateGroupWith(seen, *g, BitUtil::GetBit(other_seen, other_g));
      BitUtil::SetBitTo(
          no_nulls, *g,
          BitUtil::GetBit(no_nulls, *g) && BitUtil::GetBit(other_no_nulls, other_g));
    }
    return Status::OK();
  }

  Result<Datum> Finalize() override {
    std::shared_ptr<Buffer> null_bitmap;
    const int64_t* counts = counts_.data();
    int64_t null_count = 0;

    for (int64_t i = 0; i < num_groups_; ++i) {
      if (counts[i] >= options_.min_count) continue;

      if (null_bitmap == nullptr) {
        ARROW_ASSIGN_OR_RAISE(null_bitmap, AllocateBitmap(num_groups_, pool_));
        BitUtil::SetBitsTo(null_bitmap->mutable_data(), 0, num_groups_, true);
      }

      null_count += 1;
      BitUtil::SetBitTo(null_bitmap->mutable_data(), i, false);
    }

    ARROW_ASSIGN_OR_RAISE(auto seen, seen_.Finish());
    if (!options_.skip_nulls) {
      null_count = kUnknownNullCount;
      ARROW_ASSIGN_OR_RAISE(auto no_nulls, no_nulls_.Finish());
      Impl::AdjustForMinCount(no_nulls->mutable_data(), seen->data(), num_groups_);
      if (null_bitmap) {
        arrow::internal::BitmapAnd(null_bitmap->data(), /*left_offset=*/0,
                                   no_nulls->data(), /*right_offset=*/0, num_groups_,
                                   /*out_offset=*/0, null_bitmap->mutable_data());
      } else {
        null_bitmap = std::move(no_nulls);
      }
    }

    return ArrayData::Make(out_type(), num_groups_,
                           {std::move(null_bitmap), std::move(seen)}, null_count);
  }

  std::shared_ptr<DataType> out_type() const override { return boolean(); }

  int64_t num_groups_ = 0;
  ScalarAggregateOptions options_;
  TypedBufferBuilder<bool> seen_, no_nulls_;
  TypedBufferBuilder<int64_t> counts_;
  MemoryPool* pool_;
};

struct GroupedAnyImpl : public GroupedBooleanAggregator<GroupedAnyImpl> {
  // The default value for a group.
  static bool NullValue() { return false; }

  // Update the value for a group given an observation.
  static void UpdateGroupWith(uint8_t* seen, uint32_t g, bool value) {
    if (!BitUtil::GetBit(seen, g) && value) {
      BitUtil::SetBit(seen, g);
    }
  }

  // Combine the array of observed nulls with the array of group values.
  static void AdjustForMinCount(uint8_t* no_nulls, const uint8_t* seen,
                                int64_t num_groups) {
    arrow::internal::BitmapOr(no_nulls, /*left_offset=*/0, seen, /*right_offset=*/0,
                              num_groups, /*out_offset=*/0, no_nulls);
  }
};

struct GroupedAllImpl : public GroupedBooleanAggregator<GroupedAllImpl> {
  static bool NullValue() { return true; }

  static void UpdateGroupWith(uint8_t* seen, uint32_t g, bool value) {
    if (!value) {
      BitUtil::ClearBit(seen, g);
    }
  }

  static void AdjustForMinCount(uint8_t* no_nulls, const uint8_t* seen,
                                int64_t num_groups) {
    arrow::internal::BitmapOrNot(no_nulls, /*left_offset=*/0, seen, /*right_offset=*/0,
                                 num_groups, /*out_offset=*/0, no_nulls);
  }
};

// ----------------------------------------------------------------------
// CountDistinct/Distinct implementation

struct GroupedCountDistinctImpl : public GroupedAggregator {
  Status Init(ExecContext* ctx, const FunctionOptions* options) override {
    ctx_ = ctx;
    pool_ = ctx->memory_pool();
    return Status::OK();
  }

  Status Resize(int64_t new_num_groups) override {
    num_groups_ = new_num_groups;
    return Status::OK();
  }

  Status Consume(const ExecBatch& batch) override {
    return grouper_->Consume(batch).status();
  }

  Status Merge(GroupedAggregator&& raw_other,
               const ArrayData& group_id_mapping) override {
    auto other = checked_cast<GroupedCountDistinctImpl*>(&raw_other);

    // Get (value, group_id) pairs, then translate the group IDs and consume them
    // ourselves
    ARROW_ASSIGN_OR_RAISE(auto uniques, other->grouper_->GetUniques());
    ARROW_ASSIGN_OR_RAISE(auto remapped_g,
                          AllocateBuffer(uniques.length * sizeof(uint32_t), pool_));

    const auto* g_mapping = group_id_mapping.GetValues<uint32_t>(1);
    const auto* other_g = uniques[1].array()->GetValues<uint32_t>(1);
    auto* g = reinterpret_cast<uint32_t*>(remapped_g->mutable_data());

    for (int64_t i = 0; i < uniques.length; i++) {
      g[i] = g_mapping[other_g[i]];
    }
    uniques.values[1] =
        ArrayData::Make(uint32(), uniques.length, {nullptr, std::move(remapped_g)});

    return Consume(std::move(uniques));
  }

  Result<Datum> Finalize() override {
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Buffer> values,
                          AllocateBuffer(num_groups_ * sizeof(int64_t), pool_));
    int64_t* counts = reinterpret_cast<int64_t*>(values->mutable_data());
    std::fill(counts, counts + num_groups_, 0);

    ARROW_ASSIGN_OR_RAISE(auto uniques, grouper_->GetUniques());
    auto* g = uniques[1].array()->GetValues<uint32_t>(1);
    for (int64_t i = 0; i < uniques.length; i++) {
      counts[g[i]]++;
    }

    return ArrayData::Make(int64(), num_groups_, {nullptr, std::move(values)},
                           /*null_count=*/0);
  }

  std::shared_ptr<DataType> out_type() const override { return int64(); }

  ExecContext* ctx_;
  MemoryPool* pool_;
  int64_t num_groups_;
  std::unique_ptr<Grouper> grouper_;
  std::shared_ptr<DataType> out_type_;
};

struct GroupedDistinctImpl : public GroupedCountDistinctImpl {
  Result<Datum> Finalize() override {
    ARROW_ASSIGN_OR_RAISE(auto uniques, grouper_->GetUniques());
    ARROW_ASSIGN_OR_RAISE(auto groupings, grouper_->MakeGroupings(
                                              *uniques[1].array_as<UInt32Array>(),
                                              static_cast<uint32_t>(num_groups_), ctx_));
    return grouper_->ApplyGroupings(*groupings, *uniques[0].make_array(), ctx_);
  }

  std::shared_ptr<DataType> out_type() const override { return list(out_type_); }
};

template <typename Impl>
Result<std::unique_ptr<KernelState>> GroupedDistinctInit(KernelContext* ctx,
                                                         const KernelInitArgs& args) {
  ARROW_ASSIGN_OR_RAISE(auto impl, HashAggregateInit<Impl>(ctx, args));
  auto instance = static_cast<Impl*>(impl.get());
  instance->out_type_ = args.inputs[0].type;
  ARROW_ASSIGN_OR_RAISE(instance->grouper_,
                        Grouper::Make(args.inputs, ctx->exec_context()));
  return std::move(impl);
}

}  // namespace

Result<std::vector<const HashAggregateKernel*>> GetKernels(
    ExecContext* ctx, const std::vector<Aggregate>& aggregates,
    const std::vector<ValueDescr>& in_descrs) {
  if (aggregates.size() != in_descrs.size()) {
    return Status::Invalid(aggregates.size(), " aggregate functions were specified but ",
                           in_descrs.size(), " arguments were provided.");
  }

  std::vector<const HashAggregateKernel*> kernels(in_descrs.size());

  for (size_t i = 0; i < aggregates.size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto function,
                          ctx->func_registry()->GetFunction(aggregates[i].function));
    ARROW_ASSIGN_OR_RAISE(
        const Kernel* kernel,
        function->DispatchExact({in_descrs[i], ValueDescr::Array(uint32())}));
    kernels[i] = static_cast<const HashAggregateKernel*>(kernel);
  }
  return kernels;
}

Result<std::vector<std::unique_ptr<KernelState>>> InitKernels(
    const std::vector<const HashAggregateKernel*>& kernels, ExecContext* ctx,
    const std::vector<Aggregate>& aggregates, const std::vector<ValueDescr>& in_descrs) {
  std::vector<std::unique_ptr<KernelState>> states(kernels.size());

  for (size_t i = 0; i < aggregates.size(); ++i) {
    auto options = aggregates[i].options;

    if (options == nullptr) {
      // use known default options for the named function if possible
      auto maybe_function = ctx->func_registry()->GetFunction(aggregates[i].function);
      if (maybe_function.ok()) {
        options = maybe_function.ValueOrDie()->default_options();
      }
    }

    KernelContext kernel_ctx{ctx};
    ARROW_ASSIGN_OR_RAISE(
        states[i],
        kernels[i]->init(&kernel_ctx, KernelInitArgs{kernels[i],
                                                     {
                                                         in_descrs[i],
                                                         ValueDescr::Array(uint32()),
                                                     },
                                                     options}));
  }

  return std::move(states);
}

Result<FieldVector> ResolveKernels(
    const std::vector<Aggregate>& aggregates,
    const std::vector<const HashAggregateKernel*>& kernels,
    const std::vector<std::unique_ptr<KernelState>>& states, ExecContext* ctx,
    const std::vector<ValueDescr>& descrs) {
  FieldVector fields(descrs.size());

  for (size_t i = 0; i < kernels.size(); ++i) {
    KernelContext kernel_ctx{ctx};
    kernel_ctx.SetState(states[i].get());

    ARROW_ASSIGN_OR_RAISE(auto descr, kernels[i]->signature->out_type().Resolve(
                                          &kernel_ctx, {
                                                           descrs[i],
                                                           ValueDescr::Array(uint32()),
                                                       }));
    fields[i] = field(aggregates[i].function, std::move(descr.type));
  }
  return fields;
}

Result<std::unique_ptr<Grouper>> Grouper::Make(const std::vector<ValueDescr>& descrs,
                                               ExecContext* ctx) {
  if (GrouperFastImpl::CanUse(descrs)) {
    return GrouperFastImpl::Make(descrs, ctx);
  }
  return GrouperImpl::Make(descrs, ctx);
}

Result<Datum> GroupBy(const std::vector<Datum>& arguments, const std::vector<Datum>& keys,
                      const std::vector<Aggregate>& aggregates, bool use_threads,
                      ExecContext* ctx) {
  auto task_group =
      use_threads
          ? arrow::internal::TaskGroup::MakeThreaded(arrow::internal::GetCpuThreadPool())
          : arrow::internal::TaskGroup::MakeSerial();

  // Construct and initialize HashAggregateKernels
  ARROW_ASSIGN_OR_RAISE(auto argument_descrs,
                        ExecBatch::Make(arguments).Map(
                            [](ExecBatch batch) { return batch.GetDescriptors(); }));

  ARROW_ASSIGN_OR_RAISE(auto kernels, GetKernels(ctx, aggregates, argument_descrs));

  std::vector<std::vector<std::unique_ptr<KernelState>>> states(
      task_group->parallelism());
  for (auto& state : states) {
    ARROW_ASSIGN_OR_RAISE(state, InitKernels(kernels, ctx, aggregates, argument_descrs));
  }

  ARROW_ASSIGN_OR_RAISE(
      FieldVector out_fields,
      ResolveKernels(aggregates, kernels, states[0], ctx, argument_descrs));

  using arrow::compute::detail::ExecBatchIterator;

  ARROW_ASSIGN_OR_RAISE(auto argument_batch_iterator,
                        ExecBatchIterator::Make(arguments, ctx->exec_chunksize()));

  // Construct Groupers
  ARROW_ASSIGN_OR_RAISE(auto key_descrs, ExecBatch::Make(keys).Map([](ExecBatch batch) {
    return batch.GetDescriptors();
  }));

  std::vector<std::unique_ptr<Grouper>> groupers(task_group->parallelism());
  for (auto& grouper : groupers) {
    ARROW_ASSIGN_OR_RAISE(grouper, Grouper::Make(key_descrs, ctx));
  }

  std::mutex mutex;
  std::unordered_map<std::thread::id, size_t> thread_ids;

  int i = 0;
  for (ValueDescr& key_descr : key_descrs) {
    out_fields.push_back(field("key_" + std::to_string(i++), std::move(key_descr.type)));
  }

  ARROW_ASSIGN_OR_RAISE(auto key_batch_iterator,
                        ExecBatchIterator::Make(keys, ctx->exec_chunksize()));

  // start "streaming" execution
  ExecBatch key_batch, argument_batch;
  while (argument_batch_iterator->Next(&argument_batch) &&
         key_batch_iterator->Next(&key_batch)) {
    if (key_batch.length == 0) continue;

    task_group->Append([&, key_batch, argument_batch] {
      size_t thread_index;
      {
        std::unique_lock<std::mutex> lock(mutex);
        auto it = thread_ids.emplace(std::this_thread::get_id(), thread_ids.size()).first;
        thread_index = it->second;
        DCHECK_LT(static_cast<int>(thread_index), task_group->parallelism());
      }

      auto grouper = groupers[thread_index].get();

      // compute a batch of group ids
      ARROW_ASSIGN_OR_RAISE(Datum id_batch, grouper->Consume(key_batch));

      // consume group ids with HashAggregateKernels
      for (size_t i = 0; i < kernels.size(); ++i) {
        KernelContext batch_ctx{ctx};
        batch_ctx.SetState(states[thread_index][i].get());
        ARROW_ASSIGN_OR_RAISE(auto batch, ExecBatch::Make({argument_batch[i], id_batch}));
        RETURN_NOT_OK(kernels[i]->resize(&batch_ctx, grouper->num_groups()));
        RETURN_NOT_OK(kernels[i]->consume(&batch_ctx, batch));
      }

      return Status::OK();
    });
  }

  RETURN_NOT_OK(task_group->Finish());

  // Merge if necessary
  for (size_t thread_index = 1; thread_index < thread_ids.size(); ++thread_index) {
    ARROW_ASSIGN_OR_RAISE(ExecBatch other_keys, groupers[thread_index]->GetUniques());
    ARROW_ASSIGN_OR_RAISE(Datum transposition, groupers[0]->Consume(other_keys));
    groupers[thread_index].reset();

    for (size_t i = 0; i < kernels.size(); ++i) {
      KernelContext batch_ctx{ctx};
      batch_ctx.SetState(states[0][i].get());

      RETURN_NOT_OK(kernels[i]->resize(&batch_ctx, groupers[0]->num_groups()));
      RETURN_NOT_OK(kernels[i]->merge(&batch_ctx, std::move(*states[thread_index][i]),
                                      *transposition.array()));
      states[thread_index][i].reset();
    }
  }

  // Finalize output
  ArrayDataVector out_data(arguments.size() + keys.size());
  auto it = out_data.begin();

  for (size_t i = 0; i < kernels.size(); ++i) {
    KernelContext batch_ctx{ctx};
    batch_ctx.SetState(states[0][i].get());
    Datum out;
    RETURN_NOT_OK(kernels[i]->finalize(&batch_ctx, &out));
    *it++ = out.array();
  }

  ARROW_ASSIGN_OR_RAISE(ExecBatch out_keys, groupers[0]->GetUniques());
  for (const auto& key : out_keys.values) {
    *it++ = key.array();
  }

  int64_t length = out_data[0]->length;
  return ArrayData::Make(struct_(std::move(out_fields)), length,
                         {/*null_bitmap=*/nullptr}, std::move(out_data),
                         /*null_count=*/0);
}

Result<std::shared_ptr<ListArray>> Grouper::ApplyGroupings(const ListArray& groupings,
                                                           const Array& array,
                                                           ExecContext* ctx) {
  ARROW_ASSIGN_OR_RAISE(Datum sorted,
                        compute::Take(array, groupings.data()->child_data[0],
                                      TakeOptions::NoBoundsCheck(), ctx));

  return std::make_shared<ListArray>(list(array.type()), groupings.length(),
                                     groupings.value_offsets(), sorted.make_array());
}

Result<std::shared_ptr<ListArray>> Grouper::MakeGroupings(const UInt32Array& ids,
                                                          uint32_t num_groups,
                                                          ExecContext* ctx) {
  if (ids.null_count() != 0) {
    return Status::Invalid("MakeGroupings with null ids");
  }

  ARROW_ASSIGN_OR_RAISE(auto offsets, AllocateBuffer(sizeof(int32_t) * (num_groups + 1),
                                                     ctx->memory_pool()));
  auto raw_offsets = reinterpret_cast<int32_t*>(offsets->mutable_data());

  std::memset(raw_offsets, 0, offsets->size());
  for (int i = 0; i < ids.length(); ++i) {
    DCHECK_LT(ids.Value(i), num_groups);
    raw_offsets[ids.Value(i)] += 1;
  }
  int32_t length = 0;
  for (uint32_t id = 0; id < num_groups; ++id) {
    auto offset = raw_offsets[id];
    raw_offsets[id] = length;
    length += offset;
  }
  raw_offsets[num_groups] = length;
  DCHECK_EQ(ids.length(), length);

  ARROW_ASSIGN_OR_RAISE(auto offsets_copy,
                        offsets->CopySlice(0, offsets->size(), ctx->memory_pool()));
  raw_offsets = reinterpret_cast<int32_t*>(offsets_copy->mutable_data());

  ARROW_ASSIGN_OR_RAISE(auto sort_indices, AllocateBuffer(sizeof(int32_t) * ids.length(),
                                                          ctx->memory_pool()));
  auto raw_sort_indices = reinterpret_cast<int32_t*>(sort_indices->mutable_data());
  for (int i = 0; i < ids.length(); ++i) {
    raw_sort_indices[raw_offsets[ids.Value(i)]++] = i;
  }

  return std::make_shared<ListArray>(
      list(int32()), num_groups, std::move(offsets),
      std::make_shared<Int32Array>(ids.length(), std::move(sort_indices)));
}

namespace {
const FunctionDoc hash_count_doc{"Count the number of null / non-null values",
                                 ("By default, non-null values are counted.\n"
                                  "This can be changed through ScalarAggregateOptions."),
                                 {"array", "group_id_array"},
                                 "CountOptions"};

const FunctionDoc hash_sum_doc{"Sum values of a numeric array",
                               ("Null values are ignored."),
                               {"array", "group_id_array"},
                               "ScalarAggregateOptions"};

const FunctionDoc hash_product_doc{
    "Compute product of values of a numeric array",
    ("Null values are ignored.\n"
     "Overflow will wrap around as if the calculation was done with unsigned integers."),
    {"array", "group_id_array"},
    "ScalarAggregateOptions"};

const FunctionDoc hash_mean_doc{
    "Average values of a numeric array",
    ("Null values are ignored.\n"
     "For integers and floats, NaN is returned if min_count = 0 and\n"
     "there are no values. For decimals, null is returned instead."),
    {"array", "group_id_array"},
    "ScalarAggregateOptions"};

const FunctionDoc hash_stddev_doc{
    "Calculate the standard deviation of a numeric array",
    ("The number of degrees of freedom can be controlled using VarianceOptions.\n"
     "By default (`ddof` = 0), the population standard deviation is calculated.\n"
     "Nulls are ignored.  If there are not enough non-null values in the array\n"
     "to satisfy `ddof`, null is returned."),
    {"array", "group_id_array"}};

const FunctionDoc hash_variance_doc{
    "Calculate the variance of a numeric array",
    ("The number of degrees of freedom can be controlled using VarianceOptions.\n"
     "By default (`ddof` = 0), the population variance is calculated.\n"
     "Nulls are ignored.  If there are not enough non-null values in the array\n"
     "to satisfy `ddof`, null is returned."),
    {"array", "group_id_array"}};

const FunctionDoc hash_tdigest_doc{
    "Calculate approximate quantiles of a numeric array with the T-Digest algorithm",
    ("By default, the 0.5 quantile (median) is returned.\n"
     "Nulls and NaNs are ignored.\n"
     "A null array is returned if there are no valid data points."),
    {"array", "group_id_array"},
    "TDigestOptions"};

const FunctionDoc hash_min_max_doc{
    "Compute the minimum and maximum values of a numeric array",
    ("Null values are ignored by default.\n"
     "This can be changed through ScalarAggregateOptions."),
    {"array", "group_id_array"},
    "ScalarAggregateOptions"};

const FunctionDoc hash_any_doc{"Test whether any element evaluates to true",
                               ("Null values are ignored."),
                               {"array", "group_id_array"},
                               "ScalarAggregateOptions"};

const FunctionDoc hash_all_doc{"Test whether all elements evaluate to true",
                               ("Null values are ignored."),
                               {"array", "group_id_array"},
                               "ScalarAggregateOptions"};

const FunctionDoc hash_count_distinct_doc{
    "Count the distinct values in each group",
    ("Nulls are counted. NaNs and signed zeroes are not normalized."),
    {"array", "group_id_array"}};

const FunctionDoc hash_distinct_doc{
    "Keep the distinct values in each group",
    ("Nulls are kept. NaNs and signed zeroes are not normalized."),
    {"array", "group_id_array"}};
}  // namespace

void RegisterHashAggregateBasic(FunctionRegistry* registry) {
  static auto default_scalar_aggregate_options = ScalarAggregateOptions::Defaults();
  static auto default_tdigest_options = TDigestOptions::Defaults();
  static auto default_variance_options = VarianceOptions::Defaults();

  {
    static auto default_count_options = CountOptions::Defaults();
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_count", Arity::Binary(), &hash_count_doc, &default_count_options);

    DCHECK_OK(func->AddKernel(
        MakeKernel(ValueDescr::ARRAY, HashAggregateInit<GroupedCountImpl>)));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_sum", Arity::Binary(), &hash_sum_doc, &default_scalar_aggregate_options);
    DCHECK_OK(AddHashAggKernels({boolean()}, GroupedSumFactory::Make, func.get()));
    DCHECK_OK(AddHashAggKernels(SignedIntTypes(), GroupedSumFactory::Make, func.get()));
    DCHECK_OK(AddHashAggKernels(UnsignedIntTypes(), GroupedSumFactory::Make, func.get()));
    DCHECK_OK(
        AddHashAggKernels(FloatingPointTypes(), GroupedSumFactory::Make, func.get()));
    // Type parameters are ignored
    DCHECK_OK(AddHashAggKernels({decimal128(1, 1), decimal256(1, 1)},
                                GroupedSumFactory::Make, func.get()));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_product", Arity::Binary(), &hash_product_doc,
        &default_scalar_aggregate_options);
    DCHECK_OK(AddHashAggKernels({boolean()}, GroupedProductFactory::Make, func.get()));
    DCHECK_OK(
        AddHashAggKernels(SignedIntTypes(), GroupedProductFactory::Make, func.get()));
    DCHECK_OK(
        AddHashAggKernels(UnsignedIntTypes(), GroupedProductFactory::Make, func.get()));
    DCHECK_OK(
        AddHashAggKernels(FloatingPointTypes(), GroupedProductFactory::Make, func.get()));
    // Type parameters are ignored
    DCHECK_OK(AddHashAggKernels({decimal128(1, 1), decimal256(1, 1)},
                                GroupedProductFactory::Make, func.get()));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_mean", Arity::Binary(), &hash_mean_doc, &default_scalar_aggregate_options);
    DCHECK_OK(AddHashAggKernels({boolean()}, GroupedMeanFactory::Make, func.get()));
    DCHECK_OK(AddHashAggKernels(SignedIntTypes(), GroupedMeanFactory::Make, func.get()));
    DCHECK_OK(
        AddHashAggKernels(UnsignedIntTypes(), GroupedMeanFactory::Make, func.get()));
    DCHECK_OK(
        AddHashAggKernels(FloatingPointTypes(), GroupedMeanFactory::Make, func.get()));
    // Type parameters are ignored
    DCHECK_OK(AddHashAggKernels({decimal128(1, 1), decimal256(1, 1)},
                                GroupedMeanFactory::Make, func.get()));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_stddev", Arity::Binary(), &hash_stddev_doc, &default_variance_options);
    DCHECK_OK(AddHashAggKernels(SignedIntTypes(),
                                GroupedVarStdFactory<VarOrStd::Std>::Make, func.get()));
    DCHECK_OK(AddHashAggKernels(UnsignedIntTypes(),
                                GroupedVarStdFactory<VarOrStd::Std>::Make, func.get()));
    DCHECK_OK(AddHashAggKernels(FloatingPointTypes(),
                                GroupedVarStdFactory<VarOrStd::Std>::Make, func.get()));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_variance", Arity::Binary(), &hash_variance_doc, &default_variance_options);
    DCHECK_OK(AddHashAggKernels(SignedIntTypes(),
                                GroupedVarStdFactory<VarOrStd::Var>::Make, func.get()));
    DCHECK_OK(AddHashAggKernels(UnsignedIntTypes(),
                                GroupedVarStdFactory<VarOrStd::Var>::Make, func.get()));
    DCHECK_OK(AddHashAggKernels(FloatingPointTypes(),
                                GroupedVarStdFactory<VarOrStd::Var>::Make, func.get()));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_tdigest", Arity::Binary(), &hash_tdigest_doc, &default_tdigest_options);
    DCHECK_OK(
        AddHashAggKernels(SignedIntTypes(), GroupedTDigestFactory::Make, func.get()));
    DCHECK_OK(
        AddHashAggKernels(UnsignedIntTypes(), GroupedTDigestFactory::Make, func.get()));
    DCHECK_OK(
        AddHashAggKernels(FloatingPointTypes(), GroupedTDigestFactory::Make, func.get()));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_min_max", Arity::Binary(), &hash_min_max_doc,
        &default_scalar_aggregate_options);
    DCHECK_OK(AddHashAggKernels(NumericTypes(), GroupedMinMaxFactory::Make, func.get()));
    // Type parameters are ignored
    DCHECK_OK(AddHashAggKernels({decimal128(1, 1), decimal256(1, 1)},
                                GroupedMinMaxFactory::Make, func.get()));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_any", Arity::Binary(), &hash_any_doc, &default_scalar_aggregate_options);
    DCHECK_OK(func->AddKernel(MakeKernel(boolean(), HashAggregateInit<GroupedAnyImpl>)));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_all", Arity::Binary(), &hash_all_doc, &default_scalar_aggregate_options);
    DCHECK_OK(func->AddKernel(MakeKernel(boolean(), HashAggregateInit<GroupedAllImpl>)));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>(
        "hash_count_distinct", Arity::Binary(), &hash_count_distinct_doc);
    DCHECK_OK(func->AddKernel(
        MakeKernel(ValueDescr::ARRAY, GroupedDistinctInit<GroupedCountDistinctImpl>)));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<HashAggregateFunction>("hash_distinct", Arity::Binary(),
                                                        &hash_distinct_doc);
    DCHECK_OK(func->AddKernel(
        MakeKernel(ValueDescr::ARRAY, GroupedDistinctInit<GroupedDistinctImpl>)));
    DCHECK_OK(registry->AddFunction(std::move(func)));
  }
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
