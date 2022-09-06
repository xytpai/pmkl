#pragma once

#include <sstream>
#include "exception.h"

namespace pmkl {

enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1, // CUDA.
    // NB: If you add more devices:
    //  - Change the implementations of DeviceTypeName and isValidDeviceType
    //    in DeviceType.cpp
    //  - Change the number below
    COMPILE_TIME_MAX_DEVICE_TYPES = 2,
};

using DeviceIndex = int8_t;

/// Represents a a compute device on which a tensor is located. A device is
/// uniquely identified by a type, which specifies the type of machine it is
/// (e.g. CPU or CUDA GPU), and a device index or ordinal, which identifies the
/// specific compute device when there is more than one of a certain type. The
/// device index is optional, and in its defaulted state represents (abstractly)
/// "the current device". Further, there are two constraints on the value of the
/// device index, if one is explicitly stored:
/// 1. A negative index represents the current device, a non-negative index
/// represents a specific, concrete device,
/// 2. When the device type is CPU, the device index must be zero.
struct Device final {
    using Type = DeviceType;

    Device(DeviceType type, DeviceIndex index = -1) :
        type_(type), index_(index) {
        CHECK_FAIL(
            index_ == -1 || index_ >= 0,
            "Device index must be -1 or non-negative, got ",
            (int)index_);
        CHECK_FAIL(
            !is_cpu() || index_ <= 0,
            "CPU device index must be -1 or zero, got ",
            (int)index_);
    }

    /// Returns true if the type and index of this `Device` matches that of
    /// `other`.
    bool operator==(const Device &other) const noexcept {
        return this->type_ == other.type_ && this->index_ == other.index_;
    }

    /// Returns true if the type or index of this `Device` differs from that of
    /// `other`.
    bool operator!=(const Device &other) const noexcept {
        return !(*this == other);
    }

    /// Sets the device index.
    void set_index(DeviceIndex index) {
        index_ = index;
    }

    /// Returns the type of device this is.
    DeviceType type() const noexcept {
        return type_;
    }

    /// Returns the optional index.
    DeviceIndex index() const noexcept {
        return index_;
    }

    /// Returns true if the device has a non-default index.
    bool has_index() const noexcept {
        return index_ != -1;
    }

    /// Return true if the device is of CUDA type.
    bool is_cuda() const noexcept {
        return type_ == DeviceType::CUDA;
    }

    /// Return true if the device is of CPU type.
    bool is_cpu() const noexcept {
        return type_ == DeviceType::CPU;
    }

private:
    DeviceType type_;
    DeviceIndex index_ = -1;
};

}; // namespace pmkl