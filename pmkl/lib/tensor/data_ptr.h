#pragma once

#include <memory>

#include "device.h"

namespace pmkl {
namespace memory {

using DeleterFnPtr = void (*)(void *);
void delete_nothing(void *) {
}

class UniqueVoidPtr {
private:
    // Lifetime tied to ctx_
    void *data_;
    std::unique_ptr<void, DeleterFnPtr> ctx_;

public:
    UniqueVoidPtr() :
        data_(nullptr), ctx_(nullptr, &delete_nothing) {
    }
    explicit UniqueVoidPtr(void *data) :
        data_(data), ctx_(nullptr, &delete_nothing) {
    }
    UniqueVoidPtr(void *data, void *ctx, DeleterFnPtr ctx_deleter) :
        data_(data), ctx_(ctx, ctx_deleter ? ctx_deleter : &delete_nothing) {
    }
    void *operator->() const {
        return data_;
    }
    void clear() {
        ctx_ = nullptr;
        data_ = nullptr;
    }
    void *get() const {
        return data_;
    }
    void *get_context() const {
        return ctx_.get();
    }
    void *release_context() {
        return ctx_.release();
    }
    std::unique_ptr<void, DeleterFnPtr> &&move_context() {
        return std::move(ctx_);
    }
    [[nodiscard]] bool compare_exchange_deleter(
        DeleterFnPtr expected_deleter,
        DeleterFnPtr new_deleter) {
        if (get_deleter() != expected_deleter)
            return false;
        ctx_ = std::unique_ptr<void, DeleterFnPtr>(ctx_.release(), new_deleter);
        return true;
    }

    template <typename T>
    T *cast_context(DeleterFnPtr expected_deleter) const {
        if (get_deleter() != expected_deleter)
            return nullptr;
        return static_cast<T *>(get_context());
    }
    operator bool() const {
        return data_ || ctx_;
    }
    DeleterFnPtr get_deleter() const {
        return ctx_.get_deleter();
    }
};

inline bool operator==(const UniqueVoidPtr &sp, std::nullptr_t) noexcept {
    return !sp;
}
inline bool operator==(std::nullptr_t, const UniqueVoidPtr &sp) noexcept {
    return !sp;
}
inline bool operator!=(const UniqueVoidPtr &sp, std::nullptr_t) noexcept {
    return sp;
}
inline bool operator!=(std::nullptr_t, const UniqueVoidPtr &sp) noexcept {
    return sp;
}

// A DataPtr is a unique pointer (with an attached deleter and some
// context for the deleter) to some memory, which also records what
// device is for its data.
//
// nullptr DataPtrs can still have a nontrivial device; this allows
// us to treat zero-size allocations uniformly with non-zero allocations.
//
class DataPtr {
private:
    UniqueVoidPtr ptr_;
    Device device_;

public:
    // Choice of CPU here is arbitrary; if there's an "undefined" device
    // we could use that too
    DataPtr() :
        ptr_(), device_(DeviceType::CPU) {
    }
    DataPtr(void *data, Device device) :
        ptr_(data), device_(device) {
    }
    DataPtr(void *data, void *ctx, DeleterFnPtr ctx_deleter, Device device) :
        ptr_(data, ctx, ctx_deleter), device_(device) {
    }
    void *operator->() const {
        return ptr_.get();
    }
    void clear() {
        ptr_.clear();
    }
    void *get() const {
        return ptr_.get();
    }
    void *get_context() const {
        return ptr_.get_context();
    }
    void *release_context() {
        return ptr_.release_context();
    }
    std::unique_ptr<void, DeleterFnPtr> &&move_context() {
        return ptr_.move_context();
    }
    operator bool() const {
        return static_cast<bool>(ptr_);
    }
    template <typename T>
    T *cast_context(DeleterFnPtr expected_deleter) const {
        return ptr_.cast_context<T>(expected_deleter);
    }
    DeleterFnPtr get_deleter() const {
        return ptr_.get_deleter();
    }
    /**
   * Compare the deleter in a DataPtr to expected_deleter.
   * If it matches, replace the deleter with new_deleter
   * and return true; otherwise, does nothing and returns
   * false.
   *
   * In general, it is not safe to unconditionally set the
   * deleter on a DataPtr, because you don't know what
   * the deleter is, and thus will have a hard time properly
   * disposing of the deleter without storing the original
   * deleter (this is difficult to do, because DeleterFnPtr
   * is not a closure, and because the context on DataPtr is
   * only a single word, you generally don't have enough
   * space to store both the original deleter and its context).
   * However, in some cases, you know /exactly/ what the deleter
   * is, and you have a new deleter that manually wraps
   * the old one.  In this case, you can safely swap the deleter
   * after asserting that the deleters line up.
   *
   * What are the requirements on new_deleter?  It must still
   * properly dispose of the void* pointer passed in as its argument,
   * where void* is whatever the context of the original deleter
   * is.  So in general, you expect the new deleter to look something
   * like this:
   *
   *      [](void* ptr) {
   *        some_new_stuff(ptr);
   *        get_orig_allocator()->raw_deleter(ptr);
   *      }
   *
   * Note that it won't work to close over the original
   * allocator; you don't have enough space to do that!  Also,
   * it's unsafe to assume that the passed in pointer in
   * question is the memory pointer in question; it might not
   * be; be sure to read the source code of the Allocator
   * in question to confirm this.
   */
    [[nodiscard]] bool compare_exchange_deleter(
        DeleterFnPtr expected_deleter,
        DeleterFnPtr new_deleter) {
        return ptr_.compare_exchange_deleter(expected_deleter, new_deleter);
    }
    Device device() const {
        return device_;
    }
    // Unsafely mutates the device on a DataPtr.  Under normal use,
    // you should never actually need to call this function.
    // We need this for the implementation of the hack detailed
    void unsafe_set_device(Device device) {
        device_ = device;
    }
};

inline bool operator==(const DataPtr &dp, std::nullptr_t) noexcept {
    return !dp;
}
inline bool operator==(std::nullptr_t, const DataPtr &dp) noexcept {
    return !dp;
}
inline bool operator!=(const DataPtr &dp, std::nullptr_t) noexcept {
    return dp;
}
inline bool operator!=(std::nullptr_t, const DataPtr &dp) noexcept {
    return dp;
}

}
} // namespace pmkl::memory