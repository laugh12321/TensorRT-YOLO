#include "deploy/core/memory.hpp"

namespace deploy {

template <typename MemType>
MemoryManager<MemType>::MemoryManager(MemoryManager&& other) noexcept {
    free();
    other.host_   = nullptr;
    other.device_ = nullptr;
    other.size_   = 0;
}

template <typename MemType>
MemoryManager<MemType>& MemoryManager<MemType>::operator=(MemoryManager&& other) noexcept {
    if (this != &other) {
        free();
        other.host_   = nullptr;
        other.device_ = nullptr;
        other.size_   = 0;
    }
    return *this;
}

template <typename MemType>
void MemoryManager<MemType>::free() {
    MemType::free(host_, device_);
    host_   = nullptr;
    device_ = nullptr;
    size_   = 0;
}

template <typename MemType>
void MemoryManager<MemType>::allocate(size_t size) {
    if (size > size_) {
        free();
        MemType::allocate(host_, device_, size);
        size_ = size;
    }
}

template class MemoryManager<PinnedMemory>;

}  // namespace deploy