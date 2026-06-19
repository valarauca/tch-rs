use libc::c_int;

extern "C" {
    /// Returns the number of Intel XPU (oneAPI) devices available.
    pub fn atc_xpu_device_count() -> c_int;

    /// Returns true if at least one Intel XPU (oneAPI) device is available.
    pub fn atc_xpu_is_available() -> c_int;

    /// Returns true if a Vulkan device is available.
    pub fn atc_vulkan_is_available() -> c_int;
}
