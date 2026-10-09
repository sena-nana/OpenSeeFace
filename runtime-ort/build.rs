fn main() {
    // The only Windows ORT prebuilt bundles the DirectML EP. Inference never
    // registers it, so delay-load DirectML and D3D12: neither is mapped at run
    // time and DirectML.dll does not ship with the tracker.
    let windows_msvc = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if windows_msvc {
        for dll in ["DirectML.dll", "d3d12.dll"] {
            println!("cargo:rustc-link-arg-bins=/DELAYLOAD:{dll}");
        }
        println!("cargo:rustc-link-arg-bins=delayimp.lib");
    }
}
