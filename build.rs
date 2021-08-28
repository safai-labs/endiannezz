#[rustversion::nightly]
fn main() {
    println!("cargo:rustc-cfg=unstable_feature");
    println!("cargo:rustc-cfg=const_generic");
}

#[rustversion::not(nightly)]
fn main() {}
