clean:
	cargo clean

build:
	cargo build

test:
	cargo test

lint:
	cargo clippy --all-targets --all-features -- -D warnings

build_release:
	cargo build --release
