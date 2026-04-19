cargo build --bin powergrid-server

trap 'kill $(jobs -p) 2>/dev/null' EXIT

cargo run --bin powergrid-reimagined -- --name brent --color red --url ws://localhost:3000/ws &
sleep 1
cargo run --bin powergrid-reimagined -- --name brad --color blue --url ws://localhost:3000/ws &
sleep 1
cargo run --bin powergrid-reimagined -- --name nick --color green --url ws://localhost:3000/ws &
sleep 1
RUST_LOG=info cargo run --bin powergrid-server
