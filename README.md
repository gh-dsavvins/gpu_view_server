# gpu_view_server

Proof of concept: a view server that keeps a live order table in GPU memory and serves many
differently filtered, sorted and pivoted viewports every tick, compared against an all-cores CPU engine.

- `server/` C++20 server: order/market-data simulator, CPU engine, Vulkan GPU engine, WebSocket + protobuf API
- `server/shaders/` GLSL compute kernels (derived columns, 8-pass radix select, pivot)
- `client/` TypeScript browser client (Vite): live views, sort by header click, scrolling, pivots, CPU/GPU switch
- `proto/gvs.proto` wire protocol
- `doc/business_case.html` the case for management

## Build the server (Debian 13, Mac Pro)

```sh
sudo apt install build-essential cmake libvulkan-dev glslang-tools mesa-vulkan-drivers
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`-DGVS_GPU=OFF` builds the CPU engine only (no Vulkan needed).

## Benchmark

```sh
./build/server/gvs_bench --rows 4000000 --views 8,32,128 --ticks 10 --json bench.json
```

Runs the same simulated ticks and views through both engines, prints engine time per tick and checks that
the CPU and GPU results are identical (totals, viewport rows, pivot sums).

## Run the server

```sh
./build/server/gvs_server --rows 4000000 --load-views 64 --backend gpu --validate
```

`--load-views N` adds N synthetic views (random filters/sorts/pivots) evaluated every tick, standing in for
other users. `--validate` runs the other engine too and reports mismatches. `--help` lists all options.

## Run the client (Windows or Mac)

```sh
cd client
npm install
npm run dev
```

Open http://localhost:5173 and connect to `ws://<macpro>:8080/ws`.
Or build it (`npm run build`) and start the server with `--www client/dist`, then open `http://<macpro>:8080/`.

## How the CPU engine works

Tuned so the comparison is against the best CPU approach, not a naive one: blocks of 8 views share one
cache-blocked pass over the table, rows are split across all cores whatever the view count, and selection
is a two-pass 11-bit histogram select (skipping key bits that were constant last tick) with `nth_element`
only on the few candidates left in the edge buckets.

## How the GPU engine works

1. The whole table stays in GPU memory. Each tick uploads only changed cells and instrument prices.
2. Derived columns (mid, leaves, notional, distance to market in bps, P&L) are recalculated once for all rows.
3. Every view's sort columns are packed into a 64-bit key (40 bits of sort columns, 24 bits row index).
4. Radix select, 8 passes of 8 bits: each pass histograms the surviving candidates, accepts buckets that fall
   inside the viewport's rank window and keeps only the edge buckets. Up to 8 views run together per batch.
5. The selected keys (at most 256) are sorted in one workgroup. Row ids, totals and pivot cells go to host memory.

Both engines use the same keys, so results match bit for bit (`-ffp-contract=off` on the CPU, `precise` in
the shaders). Second and third sort columns share the leftover key bits, so very close values in those
columns can tie and fall back to row order.
