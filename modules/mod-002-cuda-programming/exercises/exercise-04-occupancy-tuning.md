# Ex 04: Occupancy Tuning

Take your tiled matmul. Use `cudaOccupancyMaxPotentialBlockSize` to find a
starting block size. Then sweep block sizes (64, 128, 256, 512) and tile
sizes; measure actual achieved occupancy + TFLOPS.

Deliverable: `OCCUPANCY_REPORT.md` with sweep results + recommendation.
