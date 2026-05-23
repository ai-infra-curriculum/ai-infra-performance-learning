# Ex 01: nsys trace + bottleneck identification

Profile a ResNet-50 training step with nsys. Identify the top 3 hotspot
kernels. For each: report total time, % of step time, kernel name.

Add NVTX ranges around forward/backward/optimizer. Re-profile.

Deliverable: screenshot + `HOTSPOTS.md` listing the 3 hotspots + your fix hypothesis.
