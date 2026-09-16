# AstraSR — Round 3 Research Notes

## Focus
Stress-test the proposed differentiation against ESA OpenSR, then assess the feasibility of the remaining differentiation paths without selecting one for the human team.

## Findings

### 1. OpenSR already supplies the exact technical baseline
The public ESAOpenSR opensr-model repository implements latent-diffusion SR for RGB-NIR Sentinel-2 and exposes a 4x tensor path (128x128 -> 512x512). Its demo also generates an uncertainty map using multiple stochastic variations. opensr-utils handles tiling, overlap blending, stitching, CRS/transform preservation and GeoTIFF/COG output. Sources: https://github.com/ESAOpenSR/opensr-model ; https://github.com/ESAOpenSR/opensr-model/blob/main/demo.py

This means the current project's claimed combination of 4x SR + uncertainty + geospatial output is not a sufficient differentiator.

### 2. A generic benchmark study is also weak as novelty
OpenSR-test already contains 2.5m NAIP and SPOT x4 datasets plus Venµs x2 and Spain x4 datasets, and is explicitly designed for real-world Sentinel-2 SR benchmarking. Therefore “we benchmark our model against OpenSR on OpenSR-test” is valuable experimental evidence but should not itself be presented as novel research. Source: https://github.com/ESAOpenSR/opensr-test

### 3. India-specific validation is technically meaningful but data access is a constraint
Bhoonidhi is the official ISRO/NRSC EO data hub. It currently lists CartoSat-2S and CartoSat-3 FCC-NCC (MX) products for ordering, and states that remote-sensing data below 5m GSD are priced for non-government entities under Indian Space Policy 2023. Thus an Indian 2.5m/2m reference experiment is possible in principle, but the team should not promise it without confirming price, licensing, AOI availability and delivery time. Sources: https://bhoonidhi.nrsc.gov.in/bhoonidhi/home.html ; https://bhoonidhi.nrsc.gov.in/bhoonidhi/registration.html

### 4. Compute-tier differentiation is testable
The ESA OpenSR implementation is publicly runnable and already supports 100 diffusion sampling steps plus uncertainty sampling. That makes a measured comparison of inference latency, VRAM and throughput against a lightweight deterministic model technically meaningful. But it is only a contribution if the team actually measures the trade-off on a stated hardware tier. Source: https://github.com/ESAOpenSR/opensr-model

## Implication
The draft should stop describing “comparative benchmark study” alone as a differentiator. The human team needs to choose a concrete experimental contribution:
A) Indian-data generalization, if sub-5m Cartosat access is secured;
B) compute-constrained SR, with measured quality/latency/VRAM trade-offs against OpenSR;
C) both, if time/data allow.

A and B are hypotheses until experiments are completed. Neither should be presented as an established advantage beforehand.
