# AstraSR — Round 2 Research Notes

## Focus
Resolve the sub-4m validation gap identified in Round 1.

## Findings

### 1. SEN2NAIP directly supports a 4x / 2.5m Sentinel-2 SR target
The 2024 Scientific Data paper describes 2,851 real cross-sensor Sentinel-2/NAIP pairs. Sentinel-2 inputs are 10m RGBNIR and the NAIP reference is represented at 2.5m, explicitly establishing a 4x SR setup. The authors also created a large synthetic component using a learned degradation pipeline. Source: https://www.nature.com/articles/s41597-024-04214-y

This resolves the earlier claim that only 5m references were available. SEN2VENµS remains valuable for same-day registered 5m validation, but SEN2NAIP provides the direct 2.5m / 4x evaluation route.

### 2. SEN2NAIP's real pairs have explicit quality filtering
The dataset paper reports filtering based on spatial and spectral alignment, using LightGlue/DISK-derived spatial quality and spectral-angle quality, followed by visual inspection. This supports using the real cross-sensor subset as a held-out external test set rather than treating all synthetic pairs as ground truth.

### 3. SEN2NAIPv2 expands the available training material
The current dataset release reports 62,242 LR/HR pairs, with synthetic RGBN NAIP at 2.5m and 10m and an x4 setup. Its degradation pipeline includes Gaussian blur/downsampling, reflectance harmonization, and noise. Source: https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2

### 4. DiffFuSR validates the same target scale
DiffFuSR targets all 12 Sentinel-2 L2A bands at 2.5m GSD and evaluates on the OpenSR benchmark. Its use of harmonized NAIP/WorldStrat RGB imagery plus a multispectral fusion stage makes it an especially relevant baseline for our target system. Source: https://arxiv.org/abs/2506.11764

## Revised validation plan
- Training: use SEN2NAIP synthetic pairs and, where appropriate, SEN2VENµS.
- Development: validation split by geographic region, not random overlapping patches.
- External test: hold out the real SEN2NAIP cross-sensor 2.5m pairs geographically.
- Benchmark comparison: reproduce/compare against DiffFuSR where code/models and licensing permit.
- Secondary sanity check: evaluate 10/20m -> 5m on SEN2VENµS to test performance at a less aggressive scale.
- Report both conventional metrics and OpenSR-style spectral, spatial, hallucination, omission and improvement metrics.

## Remaining caveat
SEN2NAIP is US-focused and cross-sensor rather than same-sensor. Therefore 2.5m validation is credible for technical benchmarking, but geographic transfer to India should be presented as an external generalization question, not assumed.
