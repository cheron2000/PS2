# AegisSRM — SIH26142 Defense Script

## Delivery plan

Target duration: approximately 4–5 minutes. Spend 35–45 seconds on the technical slides, 30–40 seconds on the proof plan, and 25–30 seconds on the final evidence slide. The central defense message is: **we are not claiming that super-resolution invents truth; we are producing a sharper, geospatially consistent product with explicit confidence and measurable utility.**

## Slide 1 — Opening | 20 seconds

Good morning. Our problem is satellite imagery that is useful but not detailed enough for many decisions. We propose an uncertainty-aware super-resolution framework for Sentinel-2 imagery, targeting a sub-4-meter product while preserving spectral and geospatial consistency. The key distinction is that we do not present sharper pixels as guaranteed truth. We also expose where the model is uncertain.

## Slide 2 — Idea title: problem to solution | 40 seconds

The challenge is not simply blur. It is trust in reconstructed detail. The left image slot represents the 10-meter Sentinel-2 input, and the right slot represents the enhanced product we will validate.

The model combines CNN attention for local texture with transformer fusion for broader spatial context. A separate uncertainty head predicts confidence alongside the reconstruction. That gives the user two outputs: a sharper map and a signal showing where the detail may be inferred rather than directly observed.

The practical promise is better boundaries and more usable imagery without pretending that every reconstructed feature is equally reliable.

## Slide 3 — Technical approach | 45 seconds

The pipeline begins with Sentinel-2 L2A ingestion, cloud and invalid-pixel masking, normalization, and band resampling. Before training, paired images go through quality control and co-registration. We also use geographically separated train, validation, and test regions to reduce spatial leakage.

The reconstruction path uses a CNN feature stem and local high-frequency attention, followed by transformer fusion for global context. The sub-pixel reconstruction produces the enhanced image, while the uncertainty branch predicts mean and variance.

Finally, the product is exported with CRS, affine transform, and band metadata preserved. This matters because an attractive image that cannot be trusted spatially is not useful for downstream analysis.

## Slide 4 — Feasibility and proof plan | 45 seconds

Our proof plan has three parts. First, training uses scalable synthetic SEN2NAIP and SEN2NAIPv2 pairs, while evaluation uses geographically separated areas to avoid spatial leakage.

Second, we test against real references. SEN2NAIP provides a 4× cross-sensor route, SEN2VENµS provides a lower-scale registered check, and Cartosat or other Indian reference scenes are used when access is secured.

Third, we compare more than visual sharpness. We report PSNR, SSIM, spectral-angle error, utility metrics such as IoU or accuracy, and efficiency metrics including latency, VRAM, and tiles per minute. The India-domain test trains on non-Indian pairs and evaluates on held-out Indian AOIs using the same preprocessing. We report spectral error, boundary fidelity, and uncertainty calibration.

## Slide 5 — Impact and benefits | 30 seconds

One capability supports four decision contexts. Agriculture can use sharper crop and field boundaries. Urban teams can use improved built-up detail. Disaster-response teams can benefit from clearer situational awareness. Researchers receive reproducible confidence scores rather than only a visual output.

The common layer is analytical utility. We will not claim impact from image quality alone. We will check whether the enhanced product actually helps a downstream task compared with bicubic upsampling and, where available, real high-resolution references.

## Slide 6 — Research, baselines, and differentiation | 45 seconds

Our evidence stack is deliberately honest. SEN2NAIP and SEN2NAIPv2 support scalable training and real 4× testing. SEN2VENµS and Indian scenes support validation. SEN2SR, OpenSR, and Donike et al. represent important prior art, so uncertainty and spectral consistency are not presented as standalone novelty.

Our differentiation must be measured. The first path is Indian-domain generalization: how well does a model trained on public pairs transfer to Indian AOIs? The second is compute efficiency: parameters, VRAM, latency, and throughput under a fixed patch or tile protocol. We compare our model with practical baselines such as SEN2SRLite and OpenSR. DiffFuSR is treated as a paper reference because full-tile reproduction is not realistic for hackathon compute.

## Closing — 20 seconds

Our proposal is therefore a measurable engineering system, not only a model architecture. We produce enhanced imagery, preserve its geospatial meaning, expose uncertainty, test real Indian transfer, and report the quality–utility–compute trade-off. That is how we intend to make super-resolution useful for decisions rather than merely impressive in a screenshot.

## Likely judge questions and answers

### How is this different from OpenSR or SEN2SR?

We do not claim that the architecture alone is novel. We will benchmark against those systems and focus our contribution on Indian-domain transfer and measured compute efficiency. The comparison will use the same preprocessing, patch size, hardware, and quality metrics.

### Can the model hallucinate details?

Yes. Super-resolution is an inverse problem. That is why the system includes a calibrated uncertainty product, reports hallucination and omission where possible, and compares outputs against real references and downstream-task performance.

### Why use Cartosat if access is uncertain?

Cartosat is an optional validation route, not the only feasibility dependency. The core pipeline can begin with public SEN2NAIP, SEN2NAIPv2, and SEN2VENµS data. If Indian scenes are secured, they become the strongest domain-transfer test.

### What does compute efficiency mean here?

We will fix the patch or tile size and hardware, then measure parameter count, VRAM, latency, and throughput. This prevents an unfair comparison where one model processes smaller patches or uses a different runtime setup.

### How will you prove analytical utility?

We will compare SR, bicubic, and available real-HR references on at least one aligned downstream task, such as land-cover or crop-boundary analysis. We will report IoU or per-class accuracy alongside image-quality metrics.

### What is the minimum viable implementation?

The minimum system is the CNN–transformer reconstruction path, uncertainty head, geospatial export, and a controlled benchmark against bicubic and a practical Sentinel-2 baseline. Diffusion remains optional and is not required for the core claim.
