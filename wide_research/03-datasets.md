# Datasets, Sensors, and Benchmark Validity Audit

## Executive conclusion

The repository contains useful **prototype data-quality primitives**, but it does not yet implement a scientifically valid end-to-end benchmark for the claims in the problem statement. The strongest implemented pieces are integer-shift alignment/QC for the NumPy pair format, optional CRS/grid-aware Cartosat pairing, categorical WorldCover reprojection, and a scene-manifest validator. The central weakness is that these pieces are not connected to verified real products, immutable scene provenance, or the training/evaluation command. In particular, the primary loader does not read the official SEN2NAIP release format, it applies Sentinel-2 reflectance normalization to NAIP values, and the default training split is by sample index rather than scene or geographic area.

External documentation supports a **conditional** 4× Sentinel-2-to-2.5 m US cross-sensor experiment through SEN2NAIP, and a 2× Sentinel-2-to-5 m same-day validation route through SEN2VENµS. It does not support a blanket claim that the model reconstructs true 2.5 m detail everywhere, that it generalizes to India, or that WorldCover provides independent high-resolution ground truth. Cartosat-3 could supply an India-relevant reference, but it is a different sensor with different spectral bands, access constraints, and no demonstrated pairing in this repository. Until these issues are resolved, the defensible description is **evaluation infrastructure and a benchmark-ingestion prototype**, not a validated super-resolution system.

## Scope and evidence convention

This audit separates three kinds of statements:

- **Implemented fact** means behavior visible in repository code. File paths and line numbers are provided so it can be checked directly.
- **Repository claim or plan** means a statement in project documentation or a code comment. It is not treated as experimental evidence.
- **External evidence** means an official product page, dataset record, or peer-reviewed/reputable technical source. External facts are cited in the References section.

The repository was not modified except for creating this report. No claim below should be read as evidence that a real-data training run has completed.

## 1. What the benchmark is supposed to establish

The problem statement asks for fine-scale reconstruction that is useful for interpretation and analysis, with uncertainty management and validation against high-resolution references (`problem-statement.md:18-23`). The solution draft proposes a primary 4× Sentinel-2-to-NAIP route, a VENµS validation route, and a Cartosat route for Indian data, together with downstream use of WorldCover (`solution-draft.md:25-60`). Those are reasonable hypotheses, but they are different estimands:

1. **SEN2NAIP** measures cross-sensor reconstruction over the contiguous United States, with Sentinel-2 at 10 m and NAIP-derived 2.5 m imagery. It tests agreement with an aerial sensor, not recovery of a universal 2.5 m physical truth.
2. **SEN2VENµS** measures same-day satellite cross-sensor agreement at approximately 5 m. It is better for a controlled 2× validation target, but its sites and viewing geometries are limited.
3. **Cartosat-3** could test Indian geographic transfer and very high spatial detail. It is not interchangeable with NAIP or VENµS because its bands, point-spread function, viewing geometry, radiometry, and access conditions differ.
4. **WorldCover** is an 11-class, approximately 10 m land-cover map derived from Sentinel-1 and Sentinel-2. It is a downstream label product, not an independent 2.5 m reference image, and its labels carry non-trivial classification error.

A single aggregate PSNR/SSIM or downstream score across these products cannot establish all four claims. Each route needs its own acquisition metadata, registration tolerance, temporal rule, sensor harmonization policy, geographic split, and uncertainty interval.

## 2. Repository implementation audit

### 2.1 SEN2NAIP loader: format and scale are not production-ready

The primary loader explicitly targets a local convention of `lr/<id>.npy` and `hr/<id>.npy`, and explicitly states that this is **not** the official release format (`src/datasets/sen2naip.py:8-20`). The module says that the official Hugging Face SEN2NAIPv2 release is a TACO-format columnar archive and that no real sample was tested (`src/datasets/sen2naip.py:14-20`, `src/datasets/sen2naip.py:31-38`). Therefore, the repository currently has no demonstrated ingestion path from the named public benchmark to the actual training loader.

The intended task is 10 m Sentinel-2 RGBNIR to 2.5 m NAIP, represented by `SCALE_FACTOR = 4` (`src/datasets/sen2naip.py:60`). The loader checks array shape, finite values, channel count, and an approximate band schema, then estimates an integer shift by normalized cross-correlation (`src/datasets/sen2naip.py:181-230`). The alignment search is useful as a rejection/QC mechanism, but it uses only the first band and an integer shift (`src/datasets/sen2naip.py:81-117`). The crop converts the shift to low-resolution pixels with integer floor division (`src/datasets/sen2naip.py:120-143`). This cannot represent subpixel phase, affine distortion, rotation, relief displacement, or local residual misregistration.

A more serious implemented mismatch is radiometric normalization. `__getitem__` applies `normalize_bands(..., method="reflectance")` independently to both LR and HR (`src/datasets/sen2naip.py:235-251`). The reflectance method divides every input by 10,000 (`src/preprocessing.py:58-76`). Sentinel-2 L2A digital numbers conventionally use a 10,000 scale, but NAIP is 8-bit/broadband aerial imagery in the public product description and is declared in the repository schema with a 1/255 scale (`src/datasets/band_schema.py:95-105`). The loader ignores that schema scale and divides raw NAIP values by 10,000. Unless an upstream converter has already transformed NAIP into 0–10,000 reflectance-like values, the HR target is compressed by about a factor of 39.2 relative to the intended 0–1 mapping. This can invalidate loss values, spectral metrics, and any claim of radiometric fidelity.

The band validator is useful but limited. It checks dimensions, constant bands, and near-duplicate channels, while explicitly admitting that array values cannot verify wavelength identity (`src/datasets/band_schema.py:149-159`). The loader therefore cannot detect swapped RGBNIR order, incorrect scale/offset metadata, or an array that has been preprocessed inconsistently unless those facts are represented in an external manifest.

### 2.2 SEN2VENµS loader: correct nominal factor, weak provenance and validation

The VENµS loader sets a nominal 2× factor for 10 m Sentinel-2 to 5 m VENµS (`src/datasets/sen2venus.py:42-53`). It requires matching `.npy` filenames, checks shape, estimates the same integer first-band NCC shift, and normalizes both arrays with the same generic function (`src/datasets/sen2venus.py:67-133`). Unlike the SEN2NAIP loader, it does not validate finite values, declared band identity, sensor units, cloud masks, or product metadata before pairing.

The code does not read the published SEN2VENµS tensor/index format. It assumes one 3-D NumPy array per side and derives pair identity from a filename (`src/datasets/sen2venus.py:67-79`). It also does not enforce the published same-day acquisition relationship, site IDs, viewing angle, or the recommended site/pair split. Consequently, a high NCC score only means that the first arrays correlate after a small integer translation; it does not establish same-date registration or independent reference validity.

### 2.3 Cartosat pairing: useful geospatial hooks, but dangerous channel fallback

The Cartosat adapter adds optional CRS, affine, pixel-size, and grid-phase contracts through `GridSpec` (`src/datasets/cartosat_pairing.py:151-205`). It also propagates nodata/valid masks, returns alignment statistics, and records a nominal `reference_gsd_m` (`src/datasets/cartosat_pairing.py:207-235`). These are good engineering foundations.

However, when metadata does not include both `lr_grid` and `hr_grid`, the code falls back to shape-only pairing and integer NCC alignment (`src/datasets/cartosat_pairing.py:180-205`). The adapter does not require acquisition date, CRS, affine transform, source product ID, or a valid band map. More importantly, `_match_channels` explicitly repeats a one-band HR array or truncates extra HR bands to match the LR count (`src/datasets/cartosat_pairing.py:71-82`), and `prepare_pair` invokes it during affine harmonization (`src/datasets/cartosat_pairing.py:123-136`, `src/datasets/cartosat_pairing.py:217-219`). That behavior can turn a panchromatic product into repeated pseudo-spectral channels or silently discard spectral channels. It is not valid evidence of multispectral reconstruction. Cartosat ingestion should reject unsupported channel counts unless an explicit, recorded band-response mapping is supplied.

The adapter's affine mean/std matching is candidly documented as a prototype rather than a calibrated sensor-response model (`src/datasets/cartosat_pairing.py:85-97`). That is the correct caveat. It cannot correct different spectral response functions, atmospheric states, bidirectional reflectance effects, saturation, or scene-dependent nonlinear radiometry.

### 2.4 WorldCover loader: technically sound reprojection, scientifically limited label target

The WorldCover loader maps raw ESA class codes 10–100 to consecutive labels 0–10 and maps unrecognized values, including nodata, to -1 (`src/datasets/worldcover.py:39-71`). It uses nearest-neighbor reprojection to an exact reference grid, correctly avoiding interpolation of categorical classes (`src/datasets/worldcover.py:86-117`). This is one of the strongest implemented components.

The limitation is not primarily code correctness; it is benchmark interpretation. The loader treats WorldCover v200's 11-class schema as a compatible truth target, but does not encode product version, epoch, input-quality masks, validation stratum, or an acquisition-time relationship to the imagery (`src/datasets/worldcover.py:39-83`). The repository's scene evaluator can enforce an integer 11-class label schema (`src/scene_protocol.py:145-158`), but schema validity is not label accuracy or temporal independence.

The downstream harness also contains a separate three-class NDVI threshold classifier (water, non-vegetated, vegetation) (`src/eval_downstream.py:65-87`) and compares SR, bicubic, and optional HR images by IoU (`src/eval_downstream.py:90-135`). This is runnable, but it is not a WorldCover 11-class evaluation unless an explicit ontology conversion is defined. A 3-class threshold map should not be reported as an 11-class WorldCover score, and WorldCover should not be treated as ground truth for every pixel.

### 2.5 Scene split protocol exists but is not wired into training

`build_scene_splits` creates deterministic scene-disjoint train/validation/test assignments and `validate_scene_split_manifest` checks scene and sample overlap (`src/scene_protocol.py:24-90`). The evaluator computes per-scene means and bootstrap intervals (`src/scene_protocol.py:112-142`, `src/scene_protocol.py:161-206`), and the uncertainty helper reports Gaussian NLL and interval coverage (`src/scene_protocol.py:209-242`). These are valuable capabilities.

They do not yet establish a leakage-safe experiment. The training CLI calls `SEN2NAIPDataset` and then uses `random_split` over sample indices (`src/train.py:191-220`, `src/train.py:242-253`). The training docstring explicitly says that this is not a persisted scene/AOI-aware split and does not record file-level provenance (`src/train.py:199-208`). The CLI does not accept or load a scene manifest. If patches from one scene, tile, or neighboring ROI appear in both subsets, the validation score can measure spatial memorization rather than geographic generalization. This risk is especially high for overlapping or adjacent patches.

## 3. External sensor and dataset evidence

### 3.1 Sentinel-2: multispectral, mixed native resolutions, and L2A masks

Copernicus documentation states that Sentinel-2 MSI has 13 bands: four at 10 m, six at 20 m, and three at 60 m. The L2A product is atmospherically corrected surface reflectance and includes AOT, water-vapor, and Scene Classification Layer products at the respective 10/20/60 m grids [1] [2]. The repository's four-band choice, B2/B3/B4/B8, is therefore a deliberate 10 m subset, not the full Sentinel-2 sensor.

This matters for claims. A model trained only on B2/B3/B4/B8 cannot claim generic 13-band Sentinel-2 super-resolution. The 20 m bands are not merely lower-resolution copies of the 10 m bands; they have different spectral response functions and must be resampled with documented rules if included. Cloud, shadow, saturated/defective, snow, and no-data masks must be propagated into loss and metrics. The current SEN2NAIP loader rejects non-finite values but does not ingest SCL or a validity mask, while `normalize_bands` itself says that nodata/cloud pixels should be handled by a validity mask before normalization (`src/preprocessing.py:58-76`).

Sentinel-2 data are free and broadly accessible, which supports reproducibility. That does not solve temporal matching, processing-baseline differences, or cloud screening. Product processing baseline, L1C/L2A version, tile ID, acquisition timestamp, and SCL mask must be persisted per sample.

### 3.2 NAIP: strong US reference availability, but not a universal HR truth

USGS describes NAIP as orthorectified aerial imagery acquired during the US agricultural growing season. It was generally 1 m GSD from 2003–2017, changed to 0.6 m in 2018 with possible 0.3 m coastal data, and moved to a refresh cycle of no longer than three years. Tiles may be natural color or four-band RGBNIR and may contain up to 10% cloud cover. USGS distributes GeoTIFF and JPEG2000 products; the JPEG2000 product uses 10:1 lossy compression [3].

The SEN2NAIP paper reports a carefully filtered cross-sensor subset: 2,851 pairs; S2 10 m RGBNIR and NAIP downloaded at 2.5 m for a 4× task; candidate ROIs at least 5 km apart; S2–NAIP acquisition difference no more than one day; cloud filtering; feature-based spatial quality threshold of one pixel; spectral-angle threshold of two degrees; and visual inspection [4]. Those controls make the published dataset materially stronger than a directory of arbitrary paired arrays.

The controls do not make NAIP a physical 2.5 m Sentinel-2 truth. The sensors have different optics, spectral response functions, illumination/viewing geometries, and atmospheric paths. The paper uses histogram matching and a learned degradation model, which means the synthetic subset is partly model-generated rather than simultaneously observed at both resolutions [4]. NAIP is also US-only. A result on this route supports performance on the specified US cross-sensor distribution; it does not establish Indian or global geographic generalization.

The repository's 4× shape check is consistent with the dataset target, but the current loader does not reproduce the paper's metadata filters or feature-based registration. It performs only an integer first-band NCC search. It also does not verify whether a 2.5 m array is native NAIP, a Google Earth Engine pyramid level, or an already resampled derivative.

### 3.3 VENµS and SEN2VENµS: credible same-day 5 m validation with restricted scope

The SEN2VENµS record describes 10 m and 20 m cloud-free Sentinel-2 surface-reflectance patches paired with spatially registered 5 m VENµS surface reflectance acquired on the same day. It covers 29 locations and 132,955 patches, supporting eight Sentinel-2 bands at 5 m. The record provides site IDs, product IDs, dates, viewing zenith angles, patch footprints, and per-pair files [5]. It also recommends keeping pairs or sites separate for testing, not merely randomizing individual patches [5].

The same record states materially different licenses: Sentinel-2-derived files use the relevant Theia/Copernicus terms, VENµS patches are CC BY-NC 4.0, and remaining files are CC BY 4.0 [5]. This is compatible with research use but may restrict commercial redistribution or model/data packaging. Licensing must be recorded in the project artifact, not assumed from the word “open.”

The VENµS mission paper describes a 12-band visible/NIR radiometer with approximately 5.3 m nadir resolution and documents geometric processing and registration performance [6]. The dataset is therefore a much better validation target for a 5 m claim than an arbitrary 5 m upsampled image. Still, it is not a global test set: sites, land-cover composition, viewing angles, cloud-free selection, and acquisition dates are finite. A model can overfit site or landscape cues unless the held-out set is site-disjoint.

### 3.4 Cartosat-3: India-relevant, spectrally non-equivalent, and access-sensitive

Cartosat-3 carries a panchromatic sensor reported at up to 0.25 m and a four-band visible/NIR multispectral camera at approximately 1 m, with 16 km swath and agile multi-angle acquisition [7]. The ISRO mission page confirms the high-resolution imaging and mapping purpose, but does not by itself provide all band specifications [8]. The technical mission summary should therefore be treated as a source for planning, while the ordered product metadata must be the authority for the exact scene.

Cartosat-3's MX bands are broad blue/green/red/NIR bands and are not the same spectral response functions as Sentinel-2 B2/B3/B4/B8. PAN is one broad band and must never be repeated into four bands as a proxy for multispectral truth. The repository's current repeat/truncate fallback (`src/datasets/cartosat_pairing.py:71-82`) is incompatible with a defensible spectral benchmark unless disabled and replaced by explicit response-function mapping.

The project's own research notes report that access to finer-than-5 m Indian EO data involves Bhoonidhi/NRSC terms, registration, and potentially priced products (`external_gap_audit.md:85-90`). Those notes should be verified against the order-specific current license and product metadata. Even if access is obtained, one or a few Cartosat scenes cannot support a broad India-generalization claim. They can support a clearly scoped held-out Indian-site test, provided acquisition date, off-nadir angle, orthorectification level, GSD, band metadata, and license are published.

### 3.5 WorldCover: useful downstream labels, not independent HR truth

ESA WorldCover offers 2020 v100 and 2021 v200 global land-cover maps at approximately 10 m with 11 classes. The official page warns that differences between 2020 and 2021 reflect both real land-cover change and different algorithm versions. The products are free of charge without restriction of use under CC BY 4.0, with required attribution. The page reports independent overall accuracy of 74.4% for 2020 and 76.7% for 2021 [9].

WorldCover is generated using Sentinel-1 and Sentinel-2 information, so using it to evaluate a Sentinel-2 model can introduce source overlap and shortcut risk. If the SR input and WorldCover label share Sentinel-2 observations or a close temporal window, the downstream test is not independent in the sense needed to establish operational utility. The label also has 10 m mapping resolution and classification uncertainty; it cannot reveal whether 2.5 m texture is physically correct.

WorldCover can still support a useful, explicitly limited test: fix one product version and epoch; use its input-quality layer and nodata; define an ontology; evaluate only valid labels; report per-scene and per-class metrics; and compare SR against bicubic using exactly the same classifier and spatial support. The current code handles the class-code remap and nearest-neighbor alignment, but the benchmark documentation must add version, epoch, quality-mask, and temporal-independence fields.

## 4. Validity audit by risk category

### Spectral resolution and response mismatch

**Implemented strengths:** the repository declares the intended Sentinel-2, NAIP, Cartosat MX, and SEN2VENµS channel names and approximate centers (`src/datasets/band_schema.py:83-132`), and it rejects obvious repeated/constant bands in the SEN2NAIP path (`src/datasets/band_schema.py:149-195`).

**Gaps:** the declarations are not verified against file metadata; the SEN2NAIP loader applies one normalization method to sensors with different units; the VENµS loader skips schema validation; and Cartosat can repeat or truncate channels. No code reads spectral response functions, scale/offset tags, saturation flags, or source band names. The 4-band RGBNIR subset must be described as a subset in every claim.

**Validity consequence:** spectral-angle, SAM, PSNR, and downstream NDVI comparisons can be dominated by unit conversion and spectral-response mismatch rather than SR quality.

### Spatial resolution and true detail

**Implemented strengths:** shape checks enforce nominal integer scale factors, and Cartosat can preserve grid metadata when supplied.

**Gaps:** shape is not proof of GSD. A 484×484 array can be 2.5 m native, resampled from 1 m, or generated by an image pyramid. The code does not verify affine transforms, CRS, pixel size, sensor MTF/PSF, or whether the HR product is an orthorectified derivative. The SEN2NAIP loader's integer translation search is weaker than the published feature-based QA thresholds.

**Validity consequence:** a nominal 4× score may reward matching a resampling artifact or misregistration pattern. Claims should say “agreement with NAIP-derived 2.5 m reference” rather than “recovered 2.5 m ground truth.”

### Co-registration and geometry

**Implemented strengths:** NCC-based rejection, shift/crop, optional CRS/grid contract, and mask propagation exist. Cartosat records shift and score.

**Gaps:** alignment is integer-only, first-band-only, and global. It does not handle subpixel translation, local warps, rotations, terrain/parallax, or sensor-specific geolocation errors. The default SEN2NAIP and SEN2VENµS loaders do not require CRS/grid metadata or source footprints. An NCC threshold of 0.1 is a permissive heuristic, not a validated registration tolerance.

**Validity consequence:** pixelwise losses and full-resolution fidelity metrics may measure registration error. Registration quality must be reported separately from native-grid and registered scores, with a prespecified tolerance and an exclusion log.

### Temporal mismatch

**Implemented gap:** no loader requires acquisition date, time difference, season, cloud mask, or phenological state. Pair IDs are filenames.

**External benchmark evidence:** published SEN2NAIP filters acquisition difference to one day, while SEN2VENµS provides same-day pairs [4] [5]. NAIP is acquired during growing seasons and refreshed approximately every three years [3].

**Validity consequence:** arbitrary S2/NAIP or S2/Cartosat dates can introduce crop growth, harvest, construction, flooding, shadow, or snow changes that no deterministic registration can fix. The dataset contract must reject missing dates and specify the maximum allowed gap by land-cover/use case.

### Licensing and access

Sentinel-2 is free and open under Copernicus access terms [1] [2]. NAIP is distributed through USGS/USDA channels and is described by the SEN2NAIP paper as public-domain imagery suitable for dataset construction [3] [4], but exact downloaded product terms and derived-data attribution should be retained. SEN2VENµS includes VENµS CC BY-NC 4.0 content and thus should not be treated as unrestricted commercial data [5]. Cartosat access is order- and product-dependent through Indian EO channels; the current license and eligibility should be attached to each scene [7] [8]. WorldCover is CC BY 4.0 with attribution [9].

The repository has no committed dataset license manifest, source URL, product version, checksum, or redistribution decision. `ProvenanceManifest` exists as a data structure (`src/datasets/band_schema.py:198-240`), but no shown loader path requires or persists one for the named real datasets.

### Train/test leakage and geographic generalization

The scene protocol itself is appropriate, but the active training path uses random index splitting (`src/train.py:191-220`, `src/train.py:250-260`). It does not group by published SEN2VENµS site, SEN2NAIP ROI, NAIP flight tile, Sentinel-2 MGRS tile, or Cartosat AOI. Patch overlap and near-neighbor autocorrelation can therefore leak scene appearance into validation. Remote-sensing methodology research shows that random splits are misleading under spatial autocorrelation and that block size should reflect the application and correlation range; even spatial blocking is not sufficient when sampling bias or unrepresented regions remain [10].

For claims of geographic generalization, the minimum unit of separation should be a scene/AOI or site, not a patch. A US NAIP test and an Indian Cartosat test should be reported as separate domain tests. A random mixture of US, European, and Indian patches is not a geographic generalization experiment.

### Benchmark support for the stated claims

The current benchmark can support only narrower statements after real data and protocol wiring are completed:

- It can test **paired-array reconstruction quality** under a declared sensor route.
- It can test **same-protocol improvement over bicubic** if the reference, masks, and registration policy are fixed.
- It can test a limited downstream **land-cover proxy** if the ontology and label version are explicit.

It cannot yet support:

- universal 2.5 m recovery claims;
- 13-band Sentinel-2 claims when only four 10 m bands are used;
- India or global geographic generalization from US-only SEN2NAIP data;
- calibrated uncertainty claims from the current uncalibrated training path;
- independent downstream utility claims using WorldCover without accounting for its Sentinel-derived labels and 74.4–76.7% map accuracy;
- Cartosat multispectral claims while repeat/truncate channel fallback remains possible.

## 5. Prioritized remediation plan

### P0 — prevent invalid numbers now

1. **Fix unit conversion by product.** Read scale/offset metadata and normalize Sentinel-2, NAIP, VENµS, and Cartosat separately. For NAIP, make 8-bit DN-to-[0,1] conversion explicit or document a calibrated reflectance conversion. Add a test that fails if raw 0–255 NAIP data are divided by 10,000.
2. **Disable Cartosat channel repeat/truncate by default.** Require an explicit band map and sensor-response metadata. Reject PAN-only data for a four-band spectral benchmark.
3. **Require real metadata.** Every pair must carry source IDs, acquisition timestamps, CRS, affine transform, pixel size, band names, units, product version, cloud/QA mask, and license/attribution. Hash the source and processed arrays.
4. **Add masks to training and metrics.** SCL, nodata, saturation, cloud, shadow, and WorldCover input-quality masks must flow through normalization, loss, cropping, and evaluation.

### P1 — make the split scientifically meaningful

1. Convert official SEN2NAIP and SEN2VENµS records into a canonical scene table. Use `scene_id`, `sample_id`, `site_id`, `MGRS tile/AOI`, dates, and footprint geometry.
2. Make the training CLI consume a persisted split manifest. Reject random index splitting for benchmark mode.
3. Split by site/AOI/ROI before patch extraction. For SEN2VENµS, hold out complete sites or complete acquisition pairs. For SEN2NAIP, preserve the published spatial separation and prevent same-ROI patches from crossing splits.
4. Add a geographically held-out Cartosat test only after at least several independent Indian AOIs are available. Report it as domain transfer, not as pooled test performance.

### P2 — establish sensor-specific benchmark contracts

For each route, publish a one-page contract containing native GSD, bands and wavelength ranges, source units, preprocessing baseline, acquisition-date rule, registration method and tolerance, mask policy, reference status, license, and intended claim. Report native-grid metrics and registered metrics separately. Include per-scene metrics, bootstrap confidence intervals over scenes, dropped-pair counts/reasons, and repeated seeds.

Use SEN2NAIP for the 4× US cross-sensor experiment only after reproducing the official TACO release and its quality filters. Use SEN2VENµS for a 2× same-day satellite validation, preserving site/pair separation and its CC BY-NC restriction. Use Cartosat MX only for an explicitly harmonized 1 m multispectral Indian route; do not use repeated PAN channels as multispectral truth.

### P3 — make downstream evaluation honest

Use WorldCover v100 or v200 as a **noisy, 10 m downstream label product**, not HR truth. Freeze one epoch/version, exclude invalid quality pixels, define a mapping from the 3-class NDVI proxy to the selected label ontology or use an independently trained classifier, and report class-balanced per-scene metrics. Explicitly disclose that WorldCover uses Sentinel data and may not be temporally or sensor-independent from the input.

For utility claims, add at least one label source independent of the Sentinel-2 input where feasible, or state that the result is a conditional WorldCover-consistency test. Compare SR and bicubic with the same classifier, same valid mask, same output grid, and paired scene-level uncertainty intervals.

## 6. Bottom line for the problem statement

The repository is directionally aligned with the problem: it recognizes cross-sensor misregistration, label-schema errors, geospatial grids, uncertainty, and leakage. The most credible near-term claim is:

> “We implement a prototype pipeline for quality-controlled, cross-sensor Sentinel-2 super-resolution evaluation, with planned validation on SEN2NAIP, SEN2VENµS, and Indian Cartosat data.”

The repository cannot yet honestly claim:

> “Our method reconstructs reliable 2.5 m detail, improves analytical utility, generalizes geographically, and produces calibrated uncertainty.”

That stronger statement requires real-product ingestion, product-specific units, metadata and masks, scene-disjoint splits wired into training, published registration/temporal filters, valid Cartosat spectral mapping, and executed baseline results with confidence intervals. The gap is primarily **benchmark validity and evidence**, not the absence of additional model code.

## References

[1]: https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2 "Copernicus Data Space Sentinel-2 mission and data access"

[2]: https://documentation.dataspace.copernicus.eu/Data/Sentinel2.html "Copernicus Data Space Sentinel-2 product documentation"

[3]: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip "USGS EROS Archive: National Agriculture Imagery Program"

[4]: https://www.nature.com/articles/s41597-024-04214-y "SEN2NAIP: A dataset for Sentinel-2 super-resolution"

[5]: https://zenodo.org/records/6514159 "SEN2VENµS, a dataset for the training of Sentinel-2 super-resolution algorithms"

[6]: https://www.mdpi.com/2072-4292/14/14/3281 "VENµS: Mission Characteristics, Final Evaluation of the First Phase and Data Production"

[7]: https://www.eoportal.org/satellite-missions/cartosat-3 "Cartosat-3 mission and sensor specifications"

[8]: https://www.isro.gov.in/Cartosat_3.html "ISRO Cartosat-3 mission page"

[9]: https://esa-worldcover.org/en/data-access "ESA WorldCover data access, license, accuracy, and product versions"

[10]: https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1531097/full "Choosing blocks for spatial cross-validation: lessons from a marine remote-sensing application"

## Repository evidence index

- Problem requirements: `problem-statement.md:18-23`.
- Planned datasets and evaluation: `solution-draft.md:25-60`.
- SEN2NAIP local format gap and untested real data: `src/datasets/sen2naip.py:8-38`.
- SEN2NAIP scale, alignment, and loader checks: `src/datasets/sen2naip.py:60-117`, `src/datasets/sen2naip.py:181-251`.
- Product-specific band declarations and limitation of array-only validation: `src/datasets/band_schema.py:83-132`, `src/datasets/band_schema.py:149-159`.
- Generic normalization and missing-mask warning: `src/preprocessing.py:58-76`.
- SEN2VENµS loader assumptions: `src/datasets/sen2venus.py:42-133`.
- Cartosat channel fallback and harmonization: `src/datasets/cartosat_pairing.py:71-136`.
- Cartosat optional grid contract and mask propagation: `src/datasets/cartosat_pairing.py:151-235`.
- WorldCover class remap and nearest-neighbor alignment: `src/datasets/worldcover.py:39-117`.
- Three-class NDVI proxy and SR/bicubic comparison: `src/eval_downstream.py:65-135`.
- Scene split builder and leakage validator: `src/scene_protocol.py:24-90`.
- Per-scene metrics and uncertainty helper: `src/scene_protocol.py:112-242`.
- Active training index split and explicit scope limitation: `src/train.py:191-220`, `src/train.py:242-260`.
- Existing repository research caveats on Cartosat access and cross-sensor limitations: `external_gap_audit.md:85-99`.

**Audit status:** no source code was modified; this report is the only requested artifact.
