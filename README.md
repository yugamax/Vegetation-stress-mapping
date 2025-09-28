# Sentinel-2 Farm Vegetation Stress Detection API

🌾 **Smart Agricultural Monitoring using Satellite Imagery**

This FastAPI application analyzes Sentinel-2 satellite imagery to detect vegetation stress in agricultural areas. It uses advanced machine learning techniques to distinguish between farmland and other land types, providing targeted crop health analysis.
- green (healthy)
- yellow (moderate stress)
- red (severe stress)
- black (non-vegetation)

The overlay is blended over an RGB preview created from three bands of the input.

## Endpoints
- `POST /classify` - multipart/form-data
  - `file`: the multispectral .tif/.tiff file
  - `red_band` (int, form, default 1): 1-based index of red band
  - `nir_band` (int, form, default 4): 1-based index of NIR band
  - `rgb_b1`, `rgb_b2`, `rgb_b3`: bands to use for R,G,B preview (defaults 1,2,3)
  - `grid_width`, `grid_height` (int): grid cell size in pixels (defaults 64x64)
- `GET /health` - service health

## How it works (simple)
1. Reads specified bands from TIFF using `rasterio`.
2. Computes NDVI = (NIR - RED) / (NIR + RED).
3. Splits the image into rectangular grid cells, computes mean NDVI per cell.
4. Classifies each cell by NDVI thresholds and draws a semi-transparent colored rectangle.
5. Returns a PNG stream.

## Thresholds & tuning
Current thresholds (simple defaults):
- NDVI > 0.6 => green
- 0.3 < NDVI <= 0.6 => yellow
- -0.1 <= NDVI <= 0.3 => red
- NDVI < -0.1 => black (likely non-vegetation)

Tweak thresholds in `main.py` -> `classify_ndvi_cell`.

## Run locally (example)
1. Create a virtualenv and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run:
   ```
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
3. Example curl:
   ```
   curl -X POST "http://localhost:8000/classify" -F "file=@/path/to/multiband.tif" -F "red_band=1" -F "nir_band=4" -o out.png
   ```

## Notes
- This project contains a straightforward, explainable baseline (NDVI + grid). For production or better accuracy:
  - Add georeferencing-preserving outputs (GeoTIFF) and preserve CRS/transform.
  - Use more sophisticated indices, per-pixel classification, or trained models (UNet/CNN).
  - Add sensor fusion, temporal analysis, and model-based risk predictions as described in your hackathon problem statement.
