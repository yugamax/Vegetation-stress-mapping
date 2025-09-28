from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import math
import ee
import os
import json
import requests # NEW: Added for non-interactive data download
import time     # NEW: Added for robust download retries
from dotenv import load_dotenv
from scipy import ndimage # Used in farmland detection

# Load environment variables from .env file
load_dotenv()

# -------------------------------
# Earth Engine Initialization (Robust & Flexible)
# -------------------------------

# Define the expected service account file name based on your provided context
SERVICE_ACCOUNT_FILE = "vernal-hall-451018-v5-390dbadc6307.json"
credentials = None
service_account_dict = None # Initialize dictionary variable

try:
    # 1. Try to load from the JSON file directly (Most robust for local dev)
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"INFO: Loading Earth Engine credentials directly from file: {SERVICE_ACCOUNT_FILE}")
        
        # Read the file content and parse it
        with open(SERVICE_ACCOUNT_FILE, 'r') as f:
            service_account_dict = json.load(f)
        
        # FIX: Aggressive cleanup of the private key string to remove hidden characters/bad line endings.
        private_key_string = service_account_dict['private_key']
        
        # Replace carriage returns and strip whitespace for a clean PEM string
        private_key_string = private_key_string.replace('\r\n', '\n').replace('\r', '\n').strip()
        private_key_data = private_key_string.encode('utf-8')
        
        credentials = ee.ServiceAccountCredentials(
            service_account_dict['client_email'],
            key_data=private_key_data
        )
        
    # 2. Fallback to loading from the environment variable (Good for deployment)
    elif os.getenv("EE_SERVICE_ACCOUNT_JSON"):
        print("INFO: Local file not found. Falling back to EE_SERVICE_ACCOUNT_JSON environment variable.")
        
        EE_SERVICE_ACCOUNT_JSON = os.getenv("EE_SERVICE_ACCOUNT_JSON")
        service_account_dict = json.loads(EE_SERVICE_ACCOUNT_JSON)
        
        # CRUCIAL FIX: Ensure newlines are un-escaped when reading from ENV
        if 'private_key' in service_account_dict:
             # Un-escape, clean up, then strip whitespace
             private_key_string = service_account_dict['private_key'].replace("\\n", "\n").replace('\r\n', '\n').replace('\r', '\n').strip()

        # FIX: Explicitly encode the cleaned private key string to bytes.
        private_key_data = private_key_string.encode('utf-8')
        
        credentials = ee.ServiceAccountCredentials(
            service_account_dict['client_email'],
            key_data=private_key_data
        )
    
    else:
        raise RuntimeError(f"EE_SERVICE_ACCOUNT_JSON not set, and local file '{SERVICE_ACCOUNT_FILE}' not found.")

    ee.Initialize(credentials)
    print("INFO: Earth Engine initialized successfully.")

except Exception as e:
    # Raise a clear error if initialization fails
    raise RuntimeError(f"Earth Engine initialization failed. Error: {e}")


# -------------------------------
# Sentinel-2 Cloud Masking Function
# -------------------------------
def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA band."""
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return image.updateMask(mask).divide(10000)

# -------------------------------
# GEEMAP REPLACEMENT FUNCTION (EE to NumPy)
# -------------------------------
def download_ee_image_to_numpy(ee_image, ee_region, scale, max_retries=5):
    """
    Replaces geemap.ee_to_numpy for non-interactive cloud deployment.
    Downloads GEE image data as a GeoTIFF blob and reads it into a NumPy array.
    """
    # 1. Define download parameters and get the signed URL
    
    # FIX: Corrected method to get the GeoJSON string directly from the computed ee.Geometry.
    # We call getInfo() on the ee.Geometry object to fetch the GeoJSON dictionary, 
    # then serialize it using json.dumps().
    try:
        region_geojson_string = json.dumps(ee_region.getInfo())
    except Exception as e:
        raise RuntimeError(f"Failed to convert ee.Geometry to GeoJSON string. Error: {e}")

    
    params = {
        'format': 'GEO_TIFF',
        'region': region_geojson_string,
        'scale': scale,
        'crs': 'EPSG:4326',
        'filePerBand': 'False'
    }
    
    try:
        download_url = ee_image.getDownloadUrl(params)
    except Exception as e:
        raise RuntimeError(f"Failed to generate GEE download URL: {e}")

    # 2. Download content (with robust retry logic)
    response = None
    for attempt in range(max_retries):
        try:
            response = requests.get(download_url, timeout=30)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            break
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Download attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {2 ** attempt}s...")
                time.sleep(2 ** attempt) # Exponential backoff
            else:
                raise HTTPException(status_code=500, detail=f"Failed to download image from GEE after {max_retries} attempts. Error: {e}")
        
    if response is None:
        raise HTTPException(status_code=500, detail="GEE download failed before getting a response.")
    
    # 3. Read GeoTIFF into memory using rasterio
    with MemoryFile(response.content) as memfile:
        with memfile.open() as dataset:
            # Read all bands (Bands, Height, Width)
            numpy_data = dataset.read()
            # Get the metadata/profile
            profile = dataset.profile
            
            # Transpose to (Height, Width, Bands) for PIL/NumPy processing
            # Transpose needed only if 3D (multi-band)
            if numpy_data.ndim == 3:
                numpy_data = np.transpose(numpy_data, (1, 2, 0))
            elif numpy_data.ndim == 2:
                # If only one band, reshape it to (H, W, 1) for consistency
                numpy_data = numpy_data[..., np.newaxis]

            return numpy_data, profile


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Sentinel-2 Veg Stress Zone Classifier")

# Middleware for CORS if this API is called from a web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity in this deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Pydantic Models
# -------------------------------
class ClassificationParameters(BaseModel):
    red_band: str = Field(default="B4", description="Red band for Sentinel-2")
    nir_band: str = Field(default="B8", description="NIR band for Sentinel-2")
    rgb_b1: str = Field(default="B4", description="Red band for RGB display")
    rgb_b2: str = Field(default="B3", description="Green band for RGB display")
    rgb_b3: str = Field(default="B2", description="Blue band for RGB display")
    grid_width: int = Field(default=20, gt=0)
    grid_height: int = Field(default=20, gt=0)
    output_format: str = Field(default="png")
    opacity: float = Field(default=0.7, ge=0.0, le=1.0)
    cloud_percentage: float = Field(default=20, ge=0.0, le=100.0, description="Maximum cloud percentage")
    buffer_size: int = Field(default=2000, ge=100, le=10000, description="Buffer size in meters (100m-10km). Smaller = zoomed in, Larger = zoomed out")
    image_scale: int = Field(default=10, ge=5, le=30, description="Image resolution in meters. Smaller = higher quality (5-30m)")
    image_quality: int = Field(default=95, ge=70, le=100, description="PNG/JPEG quality (70-100%)")
    enhance_contrast: bool = Field(default=True, description="Apply contrast enhancement")
    upscale_factor: float = Field(default=1.0, ge=1.0, le=4.0, description="Image upscaling factor (1.0-4.0). Higher = larger output image")
    detect_farmland: bool = Field(default=True, description="Only analyze agricultural areas, ignore forests/urban areas")
    farmland_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Confidence threshold for farmland detection (0.0-1.0)")

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    parameters: ClassificationParameters = Field(default_factory=ClassificationParameters)

# -------------------------------
# Constants
# -------------------------------
CLASS_COLOR_MAP = {
    "green": (0, 255, 0),
    "yellow": (255, 215, 0),
    "red": (255, 0, 0),
    "black": (0, 0, 0),
}

CLASS_VALUE_MAP = {
    "black": 0,
    "red": 1,
    "yellow": 2,
    "green": 3,
}

# -------------------------------
# Agricultural Detection Functions (Retained from original code)
# -------------------------------
def compute_agricultural_indices(red, nir, green, blue, swir1, swir2):
    """Compute multiple vegetation and land use indices for agricultural detection."""
    eps = 1e-6
    
    # NDVI - Normalized Difference Vegetation Index
    ndvi = np.where(np.abs(nir + red) > eps, (nir - red) / (nir + red + eps), np.nan)
    
    # EVI - Enhanced Vegetation Index (better for dense vegetation)
    # Using Sentinel-2 coefficients (2.5, 6, 7.5) which are appropriate for this dataset
    evi = np.where(np.abs(nir + 6*red - 7.5*blue + 1) > eps, 
                    2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps), np.nan)
    
    # SAVI - Soil Adjusted Vegetation Index (reduces soil background effects)
    L = 0.5 # soil brightness correction factor
    savi = np.where(np.abs(nir + red + L) > eps,
                    (nir - red) * (1 + L) / (nir + red + L + eps), np.nan)
    
    # NDWI - Normalized Difference Water Index (detects water bodies)
    ndwi = np.where(np.abs(green + nir) > eps, (green - nir) / (green + nir + eps), np.nan)
    
    # NDBI - Normalized Difference Built-up Index (detects urban areas)
    if swir1 is not None and swir1.shape == nir.shape:
        ndbi = np.where(np.abs(swir1 + nir) > eps, (swir1 - nir) / (swir1 + nir + eps), np.nan)
    else:
        ndbi = None
    
    return ndvi, evi, savi, ndwi, ndbi

def detect_agricultural_areas(red, nir, green, blue, swir1=None, swir2=None, confidence_threshold=0.6):
    """Detect agricultural areas using spectral indices and pattern analysis."""
    
    # Compute indices
    ndvi, evi, savi, ndwi, ndbi = compute_agricultural_indices(red, nir, green, blue, swir1, swir2)
    
    # Initialize farmland probability map
    farmland_prob = np.zeros_like(ndvi)
    
    # Rule 1: Vegetation presence (NDVI-based)
    vegetation_mask = (ndvi > 0.2) & (ndvi < 0.9) # Reasonable vegetation range
    farmland_prob += 0.3 * vegetation_mask.astype(float)
    
    # Rule 2: Moderate vegetation density (crops vs forests)
    moderate_veg_mask = (ndvi > 0.3) & (ndvi < 0.75) # Crops typically in this range
    farmland_prob += 0.2 * moderate_veg_mask.astype(float)
    
    # Rule 3: Enhanced Vegetation Index check
    if evi is not None and not np.all(np.isnan(evi)):
        healthy_crops_mask = (evi > 0.2) & (evi < 0.8)
        farmland_prob += 0.2 * healthy_crops_mask.astype(float)
    
    # Rule 4: Exclude water bodies
    water_mask = ndwi > 0.3
    farmland_prob -= 0.5 * water_mask.astype(float)
    
    # Rule 5: Exclude urban/built-up areas
    if ndbi is not None:
        urban_mask = ndbi > 0.1
        farmland_prob -= 0.4 * urban_mask.astype(float)
    
    # Rule 6: Soil presence (using SAVI)
    if savi is not None and not np.all(np.isnan(savi)):
        soil_adjusted_mask = (savi > 0.1) & (savi < 0.7)
        farmland_prob += 0.15 * soil_adjusted_mask.astype(float)
    
    # Rule 7: Texture analysis (simple version)
    # Agricultural areas often have more uniform patterns within fields
    if ndimage is not None:
        try:
            # Calculate local standard deviation as a texture measure
            texture = ndimage.generic_filter(ndvi, np.nanstd, size=5)
            # Moderate texture (not too uniform like water, not too varied like forests)
            texture_mask = (texture > 0.02) & (texture < 0.15)
            farmland_prob += 0.1 * texture_mask.astype(float)
        except Exception as e:
            print(f"WARN: Skipping texture analysis due to error: {e}") # Skip texture analysis if scipy is problematic
    
    # Normalize probability to 0-1 range
    farmland_prob = np.clip(farmland_prob, 0, 1)
    
    # Apply confidence threshold
    is_farmland = farmland_prob >= confidence_threshold
    
    return is_farmland, farmland_prob

def classify_ndvi_cell_name_with_farmland(mean_ndvi, is_farmland_cell, farmland_confidence):
    """Enhanced NDVI classification that considers farmland detection."""
    if not is_farmland_cell:
        return "black" # Not farmland, don't classify vegetation stress
    
    if np.isnan(mean_ndvi):
        return "black" 
    
    # For farmland areas, use enhanced thresholds
    if mean_ndvi > 0.65:
        return "green" # Healthy crops
    elif mean_ndvi > 0.35:
        return "yellow" # Moderate stress
    elif mean_ndvi > -0.05:
        return "red" # High stress
    else:
        return "black" # Very poor/no vegetation

# -------------------------------
# Utility Functions (Retained from original code)
# -------------------------------
def compute_ndvi(nir, red, eps=1e-6):
    """Computes Normalized Difference Vegetation Index (NDVI) robustly."""
    denominator = nir + red
    # Handle division by zero/near zero with np.where
    return np.where(np.abs(denominator) > eps, (nir - red) / denominator, np.nan)

def normalize_to_uint8(img, enhance_contrast=True):
    """Normalizes a multi-band float image to 0-255 uint8 with optional contrast enhancement."""
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(img.shape[2]):
        ch = img[:,:,c]
        valid_data = ch[~np.isnan(ch)]
        if valid_data.size == 0: continue
        
        if enhance_contrast:
            # Use 1-99 percentile for better contrast 
            lo, hi = np.nanpercentile(valid_data, 1), np.nanpercentile(valid_data, 99)
        else:
            # Use 5-95 percentile for more conservative stretching
            lo, hi = np.nanpercentile(valid_data, 5), np.nanpercentile(valid_data, 95)
        
        if hi - lo < 1e-6:
            out[:,:,c] = np.clip(ch, 0, 255).astype(np.uint8) 
        else:
            scaled = (ch - lo) / (hi - lo) * 255
            out[:,:,c] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out

def classify_ndvi_cell_name(mean_ndvi):
    """Classifies mean NDVI into stress categories."""
    if np.isnan(mean_ndvi):
        return "black" 
    if mean_ndvi > 0.6:
        return "green"
    elif mean_ndvi > 0.3:
        return "yellow"
    elif mean_ndvi > -0.1:
        return "red"
    else:
        return "black"

def generate_grid_outputs(rgb_uint8, ndvi, grid_size=(20,20), opacity=0.7, detect_farmland=True, farmland_confidence=0.6, red=None, nir=None, green=None, blue=None):
    """Generates the classified grid overlay on the RGB image with optional farmland detection."""
    H, W = ndvi.shape
    gw, gh = grid_size # Note: The original code defined (grid_width, grid_height) but used gw, gh
    nx = math.ceil(W / gw)
    ny = math.ceil(H / gh)
    classification_array = np.zeros_like(ndvi, dtype=np.uint8)
    
    # Detect farmland areas if enabled
    farmland_mask = None
    if detect_farmland and red is not None and nir is not None and green is not None and blue is not None:
        print("INFO: Running farmland detection...")
        # Note: swir1/swir2 are currently None in this function call, but the detect_agricultural_areas handles None.
        farmland_mask, farmland_prob = detect_agricultural_areas(
            red, nir, green, blue, confidence_threshold=farmland_confidence
        )
        print(f"INFO: {np.sum(farmland_mask) / farmland_mask.size * 100:.1f}% of area detected as farmland")
    
    img = Image.fromarray(rgb_uint8.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    alpha_value = int(opacity * 255)

    for i in range(ny):
        for j in range(nx):
            x0, y0 = j*gw, i*gh
            x1, y1 = min((j+1)*gw,W), min((i+1)*gh,H)
            
            if x0 >= W or y0 >= H: continue
                
            cell_ndvi = ndvi[y0:y1, x0:x1]
            if cell_ndvi.size == 0: continue
            
            mean_ndvi = np.nanmean(cell_ndvi)
            
            # Check if this cell is farmland
            is_farmland_cell = True # Default to True if not detecting farmland
            if detect_farmland and farmland_mask is not None:
                cell_farmland = farmland_mask[y0:y1, x0:x1]
                farmland_percentage = np.sum(cell_farmland) / cell_farmland.size
                is_farmland_cell = farmland_percentage > 0.3 # At least 30% of cell should be farmland
            
            # Use appropriate classification function
            if detect_farmland:
                cls_name = classify_ndvi_cell_name_with_farmland(mean_ndvi, is_farmland_cell, farmland_confidence)
            else:
                cls_name = classify_ndvi_cell_name(mean_ndvi)
            
            classification_array[y0:y1, x0:x1] = CLASS_VALUE_MAP[cls_name]
            
            fill_alpha = 0 if cls_name=="black" else alpha_value
            fill_color = CLASS_COLOR_MAP[cls_name] + (fill_alpha,)
            
            draw.rectangle([x0,y0,x1-1,y1-1], fill=fill_color)

    grid_lines = Image.new("RGBA", img.size, (0,0,0,0))
    gdraw = ImageDraw.Draw(grid_lines)
    line_color = (0,0,0,80)
    for i in range(ny+1):
        gdraw.line([(0,i*gh),(W,i*gh)], fill=line_color, width=1)
    for j in range(nx+1):
        gdraw.line([(j*gw,0),(j*gw,H)], fill=line_color, width=1)

    composed_png = Image.alpha_composite(img, overlay)
    composed_png = Image.alpha_composite(composed_png, grid_lines)
    return composed_png, classification_array

def create_classified_geotiff(classification_array, profile):
    """Creates a GeoTIFF of the classification array."""
    new_profile = profile.copy()
    new_profile.update(
        dtype=rasterio.uint8, 
        count=1, 
        nodata=CLASS_VALUE_MAP["black"],
        driver='GTiff', 
        interleave='band'
    )
    
    with MemoryFile() as memfile:
        with memfile.open(**new_profile) as dst:
            # Check for 3D data and handle if necessary (classification_array should be 2D here)
            if classification_array.ndim == 2:
                dst.write(classification_array[np.newaxis,...].astype(rasterio.uint8), 1)
            else:
                # Should not happen if data is classified correctly, but safe guard
                dst.write(classification_array.astype(rasterio.uint8), 1)

            dst.set_band_description(1,"Veg Stress Classification")
        return memfile.read()

# -------------------------------
# FastAPI Endpoints
# -------------------------------
@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/classify-location")
async def classify_location(request: LocationRequest):
    lat, lon = request.latitude, request.longitude
    start_date, end_date = request.start_date, request.end_date
    params = request.parameters

    # Buffer the point by configurable distance for zoom control
    region = ee.Geometry.Point(lon,lat).buffer(params.buffer_size).bounds()

    try:
        # 1. Image Collection Filtering
        dataset = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', params.cloud_percentage))
            .map(mask_s2_clouds)
        )
        
        image_count = dataset.size()
        print(f"Found {image_count.getInfo()} images after cloud filtering")
        
        if image_count.getInfo() == 0:
            print("No images found with strict cloud filter, trying relaxed filter...")
            dataset = (
                ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterDate(start_date, end_date)
                .filterBounds(region)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
                .map(mask_s2_clouds)
            )
            image_count = dataset.size()
            print(f"Found {image_count.getInfo()} images with relaxed cloud filtering")
        
        if image_count.getInfo() == 0:
            raise HTTPException(status_code=404, detail=f"No Sentinel-2 images found for coordinates ({lat}, {lon}) between {start_date} and {end_date}. Try a different date range.")
        
        image_composite = dataset.mean()
        band_names_available = image_composite.bandNames()

        # 2. Band Selection
        bands_to_get = list({params.red_band, params.nir_band, params.rgb_b1, params.rgb_b2, params.rgb_b3})
        
        if params.detect_farmland:
            additional_bands = {'B11', 'B12'}
            bands_to_get.extend(additional_bands)
            bands_to_get = list(set(bands_to_get))
        
        available_bands = band_names_available.getInfo()
        missing_bands = [band for band in bands_to_get if band not in available_bands]
        if missing_bands:
            raise HTTPException(status_code=400, detail=f"Requested bands {missing_bands} not available. Available bands: {available_bands}")
        
        image_selected = image_composite.select(bands_to_get)

        # 3. Download Image Data (GEEMAP REPLACEMENT)
        numpy_data, profile = download_ee_image_to_numpy(
            image_selected, 
            ee_region=region, 
            scale=params.image_scale
        )
        
        # 4. Data Extraction and Reorganization
        # Create a mapping from band name to its index in the numpy_data array
        band_map_indices = {name: i for i, name in enumerate(bands_to_get)}

        # Extract components using the defined band names
        red_idx = band_map_indices[params.red_band]
        nir_idx = band_map_indices[params.nir_band]
        
        red = numpy_data[..., red_idx]
        nir = numpy_data[..., nir_idx]
        
        # Extract Green and Blue for both RGB and Farmland detection
        green = numpy_data[..., band_map_indices[params.rgb_b2]]
        blue = numpy_data[..., band_map_indices[params.rgb_b3]]
        
        # Stack RGB for visualization
        rgb_indices = [band_map_indices[params.rgb_b1], band_map_indices[params.rgb_b2], band_map_indices[params.rgb_b3]]
        rgb = numpy_data[..., rgb_indices]
        
        # SWIR bands for advanced land use detection (if available)
        swir1 = numpy_data[..., band_map_indices['B11']] if 'B11' in band_map_indices else None
        swir2 = numpy_data[..., band_map_indices['B12']] if 'B12' in band_map_indices else None

    except HTTPException:
        raise 
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing satellite image data: {e}")

    # 5. Processing Steps
    ndvi = compute_ndvi(nir, red)
    rgb_uint8 = normalize_to_uint8(rgb, enhance_contrast=params.enhance_contrast)
    
    composed_png, classification_array = generate_grid_outputs(
        rgb_uint8, ndvi, 
        # Note: grid_width is used as X (width), grid_height as Y (height) in logic
        grid_size=(params.grid_width, params.grid_height), 
        opacity=params.opacity,
        detect_farmland=params.detect_farmland,
        farmland_confidence=params.farmland_confidence,
        red=red, nir=nir, green=green, blue=blue,
        # swir1/swir2 are implicitly used inside detect_agricultural_areas
    )

    # Apply upscaling if requested
    if params.upscale_factor > 1.0:
        original_size = composed_png.size
        new_size = (int(original_size[0] * params.upscale_factor), int(original_size[1] * params.upscale_factor))
        composed_png = composed_png.resize(new_size, Image.Resampling.LANCZOS)

    # 6. Output Generation
    if params.output_format.lower()=='png':
        buf = BytesIO()
        composed_png.save(buf, format="PNG", optimize=True, compress_level=1) 
        buf.seek(0)
        return StreamingResponse(
            buf, 
            media_type="image/png", 
            headers={"Content-Disposition":"inline; filename=sentinel2_classification.png"}
        )
    else: # GeoTIFF output
        # Rasterio profile is already available from the download step
        geotiff_data = create_classified_geotiff(classification_array, profile)
        buf = BytesIO(geotiff_data)
        return StreamingResponse(
            buf, 
            media_type="image/tiff", 
            headers={"Content-Disposition":"attachment; filename=sentinel2_classification.tif"}
        )

# -------------------------------
# Run App
# -------------------------------
if __name__=="__main__":
    import uvicorn
    # Use 0.0.0.0 for deployment to make the service accessible externally
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
