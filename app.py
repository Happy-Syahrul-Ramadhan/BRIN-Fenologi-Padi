from flask import Flask, render_template, jsonify, request
import ee
import json
import logging
from datetime import datetime
from functools import wraps
import geemap
import geemap.foliumap as geemap
import numpy as np
import pandas as pd
import joblib
import os
from itertools import zip_longest as zip

# Import konfigurasi dari file terpisah
from config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Konfigurasi dan Parameter - menggunakan config dari environment
SCALE = config.SCALE
MAX_TRAINING_POINTS = config.MAX_TRAINING_POINTS
BANDS_SELECTED = ['VV_int', 'VH_int', 'RPI', 'API', 'NDPI', 'RVI', 'angle']

# Warna berdasarkan siklus pertumbuhan padi: vegetatif -> generatif
# Hijau muda -> Hijau tua -> Kuning -> Coklat
PALET_RICE_PHASES = ['#32CD32', '#FF1493', '#FF4500', '#4B0082']  # Light Green, Green, Gold, Brown
RICE_PHASE_ORDER = ['vegetatif 1', 'vegetatif 2', 'generatif 1', 'generatif 2']  # Urutan siklus pertumbuhan

PALET = ['#00FFFF', '#FFD700', '#32CD32', '#FF1493', '#FF4500', '#4B0082', '#8B4513']  # untuk backward compatibility
LABEL = ['Unknown', 'Air', 'Early Vegetatif', 'Vegetatif 1', 'Vegetatif 2', 'Generatif 1', 'Generatif 2', 'Bare']

# Mapping dasarian to month names
MONTH_NAMES = [
    'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
    'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'
]

def get_ordered_rice_phases():
    """
    Mendapatkan mapping fase padi berdasarkan urutan pertumbuhan yang benar
    """
    model = load_trained_model()
    if model and hasattr(model, 'classes_'):
        model_classes = list(model.classes_)
        
        # Mapping fase ke urutan yang benar berdasarkan siklus pertumbuhan
        ordered_phases = []
        ordered_colors = []
        
        # Urutkan berdasarkan RICE_PHASE_ORDER
        for phase in RICE_PHASE_ORDER:
            if phase in model_classes:
                ordered_phases.append(phase)
                phase_index = RICE_PHASE_ORDER.index(phase)
                ordered_colors.append(PALET_RICE_PHASES[phase_index])
        
        # Tambahkan fase lain yang mungkin ada di model tapi tidak di RICE_PHASE_ORDER
        for phase in model_classes:
            if phase not in ordered_phases:
                ordered_phases.append(phase)
                # Gunakan warna default untuk fase tambahan
                extra_index = len(ordered_phases) - 1
                if extra_index < len(PALET):
                    ordered_colors.append(PALET[extra_index])
                else:
                    ordered_colors.append('#808080')  # Abu-abu untuk fase tidak dikenali
        
        return ordered_phases, ordered_colors
    else:
        # Fallback jika model tidak tersedia
        return RICE_PHASE_ORDER, PALET_RICE_PHASES

def get_dasarian_info(dasarian):
    """Get month and period info for a dasarian"""
    month = ((dasarian - 1) // 3) + 1
    period = ((dasarian - 1) % 3) + 1
    month_name = MONTH_NAMES[month - 1]
    return {
        'dasarian': dasarian,
        'month': month,
        'period': period,
        'month_name': month_name,
        'display_name': f"{month_name} P{period}",
        'full_name': f"Dasarian {dasarian} ({month_name} Periode {period})"
    }

def date_to_dasarian(date_str):
    """Convert date string (YYYY-MM-DD) to dasarian"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        month = date_obj.month
        day = date_obj.day
        
        # Determine period within month
        if day <= 10:
            period = 1
        elif day <= 20:
            period = 2
        else:
            period = 3
            
        # Calculate dasarian (1-36)
        dasarian = (month - 1) * 3 + period
        return dasarian
    except:
        return None

def dasarian_to_date_range(dasarian):
    """Convert dasarian to date range"""
    try:
        month = ((dasarian - 1) // 3) + 1
        period = ((dasarian - 1) % 3) + 1
        
        if period == 1:
            start_day, end_day = 1, 10
        elif period == 2:
            start_day, end_day = 11, 20
        else:
            start_day, end_day = 21, 31
            
        # Use current year (or you can make this configurable)
        year = datetime.now().year
        
        start_date = f"{year}-{month:02d}-{start_day:02d}"
        end_date = f"{year}-{month:02d}-{end_day:02d}"
        
        return start_date, end_date
    except:
        return None, None

# Global variable untuk menyimpan model yang sudah dimuat
loaded_model = None

def load_trained_model():
    """Load model Random Forest yang sudah dilatih dengan joblib"""
    global loaded_model
    
    if loaded_model is None:
        try:
            if os.path.exists(config.MODEL_PATH):
                loaded_model = joblib.load(config.MODEL_PATH)
                logger.info(f"Model berhasil dimuat dari {config.MODEL_PATH}")
                logger.info(f"Fitur yang digunakan: {loaded_model.feature_names_in_}")
                logger.info(f"Jumlah kelas: {len(loaded_model.classes_)}")
                logger.info(f"Kelas: {loaded_model.classes_}")
            else:
                logger.error(f"File model tidak ditemukan: {config.MODEL_PATH}")
                return None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    return loaded_model

def get_indramayu_aoi():
    """
    Mendefinisikan Area of Interest untuk Kabupaten Indramayu
    """
    try:
        # Gunakan asset boundary Indramayu dari konfigurasi environment
        aoi_asset = ee.Image(config.INDRAMAYU_AOI_ASSET)
        # Ambil geometry dari bounds image
        aoi_geometry = aoi_asset.geometry()
        
        logger.info("Indramayu AOI loaded from asset successfully")
        return aoi_geometry
        
    except Exception as e:
        logger.error(f"Error loading AOI from asset: {str(e)}")
        logger.info("Using fallback coordinates for Indramayu AOI")
        
        # Fallback ke koordinat manual yang lebih luas untuk area Indramayu
        indramayu_coords = [
            [107.8, -6.0],   # Northwest - lebih luas
            [108.6, -6.0],   # Northeast  
            [108.6, -6.6],   # Southeast
            [107.8, -6.6],   # Southwest
            [107.8, -6.0]    # Close polygon
        ]
        
        return ee.Geometry.Polygon(indramayu_coords)

def check_sentinel1_availability(aoi):
    """
    Cek ketersediaan data Sentinel-1 di area tertentu
    """
    try:
        logger.info("Checking Sentinel-1 data availability...")
        
        # Check data availability for current year
        current_year = datetime.now().year
        year_start = f"{current_year}-01-01"
        year_end = f"{current_year}-12-31"
        
        s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filterDate(year_start, year_end) \
            .filterBounds(aoi)
        
        total_images = s1_collection.size().getInfo()
        logger.info(f"Total Sentinel-1 images available in {current_year}: {total_images}")
        
        if total_images > 0:
            # Get some sample dates
            sample_dates = s1_collection.limit(5).aggregate_array('system:time_start').getInfo()
            logger.info(f"Sample dates: {[datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in sample_dates]}")
        
        return total_images > 0
        
    except Exception as e:
        logger.error(f"Error checking Sentinel-1 availability: {str(e)}")
        return False

def get_sentinel1_data_realtime(start_date, end_date, aoi):
    """
    Mengambil data Sentinel-1 real-time dari GEE berdasarkan tanggal
    """
    try:
        logger.info(f"Fetching real-time Sentinel-1 data from {start_date} to {end_date}")
        
        # Coba filter dasar tanpa pembatasan orbit yang ketat
        s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filterDate(start_date, end_date) \
            .filterBounds(aoi)
        
        collection_size = s1_collection.size().getInfo()
        logger.info(f"Found {collection_size} Sentinel-1 images for initial search")
        
        if collection_size == 0:
            # Extend date range by 2 months
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            extended_start = (start_dt - timedelta(days=60)).strftime('%Y-%m-%d')
            extended_end = (end_dt + timedelta(days=60)).strftime('%Y-%m-%d')
            
            logger.info(f"No data found, extending search to {extended_start} - {extended_end}")
            
            s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filterDate(extended_start, extended_end) \
                .filterBounds(aoi)
            
            collection_size = s1_collection.size().getInfo()
            logger.info(f"Extended search found {collection_size} images")
            
            if collection_size == 0:
                # Try with current year data and broader area
                current_year = datetime.now().year
                year_start = f"{current_year}-01-01"
                year_end = f"{current_year}-12-31"
                
                # Expand AOI slightly
                aoi_buffered = aoi.buffer(5000)  # 5km buffer
                
                logger.info(f"Trying whole year {current_year} with buffered AOI")
                
                s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                    .filterDate(year_start, year_end) \
                    .filterBounds(aoi_buffered) \
                    .sort('system:time_start', False) \
                    .limit(50)  # Get 50 most recent images
                
                collection_size = s1_collection.size().getInfo()
                logger.info(f"Year search with buffer found {collection_size} images")
                
                if collection_size == 0:
                    # Last resort: try last 2 years without orbit restriction
                    past_year = current_year - 1
                    past_start = f"{past_year}-01-01"
                    
                    logger.info(f"Last resort: searching from {past_start} to {year_end}")
                    
                    s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                        .filterDate(past_start, year_end) \
                        .filterBounds(aoi_buffered) \
                        .sort('system:time_start', False) \
                        .limit(20)
                    
                    collection_size = s1_collection.size().getInfo()
                    logger.info(f"Final search found {collection_size} images")
        
        if collection_size == 0:
            logger.warning("No Sentinel-1 data found even after exhaustive search")
            return None
        
        logger.info(f"Successfully found {collection_size} Sentinel-1 images")
        
        # Apply preprocessing pipeline
        processed_collection = s1_collection \
            .map(remove_border_noise) \
            .map(terrain_correction) \
            .map(speckle_filter) \
            .map(calculate_vegetation_indices_s1)
        
        return processed_collection
        
    except Exception as e:
        logger.error(f"Error getting Sentinel-1 data: {str(e)}")
        return None

def remove_border_noise(img):
    """Remove border noise dari citra Sentinel-1"""
    try:
        # Check if image has VV band (for threshold)
        band_names = img.bandNames()
        has_vv = band_names.contains('VV')
        
        # Only apply if VV band exists, otherwise return original
        cleaned_img = ee.Algorithms.If(
            has_vv,
            # Apply border noise removal
            ee.Image(img).updateMask(
                img.mask().And(img.select('VV').gt(-35))
            ).copyProperties(img, img.propertyNames()),
            # Return original if no VV band
            img
        )
        
        return ee.Image(cleaned_img)
    except Exception as e:
        logger.error(f"Error in remove_border_noise: {str(e)}")
        return img

def terrain_correction(img):
    """Koreksi terrain/incident angle (Gamma Nought)"""
    try:
        # Check if raw bands exist
        band_names = img.bandNames()
        has_vv = band_names.contains('VV')
        has_vh = band_names.contains('VH')
        
        # Only apply if raw VV/VH bands exist
        corrected_img = ee.Algorithms.If(
            ee.Algorithms.And(has_vv, has_vh),
            # Apply terrain correction
            img.expression(
                'i - 10 * log10(cos(angle * pi / 180))', {
                    'i': img.select(['VV', 'VH']),
                    'angle': img.select('angle'),
                    'pi': 3.14159265359
                }
            ).toFloat().copyProperties(img, img.propertyNames()),
            # Return original if no raw bands
            img
        )
        
        return ee.Image(corrected_img)
    except Exception as e:
        logger.error(f"Error in terrain_correction: {str(e)}")
        return img

def speckle_filter(img):
    """Aplikasi speckle filter menggunakan median filter"""
    try:
        properties = img.propertyNames()
        return img.focalMedian(5).copyProperties(img, properties)
    except Exception as e:
        logger.error(f"Error in speckle_filter: {str(e)}")
        return img

def calculate_vegetation_indices_s1(img):
    """
    Menghitung indeks vegetasi dari citra Sentinel-1 yang sudah diproses
    Menggunakan pendekatan yang sama dengan kode JavaScript
    """
    try:
        # Cek band yang tersedia
        band_names = img.bandNames()
        
        # Jika sudah ada VV_int dan VH_int (dari koleksi yang sudah diproses)
        has_vv_int = band_names.contains('VV_int')
        has_vh_int = band_names.contains('VH_int')
        has_vv = band_names.contains('VV')
        has_vh = band_names.contains('VH')
        
        # Gunakan kondisional EE tanpa .getInfo()
        vv_int = ee.Algorithms.If(
            ee.Algorithms.And(has_vv_int, has_vh_int),
            # Gunakan data linear yang sudah ada
            img.select('VV_int').toFloat(),
            ee.Algorithms.If(
                ee.Algorithms.And(has_vv, has_vh),
                # Konversi dari dB ke linear untuk data mentah Sentinel-1
                img.select('VV').toFloat().expression('10**(vv / 10)', {'vv': img.select('VV').toFloat()}).rename('VV_int').toFloat(),
                ee.Image.constant(0).rename('VV_int').toFloat()
            )
        )
        
        vh_int = ee.Algorithms.If(
            ee.Algorithms.And(has_vv_int, has_vh_int),
            # Gunakan data linear yang sudah ada
            img.select('VH_int').toFloat(),
            ee.Algorithms.If(
                ee.Algorithms.And(has_vv, has_vh),
                # Konversi dari dB ke linear untuk data mentah Sentinel-1
                img.select('VH').toFloat().expression('10**(vh / 10)', {'vh': img.select('VH').toFloat()}).rename('VH_int').toFloat(),
                ee.Image.constant(0).rename('VH_int').toFloat()
            )
        )
        
        # Cast hasil kondisional ke Image
        vv_int = ee.Image(vv_int)
        vh_int = ee.Image(vh_int)
        
        # Hitung indeks menggunakan data linear
        RPI = vh_int.divide(vv_int.add(0.001)).rename('RPI').toFloat()
        API = vv_int.add(vh_int).divide(2).rename('API').toFloat() 
        NDPI = vv_int.subtract(vh_int).divide(vv_int.add(vh_int).add(0.001)).rename('NDPI').toFloat()
        RVI = vh_int.multiply(4).divide(vv_int.add(vh_int).add(0.001)).rename('RVI').toFloat()
        
        # Pastikan ada band angle
        has_angle = band_names.contains('angle')
        angle = ee.Algorithms.If(
            has_angle,
            img.select('angle'),
            ee.Image.constant(23).rename('angle')
        )
        angle = ee.Image(angle)
        
        # Gabungkan semua band dalam format linear
        result = ee.Image.cat([
            vv_int,
            vh_int,
            angle,
            RPI, API, NDPI, RVI
        ]).copyProperties(img, img.propertyNames())
        
        return result
    
    except Exception as e:
        logger.error(f"Error in calculate_vegetation_indices_s1: {str(e)}")
        # Fallback: return original image if processing fails
        return img

def calculate_vegetation_indices(image):
    """
    Menghitung indeks vegetasi dari citra Sentinel-1
    """
    # Get band names to check what's available
    band_names = image.bandNames()
    
    # Check if VV_int and VH_int exist, otherwise use VV and VH
    has_vv_int = band_names.contains('VV_int')
    has_vh_int = band_names.contains('VH_int')
    has_vv = band_names.contains('VV')
    has_vh = band_names.contains('VH')
    
    # Select appropriate bands based on availability
    vv = ee.Algorithms.If(
        has_vv_int,
        image.select('VV_int'),
        ee.Algorithms.If(
            has_vv,
            image.select('VV').rename('VV_int'),
            ee.Image.constant(0.01).rename('VV_int')
        )
    )
    
    vh = ee.Algorithms.If(
        has_vh_int,
        image.select('VH_int'),
        ee.Algorithms.If(
            has_vh,
            image.select('VH').rename('VH_int'),
            ee.Image.constant(0.005).rename('VH_int')
        )
    )
    
    # Convert to images
    vv = ee.Image(vv)
    vh = ee.Image(vh)
    
    # Hitung RPI (Radar Polarization Index)
    rpi = vh.divide(vv.add(0.001)).rename('RPI')  # Add small constant to avoid division by zero
    
    # Hitung API (Adjusted Polarization Index)
    api = vh.subtract(vv).divide(vh.add(vv).add(0.001)).rename('API')
    
    # Hitung NDPI (Normalized Difference Polarization Index)
    ndpi = vh.subtract(vv).divide(vh.add(vv).add(0.001)).rename('NDPI')
    
    # Hitung RVI (Radar Vegetation Index)
    rvi = vh.multiply(4).divide(vv.add(vh).add(0.001)).rename('RVI')
    
    # Tambahkan angle (sudut insiden) - nilai default atau dari metadata
    angle = ee.Image.constant(23).rename('angle')
    
    return image.addBands([vv, vh, rpi, api, ndpi, rvi, angle])

def create_classifier_from_trained_model():
    """
    Membuat classifier EE dari model yang sudah dilatih
    Menggunakan pendekatan hybrid: train EE classifier dengan parameter yang mirip
    """
    model = load_trained_model()
    if model is None:
        return None
    
    try:
        # Load training data dari konfigurasi environment
        titik_pelatihan = ee.FeatureCollection(config.TRAINING_POINTS_ASSET)
        koleksi_pelatihan = ee.ImageCollection(config.COLLECTION_ASSET)
        
        # Limit training points
        titik_pelatihan_limit = titik_pelatihan.limit(MAX_TRAINING_POINTS)
        
        # Konversi fase ke numerik sesuai dengan urutan yang benar
        def konversi_label_numerik(feature):
            fase_string = ee.String(feature.get('Fase')).toLowerCase().trim()
            
            # Map fase ke angka berdasarkan RICE_PHASE_ORDER (0-3 untuk 4 kelas)
            fase_numerik = ee.Algorithms.If(
                fase_string.equals('vegetatif 1'), 0,  # vegetatif 1 -> 0 (hijau muda)
                ee.Algorithms.If(
                    fase_string.equals('vegetatif 2'), 1,  # vegetatif 2 -> 1 (hijau tua)
                    ee.Algorithms.If(
                        fase_string.equals('generatif 1'), 2,  # generatif 1 -> 2 (kuning)
                        ee.Algorithms.If(
                            fase_string.equals('generatif 2'), 3,  # generatif 2 -> 3 (coklat)
                            # Default untuk fase lain yang tidak dikenali
                            0
                        )
                    )
                )
            )
            return feature.set('FaseNumerik', fase_numerik)
        
        titik_pelatihan_numerik = titik_pelatihan_limit.map(konversi_label_numerik)
        
        # Prepare training collection - gunakan langsung tanpa map calculate_vegetation_indices
        # karena collection sudah memiliki band yang diperlukan
        koleksi_latih = koleksi_pelatihan.limit(20)
        
        # Extract features from a single image
        first_image = ee.Image(koleksi_latih.first()).select(BANDS_SELECTED)
        
        # Sample regions directly
        data_latih = first_image.sampleRegions(
            collection=titik_pelatihan_numerik,
            properties=['FaseNumerik'],
            scale=SCALE,
            geometries=False
        ).filter(ee.Filter.notNull(BANDS_SELECTED + ['FaseNumerik']))
        
        # Split data
        data_acak = data_latih.randomColumn('random', 42)
        train_set = data_acak.filter(ee.Filter.lt('random', 0.8))
        
        # Buat classifier dengan parameter yang mirip dengan model yang sudah dilatih
        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=getattr(model, 'n_estimators', 100),
            variablesPerSplit=None,
            minLeafPopulation=getattr(model, 'min_samples_leaf', 1),
            bagFraction=0.5,
            maxNodes=None,
            seed=42
        ).train(
            features=train_set,
            classProperty='FaseNumerik',
            inputProperties=BANDS_SELECTED
        )
        
        return classifier
        
    except Exception as e:
        logger.error(f"Error creating classifier: {str(e)}")
        return None

def classify_with_dasarian_filter_asset(dasarian_start=1, dasarian_end=36):
    """
    Klasifikasi menggunakan collection yang sudah ada di asset berdasarkan dasarian
    Menggunakan collection yang sudah dipreprocessing tanpa perubahan
    """
    try:
        logger.info(f"Classifying using ASSET collection for dasarian {dasarian_start} to {dasarian_end}")
        
        # Load classifier
        classifier = create_classifier_from_trained_model()
        if classifier is None:
            logger.error("Classifier tidak tersedia")
            return None
        
        # Load collection dari asset menggunakan konfigurasi environment
        collection = ee.ImageCollection(config.COLLECTION_ASSET)
        
        # Untuk klasifikasi dasarian, gunakan pendekatan sederhana berdasarkan index
        if dasarian_start == dasarian_end:
            # Ambil image berdasarkan index dasarian
            collection_list = collection.toList(collection.size())
            image_index = ee.Number(dasarian_start - 1).min(collection.size().subtract(1))
            target_image = ee.Image(collection_list.get(image_index))
        else:
            # Untuk range dasarian, ambil median dari collection
            total_images = collection.size().getInfo()
            if total_images is None or total_images == 0:
                logger.error("No images found in collection")
                return None
                
            start_idx = max(0, dasarian_start - 1)
            end_idx = min(total_images - 1, dasarian_end - 1)
            
            if start_idx < end_idx:
                # Ambil subset collection
                subset_list = collection.toList(total_images).slice(start_idx, end_idx + 1)
                subset_collection = ee.ImageCollection.fromImages(subset_list)
                target_image = subset_collection.median()
            else:
                # Fallback ke single image
                collection_list = collection.toList(collection.size())
                image_index = ee.Number(dasarian_start - 1).min(collection.size().subtract(1))
                target_image = ee.Image(collection_list.get(image_index))
        
        # Perform classification - pastikan band yang digunakan tersedia
        classified = target_image.select(BANDS_SELECTED).classify(classifier).rename('classification')
        
        # Set metadata
        dasarian_info = get_dasarian_info(dasarian_start)
        classified = classified.set({
            'dasarian': dasarian_start,
            'month_name': dasarian_info['month_name'],
            'display_name': dasarian_info['display_name'],
            'data_source': 'asset_collection'
        })
        
        logger.info(f"Asset classification completed for dasarian {dasarian_start}")
        return classified
        
    except Exception as e:
        logger.error(f"Error in asset dasarian classification: {str(e)}")
        return None

def classify_with_date_filter_realtime(start_date, end_date):
    """
    Klasifikasi menggunakan data Sentinel-1 real-time berdasarkan tanggal
    """
    try:
        logger.info(f"Classifying using REAL-TIME Sentinel-1 data for {start_date} to {end_date}")
        
        # Get AOI Indramayu
        aoi = get_indramayu_aoi()
        
        # Check data availability first
        if not check_sentinel1_availability(aoi):
            logger.warning("No Sentinel-1 data available in the area")
            return None
        
        # Get real-time Sentinel-1 data
        s1_data = get_sentinel1_data_realtime(start_date, end_date, aoi)
        
        if s1_data is None or s1_data.size().getInfo() == 0:
            logger.warning(f"No Sentinel-1 data found for {start_date} to {end_date}")
            return None
        
        # Create classifier
        classifier = create_classifier_from_trained_model()
        if classifier is None:
            logger.error("Classifier tidak tersedia")
            return None
        
        # Get median composite from the filtered collection
        composite_image = s1_data.median()
        
        # Perform classification
        classified = composite_image.select(BANDS_SELECTED).classify(classifier).rename('classification')
        
        # Set metadata
        classified = classified.set({
            'start_date': start_date,
            'end_date': end_date,
            'data_source': 'Sentinel-1_realtime'
        })
        
        logger.info(f"Real-time classification completed for period {start_date} - {end_date}")
        return classified
        
    except Exception as e:
        logger.error(f"Error in real-time date classification: {str(e)}")
        return None

def calculate_area_statistics(classified_image, scale=SCALE):
    """
    Menghitung statistik area untuk setiap kelas dalam hektar dengan persentase
    """
    try:
        logger.info("Starting area statistics calculation...")
        
        # Gunakan AOI Indramayu
        study_area = get_indramayu_aoi()
        
        # Hitung area untuk setiap kelas
        # Konversi pixel ke area dalam meter persegi, kemudian ke hektar
        pixel_area = scale * scale  # area per pixel dalam m²
        
        # Reduksi histogram untuk mendapatkan jumlah pixel per kelas
        logger.info("Computing histogram...")
        histogram = classified_image.reduceRegion(
            reducer=ee.Reducer.histogram(),
            geometry=study_area,
            scale=scale,
            maxPixels=1e9
        ).getInfo()
        
        logger.info(f"Histogram result: {histogram}")
        
        # Extract histogram data
        class_stats = {}
        total_pixels = 0
        total_area_hectares = 0
        total_area_m2 = 0  # Initialize total_area_m2
        
        if 'classification' in histogram and histogram['classification']:
            bucket_means = histogram['classification']['bucketMeans']
            counts = histogram['classification']['histogram']
            
            logger.info(f"Bucket means: {bucket_means}")
            logger.info(f"Counts: {counts}")
            
            # Get mapping fase yang terurut
            ordered_phases, ordered_colors = get_ordered_rice_phases()
            
            logger.info(f"Ordered phases: {ordered_phases}")
            logger.info(f"Ordered colors: {ordered_colors}")
            
            # Create mapping berdasarkan urutan yang benar
            class_mapping = {}
            for i, (phase_name, color) in enumerate(zip(ordered_phases, ordered_colors)):
                class_mapping[i] = {
                    'name': phase_name.title(),
                    'color': color
                }
            
            # Hitung total pixels dulu untuk persentase
            total_pixels = sum(counts)
            total_area_m2 = total_pixels * pixel_area
            total_area_hectares = total_area_m2 / 10000
            
            # Proses setiap kelas
            for i, (class_value, pixel_count) in enumerate(zip(bucket_means, counts)):
                class_idx = int(round(class_value))
                if class_idx in class_mapping and pixel_count > 0:
                    area_m2 = pixel_count * pixel_area
                    area_hectares = area_m2 / 10000  # konversi ke hektar
                    percentage = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
                    
                    class_stats[class_idx] = {
                        'class_name': class_mapping[class_idx]['name'],
                        'class_id': class_idx,
                        'pixel_count': round(pixel_count, 2),
                        'area_m2': round(area_m2, 2),
                        'area_hectares': round(area_hectares, 2),
                        'percentage': round(percentage, 2),
                        'color': class_mapping[class_idx]['color']
                    }
        
        # Tambahkan informasi total
        summary_stats = {
            'classes': class_stats,
            'total_pixels': round(total_pixels, 2),
            'total_area_hectares': round(total_area_hectares, 2),
            'total_area_m2': round(total_area_m2, 2),
            'number_of_classes': len(class_stats),
            'class_distribution': []
        }
        
        # Urutkan kelas berdasarkan persentase (terbesar ke terkecil)
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)
        
        for class_id, stats in sorted_classes:
            summary_stats['class_distribution'].append({
                'class_id': class_id,
                'class_name': stats['class_name'],
                'percentage': stats['percentage'],
                'area_hectares': stats['area_hectares'],
                'color': stats['color']
            })
        
        logger.info(f"Final class_stats summary: Total classes: {len(class_stats)}, Total area: {total_area_hectares:.2f} ha")
        logger.info("Class distribution:")
        for dist in summary_stats['class_distribution']:
            logger.info(f"  - {dist['class_name']}: {dist['percentage']:.2f}% ({dist['area_hectares']:.2f} ha)")
        
        return summary_stats
        
    except Exception as e:
        logger.error(f"Error calculating area statistics: {str(e)}")
        return None

# Visualization parameters for rice phase classification
def get_phase_vis_params():
    """Get visualization parameters based on actual model classes with proper rice phase colors"""
    ordered_phases, ordered_colors = get_ordered_rice_phases()
    n_classes = len(ordered_phases)
    
    return {
        'min': 0,
        'max': n_classes - 1,
        'palette': ordered_colors
    }

phase_vis_params = get_phase_vis_params()

def create_map(with_classification=False, dasarian_filter=None, start_date=None, end_date=None):
    """Create a base map using geemap with optional rice phase classification"""
    center_lat, center_lon = -6.3153, 108.3549
    
    # Create a geemap Map object
    my_map = geemap.Map(
        center=[center_lat, center_lon],
        zoom=10,
        height='500px',
        width='100%',
        layout=dict(
            height='500px',
            width='auto',
            margin='0px'
        )
    )
    
    # Add satellite basemap
    my_map.add_basemap('SATELLITE')
    
    # Note: geemap Map doesn't support direct style assignment
    # Style is controlled through the Map initialization parameters
    
    if with_classification:
        try:
            # Cek apakah model tersedia
            model = load_trained_model()
            if model is None:
                logger.error("Model tidak tersedia untuk klasifikasi")
                return my_map
            
            # Klasifikasi dengan filter dasarian atau tanggal
            if dasarian_filter:
                dasarian_start, dasarian_end = dasarian_filter
                classified_image = classify_with_dasarian_filter_asset(dasarian_start, dasarian_end)
                layer_name = f'Klasifikasi Fase Padi - Dasarian {dasarian_start}-{dasarian_end}'
            elif start_date and end_date:
                # Klasifikasi berdasarkan tanggal
                classified_image = classify_with_date_filter_realtime(start_date, end_date)
                layer_name = f'Klasifikasi Fase Padi - {start_date} to {end_date}'
            else:
                # Klasifikasi seluruh collection
                classified_image = classify_with_dasarian_filter_asset(1, 36)
                layer_name = 'Klasifikasi Fase Padi - Semua Dasarian'
            
            if classified_image:
                # Add classification layer
                my_map.addLayer(
                    classified_image,
                    phase_vis_params,
                    layer_name
                )
                
                # Add legend berdasarkan urutan fase pertumbuhan yang benar
                ordered_phases, ordered_colors = get_ordered_rice_phases()
                legend_dict = dict(zip([phase.title() for phase in ordered_phases], ordered_colors))
                legend_title = f"Fase Padi ({len(ordered_phases)} kelas) - Urutan Pertumbuhan"
                
                my_map.add_legend(
                    title=legend_title,
                    legend_dict=legend_dict,
                    position='bottomright'
                )
                
                logger.info(f"Classification layer added: {layer_name}")
            else:
                logger.error("Failed to create classified image")
        
        except Exception as e:
            logger.error(f"Error adding classification layers: {str(e)}")
    
    return my_map

def handle_ee_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ee.EEException as e:
            return jsonify({'error': f'Earth Engine error: {str(e)}'}), 500
        except Exception as e:
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    return decorated_function

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Earth Engine Initialization menggunakan konfigurasi dari environment
try:
    logger.info(f"Initializing Earth Engine with project: {config.GEE_PROJECT_ID}")
    ee.Initialize(project=config.GEE_PROJECT_ID)
    logger.info("Earth Engine initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize with project {config.GEE_PROJECT_ID}: {str(e)}")
    logger.info("Attempting authentication and re-initialization...")
    try:
        ee.Authenticate()
        ee.Initialize(project=config.GEE_PROJECT_ID)
        logger.info("Earth Engine authenticated and initialized successfully")
    except Exception as auth_error:
        logger.error(f"Failed to authenticate and initialize Earth Engine: {str(auth_error)}")
        raise

# Load model saat startup
logger.info("Loading trained model at startup...")
load_trained_model()

# Route utama
@app.route('/')
def home():
    try:
        # Load collection dengan indeks vegetasi menggunakan konfigurasi dari environment
        collection = ee.ImageCollection(config.COLLECTION_ASSET)
        collection_with_indices = collection.map(calculate_vegetation_indices)
        
        # Buat mosaic dari seluruh koleksi
        initial_mosaic = collection_with_indices.median()
        
        # Parameter visualisasi
        vis_params = {
            'bands': ['RPI', 'VV_int', 'VH_int'],
            'min': [0.043, 0.002, 1.65],
            'max': [0.273, 0.064, 8.135]
        }
        
        # Buat peta
        my_map = create_map()
        
        # Tambahkan layer Earth Engine
        my_map.addLayer(
            initial_mosaic, 
            vis_params, 
            'Citra Satelit - Komposit dengan Indeks'
        )
        
        # Tambahkan kontrol layer
        my_map.add_layer_control()
        
        # Convert map ke HTML
        map_html = my_map._repr_html_()
        
        return render_template('index.html', map_html=map_html)
    
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/rice-phase')
def rice_phase_view():
    """View for rice phase classification using trained model"""
    try:
        # Get parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        dasarian = request.args.get('dasarian', type=int)
        
        # Determine dasarian filter
        dasarian_filter = None
        if dasarian:
            dasarian_filter = (dasarian, dasarian)
        
        # Create map with classification
        my_map = create_map(
            with_classification=True,
            dasarian_filter=dasarian_filter,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert map to HTML
        map_html = my_map._repr_html_()
        
        # Get model info for template
        model = load_trained_model()
        model_info = {}
        if model:
            model_info = {
                'features': list(model.feature_names_in_),
                'classes': list(model.classes_),
                'n_estimators': model.n_estimators
            }
        
        # Create a list of zipped labels and colors berdasarkan urutan pertumbuhan yang benar
        ordered_phases, ordered_colors = get_ordered_rice_phases()
        zipped_data = list(zip([phase.title() for phase in ordered_phases], ordered_colors))
        
        # Generate dasarian options with month names
        dasarian_options = []
        for i in range(1, 37):
            dasarian_info = get_dasarian_info(i)
            dasarian_options.append(dasarian_info)
        
        return render_template(
            'rice_phase.html',
            map_html=map_html,
            legend_items=zipped_data,
            model_info=model_info,
            current_dasarian=dasarian,
            dasarian_options=dasarian_options
        )
    
    except Exception as e:
        logger.error(f"Error in rice phase view: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/api/classify-by-date', methods=['POST'])
@handle_ee_errors
def classify_by_date():
    """API endpoint untuk klasifikasi berdasarkan tanggal dengan dukungan custom collection"""
    try:
        data = request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        source_type = data.get('source_type', 'default')
        collection_asset = data.get('collection_asset')
        collection_type = data.get('collection_type', 'sentinel1')
        
        if not start_date or not end_date:
            return jsonify({'error': 'Start date and end date are required'}), 400
        
        logger.info(f"Classifying for {start_date} to {end_date}, source: {source_type}")
        
        # Cek model
        model = load_trained_model()
        if model is None:
            return jsonify({'error': 'Model tidak tersedia'}), 500
        
        if source_type == 'custom':
            if not collection_asset:
                return jsonify({'error': 'Collection asset harus diisi untuk data custom'}), 400
                
            logger.info(f"Using custom collection: {collection_asset} (type: {collection_type})")
            
            # Use custom collection
            classified_image, area_stats = classify_with_custom_collection(
                collection_asset, 
                collection_type,
                start_date=start_date, 
                end_date=end_date
            )
            data_source = f'Custom Collection ({collection_type})'
            
        else:
            # Use default Sentinel-1 real-time data
            logger.info("Using Sentinel-1 real-time data")
            classified_image = classify_with_date_filter_realtime(start_date, end_date)
            
            if classified_image is None:
                return jsonify({'error': 'Tidak ada data Sentinel-1 untuk periode tersebut'}), 404
            
            # Calculate area statistics for default collection
            area_stats = None
            try:
                area_stats = calculate_area_statistics(classified_image)
            except Exception as stats_error:
                logger.warning(f"Could not compute area stats: {str(stats_error)}")
                
            data_source = 'Sentinel-1 Real-time'
        
        # Create map with classification
        my_map = create_map(
            with_classification=True,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert to HTML
        map_html = my_map._repr_html_()
        
        return jsonify({
            'success': True,
            'map_html': map_html,
            'data_source': data_source,
            'date_range': f'{start_date} to {end_date}',
            'area_stats': area_stats,
            'message': f'Klasifikasi berhasil menggunakan {data_source} untuk periode {start_date} - {end_date}'
        })
        
    except Exception as e:
        logger.error(f"Error in date-based classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dasarian-info/<int:dasarian>')
def get_dasarian_info_api(dasarian):
    """API endpoint untuk mendapatkan informasi dasarian"""
    try:
        if dasarian < 1 or dasarian > 36:
            return jsonify({'error': 'Dasarian must be between 1 and 36'}), 400
        
        dasarian_info = get_dasarian_info(dasarian)
        start_date, end_date = dasarian_to_date_range(dasarian)
        
        dasarian_info.update({
            'start_date': start_date,
            'end_date': end_date
        })
        
        return jsonify({
            'success': True,
            'dasarian_info': dasarian_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_classification_map_dasarian', methods=['POST'])
@handle_ee_errors  
def get_classification_map_dasarian():
    """API endpoint untuk klasifikasi berdasarkan dasarian dengan dukungan custom collection"""
    try:
        data = request.get_json()
        dasarian = data.get('dasarian', 1)
        source_type = data.get('source_type', 'default')
        collection_asset = data.get('collection_asset')
        collection_type = data.get('collection_type', 'sentinel1')
        
        logger.info(f"Classifying for dasarian: {dasarian}, source: {source_type}")
        
        if source_type == 'custom':
            if not collection_asset:
                return jsonify({'error': 'Collection asset harus diisi untuk data custom'}), 400
                
            logger.info(f"Using custom collection: {collection_asset} (type: {collection_type})")
            
            # Use custom collection
            classified_image, area_stats = classify_with_custom_collection(
                collection_asset, 
                collection_type,
                dasarian_start=dasarian, 
                dasarian_end=dasarian
            )
            data_source = f'Custom Collection ({collection_type})'
            
        else:
            # Use default asset collection
            logger.info("Using default asset collection")
            classified_image = classify_with_dasarian_filter_asset(dasarian, dasarian)
            
            if classified_image is None:
                return jsonify({'error': 'Gagal klasifikasi dasarian'}), 500
                
            # Calculate area statistics for default collection
            area_stats = None
            try:
                area_stats = calculate_area_statistics(classified_image)
                if area_stats:
                    logger.info("Area statistics calculated successfully")
            except Exception as stats_error:
                logger.warning(f"Could not calculate area stats: {str(stats_error)}")
                
            data_source = 'Default Asset Collection'
        
        # Create map with classification
        dasarian_filter = (dasarian, dasarian)
        my_map = create_map(
            with_classification=True,
            dasarian_filter=dasarian_filter
        )
        
        # Convert to HTML
        map_html = my_map._repr_html_()
        
        dasarian_info = get_dasarian_info(dasarian)
        
        return jsonify({
            'success': True,
            'map_html': map_html,
            'data_source': data_source,
            'dasarian': dasarian,
            'area_stats': area_stats,
            'dasarian_info': dasarian_info,
            'message': f"Klasifikasi berhasil menggunakan {data_source} untuk {dasarian_info['display_name']}"
        })
        
    except Exception as e:
        logger.error(f"Error in dasarian classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
def model_info():
    """API endpoint untuk mendapatkan informasi model"""
    try:
        model = load_trained_model()
        if model is None:
            return jsonify({'error': 'Model tidak tersedia'}), 404
        
        info = {
            'model_type': str(type(model).__name__),
            'features': list(model.feature_names_in_),
            'classes': list(model.classes_),
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'model_path': config.MODEL_PATH,
            'bands_used': BANDS_SELECTED
        }
        
        return jsonify({
            'success': True,
            'model_info': info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-phase', methods=['POST'])
@handle_ee_errors
def analyze_phase():
    """API endpoint for temporal analysis of rice phases using trained model"""
    try:
        data = request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        dasarian = data.get('dasarian')
        
        logger.info(f"Analyzing phase with trained model")
        
        # Cek model
        model = load_trained_model()
        if model is None:
            return jsonify({'error': 'Model tidak tersedia'}), 500
        
        # Lakukan klasifikasi
        if start_date and end_date:
            classified_image = classify_with_date_filter_realtime(start_date, end_date)
            period_info = f'Tanggal {start_date} - {end_date}'
        elif dasarian:
            classified_image = classify_with_dasarian_filter_asset(dasarian, dasarian)
            dasarian_info = get_dasarian_info(dasarian)
            period_info = dasarian_info['full_name']
        else:
            classified_image = classify_with_dasarian_filter_asset(1, 36)
            period_info = 'Semua dasarian'
            
        if classified_image is None:
            return jsonify({'error': 'Gagal melakukan klasifikasi'}), 500
        
        # Get map tiles
        map_id = classified_image.getMapId(phase_vis_params)
        
        # Get statistics
        study_area = ee.Geometry.Rectangle([108.1549, -6.5153, 108.5549, -6.1153])
        stats = classified_image.reduceRegion(
            reducer=ee.Reducer.histogram(),
            geometry=study_area,
            scale=SCALE,
            maxPixels=1e9
        ).getInfo()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'period': period_info,
            'model_classes': list(model.classes_),
            'tile_url': map_id['tile_fetcher'].url_format
        })
        
    except Exception as e:
        logger.error(f"Error in phase analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_class_statistics', methods=['POST'])
@handle_ee_errors
def get_class_statistics():
    """API endpoint untuk mendapatkan statistik kelas yang detail dengan persentase"""
    try:
        data = request.get_json()
        dasarian = data.get('dasarian', 1)
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        use_realtime = data.get('use_realtime', False)
        
        logger.info(f"Getting class statistics for dasarian: {dasarian}, realtime: {use_realtime}")
        
        # Lakukan klasifikasi berdasarkan parameter
        if start_date and end_date:
            classified_image = classify_with_date_filter_realtime(start_date, end_date)
            period_info = f"Periode {start_date} sampai {end_date}"
            data_source = "Sentinel-1 Real-time"
        elif use_realtime:
            # Gunakan metode real-time dasarian (jika sudah diimplementasi)
            classified_image = classify_with_dasarian_filter_asset(dasarian, dasarian)
            dasarian_info = get_dasarian_info(dasarian)
            period_info = dasarian_info['full_name']
            data_source = "Real-time Dasarian Processing"
        else:
            # Gunakan collection asset
            classified_image = classify_with_dasarian_filter_asset(dasarian, dasarian)
            dasarian_info = get_dasarian_info(dasarian)
            period_info = dasarian_info['full_name']
            data_source = "Asset Collection"
        
        if classified_image is None:
            return jsonify({'error': 'Gagal melakukan klasifikasi'}), 500
        
        # Hitung statistik area
        area_stats = calculate_area_statistics(classified_image)
        
        if area_stats is None:
            return jsonify({'error': 'Gagal menghitung statistik area'}), 500
        
        # Format response
        response_data = {
            'success': True,
            'period_info': period_info,
            'data_source': data_source,
            'statistics': area_stats,
            'summary': {
                'total_classes_found': area_stats['number_of_classes'],
                'total_area_hectares': area_stats['total_area_hectares'],
                'largest_class': area_stats['class_distribution'][0] if area_stats['class_distribution'] else None,
                'dominant_phase': area_stats['class_distribution'][0]['class_name'] if area_stats['class_distribution'] else 'N/A'
            },
            'message': f"Statistik berhasil dihitung untuk {period_info} menggunakan {data_source}"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting class statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare_periods', methods=['POST'])
@handle_ee_errors
def compare_periods():
    """API endpoint untuk membandingkan statistik antar periode"""
    try:
        data = request.get_json()
        periods = data.get('periods', [])  # List of periods to compare
        
        if not periods or len(periods) < 2:
            return jsonify({'error': 'Minimal 2 periode diperlukan untuk perbandingan'}), 400
        
        comparison_results = []
        
        for period in periods:
            dasarian = period.get('dasarian')
            start_date = period.get('start_date')
            end_date = period.get('end_date')
            
            # Lakukan klasifikasi
            if start_date and end_date:
                classified_image = classify_with_date_filter_realtime(start_date, end_date)
                period_name = f"{start_date} to {end_date}"
            else:
                classified_image = classify_with_dasarian_filter_asset(dasarian, dasarian)
                dasarian_info = get_dasarian_info(dasarian)
                period_name = dasarian_info['display_name']
            
            if classified_image:
                stats = calculate_area_statistics(classified_image)
                if stats:
                    comparison_results.append({
                        'period_name': period_name,
                        'dasarian': dasarian,
                        'statistics': stats,
                        'dominant_class': stats['class_distribution'][0] if stats['class_distribution'] else None
                    })
        
        return jsonify({
            'success': True,
            'comparison_count': len(comparison_results),
            'periods': comparison_results,
            'message': f"Berhasil membandingkan {len(comparison_results)} periode"
        })
        
    except Exception as e:
        logger.error(f"Error in period comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-sentinel1-availability')
def check_availability():
    """API endpoint untuk cek ketersediaan data Sentinel-1"""
    try:
        aoi = get_indramayu_aoi()
        is_available = check_sentinel1_availability(aoi)
        
        return jsonify({
            'success': True,
            'data_available': is_available,
            'message': 'Data Sentinel-1 tersedia' if is_available else 'Data Sentinel-1 tidak tersedia di area ini'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/collection_info')
def collection_info():
    """API endpoint untuk mendapatkan informasi collection dengan indeks"""
    try:
        collection = ee.ImageCollection(config.COLLECTION_ASSET)
        collection_with_indices = collection.map(calculate_vegetation_indices).limit(3)
        
        # Dapatkan informasi dasar
        total_size = collection.size().getInfo()
        
        # Dapatkan sample properties
        sample_properties = collection_with_indices.first().getInfo().get('properties', {})
        
        return jsonify({
            'success': True,
            'total_size': total_size,
            'sample_properties': sample_properties
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate-custom-collection', methods=['POST'])
@handle_ee_errors
def validate_custom_collection():
    """Validate user's custom image collection"""
    try:
        data = request.get_json()
        collection_asset = data.get('collection_asset', '').strip()
        collection_type = data.get('collection_type', 'sentinel1')
        
        if not collection_asset:
            return jsonify({
                'success': False,
                'error': 'Path asset collection tidak boleh kosong'
            }), 400
        
        # Try to access the collection
        try:
            collection = ee.ImageCollection(collection_asset)
            
            # Get basic info
            image_count = collection.size().getInfo()
            
            if image_count == 0:
                return jsonify({
                    'success': False,
                    'error': 'Collection kosong atau tidak dapat diakses'
                }), 400
            
            # Get first image info
            first_image = collection.first()
            first_image_info = first_image.getInfo()
            
            # Get bands
            bands = first_image_info.get('bands', [])
            band_names = [band.get('id', 'unknown') for band in bands]
            
            # Get date range - using simple approach
            try:
                # Get first and last image dates
                first_img = collection.sort('system:time_start').first()
                last_img = collection.sort('system:time_start', False).first()
                
                first_date = first_img.get('system:time_start').getInfo()
                last_date = last_img.get('system:time_start').getInfo()
                
                min_date = datetime.fromtimestamp(first_date / 1000).strftime('%Y-%m-%d')
                max_date = datetime.fromtimestamp(last_date / 1000).strftime('%Y-%m-%d')
            except:
                # Fallback if date extraction fails
                min_date = "N/A"
                max_date = "N/A"
            
            # Validate bands based on collection type
            validation_result = validate_collection_bands(band_names, collection_type)
            
            if not validation_result['valid']:
                return jsonify({
                    'success': False,
                    'error': f'Collection tidak sesuai dengan tipe {collection_type}: {validation_result["message"]}'
                }), 400
            
            return jsonify({
                'success': True,
                'info': {
                    'image_count': image_count,
                    'bands': band_names,
                    'date_range': {
                        'start': min_date,
                        'end': max_date
                    },
                    'collection_type': collection_type,
                    'validation': validation_result
                }
            })
            
        except Exception as ee_error:
            logger.error(f"Error accessing collection {collection_asset}: {str(ee_error)}")
            return jsonify({
                'success': False,
                'error': f'Tidak dapat mengakses collection: {str(ee_error)}'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in validate_custom_collection: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Terjadi kesalahan: {str(e)}'
        }), 500

def validate_collection_bands(band_names, collection_type):
    """Validate that collection has required bands for the specified type"""
    required_bands = {
        'sentinel1': ['VV', 'VH'],
        'sentinel2': ['B2', 'B3', 'B4', 'B8'],
        'landsat': ['B2', 'B3', 'B4', 'B5'],
        'custom': []  # No specific requirement for custom
    }
    
    # Alternative band names that are also acceptable
    alternative_bands = {
        'sentinel1': {
            'VV': ['VV', 'VV_int', 'vv', 'VV_filtered'],
            'VH': ['VH', 'VH_int', 'vh', 'VH_filtered']
        },
        'sentinel2': {
            'B2': ['B2', 'blue', 'Blue'],
            'B3': ['B3', 'green', 'Green'], 
            'B4': ['B4', 'red', 'Red'],
            'B8': ['B8', 'nir', 'NIR']
        },
        'landsat': {
            'B2': ['B2', 'blue', 'Blue', 'SR_B2'],
            'B3': ['B3', 'green', 'Green', 'SR_B3'],
            'B4': ['B4', 'red', 'Red', 'SR_B4'],
            'B5': ['B5', 'nir', 'NIR', 'SR_B5']
        }
    }
    
    if collection_type not in required_bands:
        return {
            'valid': False,
            'message': f'Tipe collection tidak dikenali: {collection_type}'
        }
    
    required = required_bands[collection_type]
    
    if collection_type == 'custom':
        # For custom, just check if there are some bands
        return {
            'valid': len(band_names) > 0,
            'message': 'OK - Custom collection' if len(band_names) > 0 else 'Tidak ada bands ditemukan'
        }
    
    # Check if required bands or their alternatives are present
    missing_bands = []
    found_bands = []
    
    for required_band in required:
        alternatives = alternative_bands.get(collection_type, {}).get(required_band, [required_band])
        
        # Check if any alternative is present
        band_found = False
        for alt_band in alternatives:
            if alt_band in band_names:
                found_bands.append(f"{required_band}({alt_band})")
                band_found = True
                break
        
        if not band_found:
            missing_bands.append(required_band)
    
    if missing_bands:
        return {
            'valid': False,
            'message': f'Bands yang diperlukan tidak ditemukan: {", ".join(missing_bands)}. Tersedia: {", ".join(band_names)}'
        }
    
    return {
        'valid': True,
        'message': f'OK - Bands ditemukan: {", ".join(found_bands)}'
    }

def get_custom_collection_data(collection_asset, collection_type, start_date=None, end_date=None, aoi=None):
    """Get and process custom collection data"""
    try:
        collection = ee.ImageCollection(collection_asset)
        logger.info(f"Custom collection loaded: {collection_asset}")
        
        # Apply filters if provided
        if start_date and end_date:
            collection = collection.filterDate(start_date, end_date)
            logger.info(f"Filtered by date: {start_date} to {end_date}")
        if aoi:
            collection = collection.filterBounds(aoi)
            logger.info(f"Filtered by AOI")
        
        # Log jumlah image setelah filter
        try:
            count = collection.size().getInfo()
            logger.info(f"Custom collection image count after filter: {count}")
            if count == 0:
                logger.error("Custom collection kosong setelah filter!")
        except Exception as e:
            logger.error(f"Tidak bisa mendapatkan jumlah image: {str(e)}")
        
        # Process based on collection type
        if collection_type == 'sentinel1':
            # Check if collection already has processed bands
            try:
                first_img = collection.first()
                first_bands = first_img.bandNames().getInfo()
                logger.info(f"First image bands: {first_bands}")
                
                # Check if collection already has required indices
                has_indices = all(band in first_bands for band in ['VV_int', 'VH_int', 'RPI', 'API', 'NDPI', 'RVI'])
                
                if has_indices:
                    logger.info("Collection already has processed indices, skipping preprocessing")
                else:
                    # Apply Sentinel-1 specific processing with error handling
                    logger.info("Starting Sentinel-1 preprocessing pipeline")
                    collection = collection.map(remove_border_noise)
                    logger.info("Applied remove_border_noise")
                    collection = collection.map(terrain_correction)
                    logger.info("Applied terrain_correction")
                    collection = collection.map(speckle_filter)
                    logger.info("Applied speckle_filter")
                    collection = collection.map(calculate_vegetation_indices_s1)
                    logger.info("Applied calculate_vegetation_indices_s1")
            except Exception as proc_error:
                logger.error(f"Error in Sentinel-1 preprocessing: {str(proc_error)}")
                # Fallback: use collection as-is without preprocessing
                logger.warning("Using collection without preprocessing as fallback")
        elif collection_type in ['sentinel2', 'landsat']:
            # Apply optical processing
            try:
                collection = collection.map(calculate_vegetation_indices)
                logger.info("Applied optical indices pipeline")
            except Exception as proc_error:
                logger.error(f"Error in optical preprocessing: {str(proc_error)}")
                logger.warning("Using collection without preprocessing as fallback")
        elif collection_type == 'custom':
            # For custom, assume indices are already calculated or use as-is
            logger.info("No additional processing for custom type")
        
        return collection
        
    except Exception as e:
        logger.error(f"Error processing custom collection: {str(e)}")
        raise

def classify_with_custom_collection(collection_asset, collection_type, dasarian_start=None, dasarian_end=None, start_date=None, end_date=None):
    """Classify using custom collection"""
    try:
        # Get AOI
        aoi = get_indramayu_aoi()
        
        # Get custom collection data
        collection = get_custom_collection_data(
            collection_asset, 
            collection_type, 
            start_date, 
            end_date, 
            aoi
        )
        
        # Create composite
        if dasarian_start and dasarian_end:
            # For dasarian-based classification
            composite = collection.median()
        else:
            # For date-based classification
            composite = collection.median()
        logger.info("Composite (median) created from custom collection")
        
        # Ensure required bands for classification
        required_bands = ['VV_int', 'VH_int', 'RPI', 'API', 'NDPI', 'RVI', 'angle']
        
        # Check if composite has required bands - hanya panggil getInfo() sekali
        try:
            composite_bands = composite.bandNames().getInfo()
            logger.info(f"Bands in composite: {composite_bands}")
        except Exception as e:
            logger.error(f"Error getting band names: {str(e)}")
            # Fallback: assume collection is already processed if it's custom
            if collection_type == 'custom':
                composite_bands = required_bands  # Assume all required bands are present
            else:
                raise Exception(f"Tidak dapat mengakses informasi bands: {str(e)}")
        
        missing_bands = [band for band in required_bands if band not in composite_bands]
        
        if missing_bands:
            logger.warning(f"Missing bands for classification: {missing_bands}")
            # Try to calculate missing indices if possible
            if collection_type == 'sentinel1':
                composite = calculate_vegetation_indices_s1(composite)
                # Update composite_bands after processing
                try:
                    composite_bands = composite.bandNames().getInfo()
                    logger.info(f"Bands after reprocessing: {composite_bands}")
                except:
                    # If still fails, assume processing worked
                    composite_bands = required_bands
        
        # Create classifier
        classifier = create_classifier_from_trained_model()
        
        # Select bands for classification
        available_bands = [band for band in required_bands if band in composite_bands]
        logger.info(f"Bands used for classification: {available_bands}")
        
        if len(available_bands) < 3:  # Minimum bands required
            raise Exception(f"Tidak cukup bands untuk klasifikasi. Tersedia: {available_bands}")
        
        composite_selected = composite.select(available_bands)
        
        # Classify
        classified = composite_selected.classify(classifier)
        
        # Calculate statistics
        area_stats = calculate_area_statistics(classified)
        
        return classified, area_stats
        
    except Exception as e:
        logger.error(f"Error in classify_with_custom_collection: {str(e)}")
        raise

if __name__ == '__main__':
    # Menjalankan server di port yang diinginkan
    app.run(host='0.0.0.0', port=5000)