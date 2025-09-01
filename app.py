from pdf2image import convert_from_path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
from PIL import Image
import os
import json
from collections import defaultdict
from scipy import ndimage
import math
import logging
from logging.handlers import RotatingFileHandler
import secrets
import uuid
import base64
import platform
from io import BytesIO
import fitz 
import tempfile


# Only import portalocker and fcntl on non-Windows systems
if platform.system() != 'Windows':
    try:
        import portalocker
        import fcntl
    except ImportError:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://admin.neet720.com", "https://neet720.com", "http://localhost:3000", "http://localhost:3001"]}})

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create upload directory
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Setup logging for production
if not app.config['DEBUG']:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/omr_app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

    app.logger.info('OMR Application startup')


def convert_pdf_to_images(pdf_bytes, max_pages=500):
    """
    Convert PDF pages to images for OMR processing
    Returns list of image bytes for each page
    """
    if not PDF_SUPPORT:
        raise ValueError("PDF processing requires PyMuPDF. Install with: pip install PyMuPDF")
    
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Limit pages to maximum
        total_pages = len(pdf_document)
        pages_to_process = min(total_pages, max_pages)
        
        logger.info(f"Converting {pages_to_process} pages from PDF (total: {total_pages})")
        
        image_list = []
        
        for page_num in range(pages_to_process):
            try:
                # Get page
                page = pdf_document[page_num]
                
                # Convert to image with high DPI for better OMR detection
                mat = fitz.Matrix(2.0, 2.0)  # 2x scaling for 144 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                image_list.append(img_data)
                
                logger.info(f"Converted page {page_num + 1}/{pages_to_process}")
                
            except Exception as e:
                logger.error(f"Error converting page {page_num + 1}: {e}")
                continue
        
        pdf_document.close()
        
        if not image_list:
            raise ValueError("No pages could be converted from PDF")
        
        logger.info(f"Successfully converted {len(image_list)} pages from PDF")
        return image_list
        
    except Exception as e:
        logger.error(f"PDF conversion error: {e}")
        raise ValueError(f"Failed to convert PDF: {e}")

def detect_scanners():
    """
    Improved: Detect available scanners on the system (Windows TWAIN)
    Returns a list of scanner information, handles errors gracefully.
    """
    import platform
    scanners = []
    try:
        # Windows TWAIN scanner detection
        if platform.system() == 'Windows':
            try:
                import twain
                sm = twain.SourceManager(0)
                sources = sm.GetSourceList()
                if not sources:
                    logger.info("No TWAIN scanners detected.")
                    return []
                for idx, source_name in enumerate(sources):
                    scanner_info = {
                        'id': idx,
                        'name': source_name,
                        'driver': 'TWAIN',
                        'platform': 'Windows',
                        'status': 'Available',
                        'supported_formats': ['JPEG', 'PNG', 'TIFF', 'BMP'],
                        'max_resolution': 600,  # Default
                        'color_modes': ['Color', 'Grayscale', 'Black & White'],
                        'paper_sizes': ['A4', 'Letter', 'Legal', 'Custom']
                    }
                    try:
                        source = sm.OpenSource(source_name)
                        # Try to get the supported resolutions
                        try:
                            xres = source.GetCapability(twain.ICAP_XRESOLUTION)
                            if xres:
                                scanner_info['max_resolution'] = max(xres)
                        except Exception as e:
                            logger.warning(f"Could not get max resolution for {source_name}: {e}")

                        # Try to get supported color modes
                        try:
                            colormodes = source.GetCapability(twain.ICAP_PIXELTYPE)
                            color_mode_map = {0: 'Black & White', 1: 'Grayscale', 2: 'Color'}
                            if colormodes:
                                scanner_info['color_modes'] = [color_mode_map.get(mode, str(mode)) for mode in colormodes if mode in color_mode_map]
                        except Exception as e:
                            logger.warning(f"Could not get color modes for {source_name}: {e}")

                        # Try to get supported paper sizes
                        try:
                            sizes = source.GetCapability(twain.ICAP_SUPPORTEDSIZES)
                            paper_size_map = {
                                0: 'None', 1: 'A4', 2: 'A5', 3: 'B4', 4: 'B5',
                                5: 'US Letter', 6: 'US Legal', 7: 'A6', 8: 'C4', 9: 'C5', 10: 'C6'
                            }
                            if sizes:
                                scanner_info['paper_sizes'] = [paper_size_map.get(s, str(s)) for s in sizes]
                        except Exception as e:
                            logger.warning(f"Could not get paper sizes for {source_name}: {e}")
                        scanners.append(scanner_info)
                        source.RequestAcquire(0, 0)  # Close source properly
                    except Exception as e:
                        logger.warning(f"Could not open scanner {source_name}: {e}")
                        scanners.append({
                            'id': idx,
                            'name': source_name,
                            'driver': 'TWAIN',
                            'platform': 'Windows',
                            'status': 'Error',
                            'supported_formats': ['JPEG', 'PNG'],
                            'max_resolution': 300,
                            'color_modes': ['Color', 'Grayscale'],
                            'paper_sizes': ['A4', 'Letter']
                        })
            except ImportError:
                logger.error("TWAIN library not installed. Please install twain module for scanner detection on Windows.")
            except Exception as e:
                logger.error(f"TWAIN scanner detection failed: {e}")
        else:
            logger.info("Scanner detection is implemented for Windows (TWAIN) only in this function.")
    except Exception as ex:
        logger.error(f"General error in scanner detection: {ex}")

    return scanners


def scan_document(scanner_id, scan_settings=None):
    """
    Scan a document using the specified scanner
    Returns the scanned image as bytes
    """
    try:
        scanners = detect_scanners()
        
        if scanner_id >= len(scanners):
            raise ValueError(f"Invalid scanner ID: {scanner_id}")
        
        scanner = scanners[scanner_id]
        
        # Default scan settings
        default_settings = {
            'resolution': 300,
            'color_mode': 'Color',
            'paper_size': 'A4',
            'format': 'JPEG',
            'brightness': 50,
            'contrast': 50
        }
        
        if scan_settings:
            default_settings.update(scan_settings)
        
        logger.info(f"Scanning with {scanner['name']} using settings: {default_settings}")
        
        # Windows TWAIN scanning
        if platform.system() == 'Windows' and TWAIN_AVAILABLE and scanner['driver'] == 'TWAIN':
            try:
                sm = twain.SourceManager(0)
                source = sm.OpenSource(scanner['name'])
                
                # Set scan parameters
                source.SetCapability(twain.ICAP_XRESOLUTION, default_settings['resolution'])
                source.SetCapability(twain.ICAP_YRESOLUTION, default_settings['resolution'])
                
                # Set color mode
                if default_settings['color_mode'] == 'Color':
                    source.SetCapability(twain.ICAP_PIXELTYPE, twain.TWPT_RGB)
                elif default_settings['color_mode'] == 'Grayscale':
                    source.SetCapability(twain.ICAP_PIXELTYPE, twain.TWPT_GRAY)
                else:
                    source.SetCapability(twain.ICAP_PIXELTYPE, twain.TWPT_BW)
                
                # Perform scan
                source.RequestAcquire(0, 0)
                rv = source.XferImageNatively()
                
                if rv:
                    (handle, count) = rv
                    # Convert handle to image bytes
                    # This is a simplified version - actual implementation depends on TWAIN library
                    image_data = twain.DIBToBMFile(handle)
                    twain.GlobalFree(handle)
                    return image_data
                
            except Exception as e:
                logger.error(f"TWAIN scanning failed: {e}")
                raise ValueError(f"Scanning failed: {e}")
        
        # Linux SANE scanning
        if platform.system() == 'Linux' and SANE_AVAILABLE and scanner['driver'] == 'SANE':
            try:
                sane.init()
                device = sane.open(scanner['device_name'])
                
                # Set scan parameters
                device.resolution = default_settings['resolution']
                
                if default_settings['color_mode'] == 'Color':
                    device.mode = 'color'
                elif default_settings['color_mode'] == 'Grayscale':
                    device.mode = 'gray'
                else:
                    device.mode = 'lineart'
                
                # Perform scan
                device.start()
                image_data = device.snap()
                device.close()
                sane.exit()
                
                # Convert PIL image to bytes
                if isinstance(image_data, Image.Image):
                    img_byte_arr = io.BytesIO()
                    image_data.save(img_byte_arr, format=default_settings['format'])
                    return img_byte_arr.getvalue()
                
            except Exception as e:
                logger.error(f"SANE scanning failed: {e}")
                raise ValueError(f"Scanning failed: {e}")
        
        # If no specific driver worked, return error
        raise ValueError(f"Scanning not supported for {scanner['driver']} on {platform.system()}")
        
    except Exception as e:
        logger.error(f"Document scanning failed: {e}")
        raise ValueError(f"Document scanning failed: {e}")



def enhance_qr_image(img):
    """Enhanced QR code preprocessing for better detection."""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply multiple enhancement techniques
    enhanced_images = []
    
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_enhanced = clahe.apply(gray)
    enhanced_images.append(cv2.cvtColor(clahe_enhanced, cv2.COLOR_GRAY2BGR))
    
    # 2. Gaussian blur + sharpening
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpening_kernel = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)
    enhanced_images.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))
    
    # 3. Morphological operations to clean up
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    enhanced_images.append(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR))
    
    # 4. Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    enhanced_images.append(cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR))
    
    # 5. Simple thresholding with Otsu
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_images.append(cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2BGR))
    
    # 6. Histogram equalization
    hist_eq = cv2.equalizeHist(gray)
    enhanced_images.append(cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR))
    
    # 7. Original with noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    enhanced_images.append(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR))
    
    return enhanced_images

def get_qr_regions(img):
    """Get multiple potential QR code regions from the image."""
    height, width = img.shape[:2]
    regions = []
    
    # Standard positions for OMR sheets
    positions = [
        # Top-right corner (most common)
        (int(width * 0.75), 0, width, int(height * 0.18)),
        (int(width * 0.70), 0, width, int(height * 0.20)),
        (int(width * 0.80), 0, width, int(height * 0.15)),
        (int(width * 0.72), int(height * 0.01), width, int(height * 0.17)),
        
        # Top-left corner
        (0, 0, int(width * 0.25), int(height * 0.18)),
        (0, 0, int(width * 0.30), int(height * 0.20)),
        
        # Bottom-right corner
        (int(width * 0.75), int(height * 0.82), width, height),
        (int(width * 0.70), int(height * 0.80), width, height),
        
        # Bottom-left corner
        (0, int(height * 0.82), int(width * 0.25), height),
        (0, int(height * 0.80), int(width * 0.30), height),
        
        # Center regions (if QR is placed unusually)
        (int(width * 0.35), int(height * 0.35), int(width * 0.65), int(height * 0.65)),
        
        # Header center
        (int(width * 0.35), 0, int(width * 0.65), int(height * 0.15)),
        
        # Full width header (for very wide QR placements)
        (0, 0, width, int(height * 0.12)),
        (0, 0, width, int(height * 0.18)),
    ]
    
    for (x1, y1, x2, y2) in positions:
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        if x2 > x1 and y2 > y1:  # Valid region
            region = img[y1:y2, x1:x2]
            if region.size > 0:  # Non-empty region
                regions.append(region)
    
    return regions


def scan_document_omr(scanner_id, scan_settings=None):
    """
    Scan a document using the specified scanner
    Returns the scanned image as bytes
    """
    try:
        scanners = detect_scanners()
        
        if scanner_id >= len(scanners):
            raise ValueError(f"Invalid scanner ID: {scanner_id}")
        
        scanner = scanners[scanner_id]
        
        # Default scan settings
        default_settings = {
            'resolution': 300,
            'color_mode': 'Color',
            'paper_size': 'A4',
            'format': 'JPEG',
            'brightness': 50,
            'contrast': 50
        }
        
        if scan_settings:
            default_settings.update(scan_settings)
        
        logger.info(f"Scanning with {scanner['name']} using settings: {default_settings}")
        
        # Windows TWAIN scanning
        if platform.system() == 'Windows' and TWAIN_AVAILABLE and scanner['driver'] == 'TWAIN':
            try:
                sm = twain.SourceManager(0)
                source = sm.OpenSource(scanner['name'])
                
                # Set scan parameters
                source.SetCapability(twain.ICAP_XRESOLUTION, default_settings['resolution'])
                source.SetCapability(twain.ICAP_YRESOLUTION, default_settings['resolution'])
                
                # Set color mode
                if default_settings['color_mode'] == 'Color':
                    source.SetCapability(twain.ICAP_PIXELTYPE, twain.TWPT_RGB)
                elif default_settings['color_mode'] == 'Grayscale':
                    source.SetCapability(twain.ICAP_PIXELTYPE, twain.TWPT_GRAY)
                else:
                    source.SetCapability(twain.ICAP_PIXELTYPE, twain.TWPT_BW)
                
                # Perform scan
                source.RequestAcquire(0, 0)
                rv = source.XferImageNatively()
                
                if rv:
                    (handle, count) = rv
                    # Convert handle to image bytes
                    # This is a simplified version - actual implementation depends on TWAIN library
                    image_data = twain.DIBToBMFile(handle)
                    twain.GlobalFree(handle)
                    return image_data
                
            except Exception as e:
                logger.error(f"TWAIN scanning failed: {e}")
                raise ValueError(f"Scanning failed: {e}")
        
        # Linux SANE scanning
        if platform.system() == 'Linux' and SANE_AVAILABLE and scanner['driver'] == 'SANE':
            try:
                sane.init()
                device = san00000000000000000000000001ze.open(scanner['device_name'])
                
                # Set scan parameters
                device.resolution = default_settings['resolution']
                
                if default_settings['color_mode'] == 'Color':
                    device.mode = 'color'
                elif default_settings['color_mode'] == 'Grayscale':
                    device.mode = 'gray'
                else:
                    device.mode = 'lineart'
                
                # Perform scan
                device.start()
                image_data = device.snap()
                device.close()
                sane.exit()
                
                # Convert PIL image to bytes
                if isinstance(image_data, Image.Image):
                    img_byte_arr = io.BytesIO()
                    image_data.save(img_byte_arr, format=default_settings['format'])
                    return img_byte_arr.getvalue()
                
            except Exception as e:
                logger.error(f"SANE scanning failed: {e}")
                raise ValueError(f"Scanning failed: {e}")
        
        # If no specific driver worked, return error
        raise ValueError(f"Scanning not supported for {scanner['driver']} on {platform.system()}")
        
    except Exception as e:
        logger.error(f"Document scanning failed: {e}")
        raise ValueError(f"Document scanning failed: {e}")

def detect_and_correct_skew(image):
    """
    Advanced skew detection and correction for any type of scanned image
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Method 1: Advanced Hough Line Transform
        def hough_skew_detection(img):
            # Apply multiple preprocessing techniques
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(blurred)
            
            # Apply morphological operations to enhance lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # Multiple edge detection approaches
            edges1 = cv2.Canny(morph, 50, 150, apertureSize=3)
            edges2 = cv2.Canny(morph, 30, 100, apertureSize=3)
            edges3 = cv2.Canny(morph, 100, 200, apertureSize=3)
            edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
            
            # Detect lines with different parameters
            line_params = [
                (1, np.pi/180, 100, int(width*0.3), 20),
                (1, np.pi/180, 80, int(width*0.25), 15),
                (1, np.pi/180, 60, int(width*0.2), 10),
            ]
            
            all_angles = []
            for rho, theta, threshold, min_line_length, max_line_gap in line_params:
                lines = cv2.HoughLinesP(edges, rho, theta, threshold, 
                                       minLineLength=min_line_length, maxLineGap=max_line_gap)
                
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # Only consider long horizontal lines
                        if abs(x2 - x1) > width * 0.2:
                            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                            # Consider lines that are nearly horizontal
                            if -15 <= angle <= 15:
                                all_angles.append(angle)
            
            return all_angles
        
        # Method 2: Projection Profile Analysis
        def projection_profile_skew(img):
            # Apply threshold
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Test different angles
            angles = np.arange(-15, 16, 0.5)
            variances = []
            
            for angle in angles:
                # Rotate image
                center = (width // 2, height // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(thresh, M, (width, height), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT, 
                                        borderValue=255)
                
                # Calculate horizontal projection
                horizontal_proj = np.sum(rotated, axis=1)
                
                # Calculate variance (higher variance = better text line alignment)
                variance = np.var(horizontal_proj)
                variances.append(variance)
            
            # Find angle with maximum variance
            best_angle_idx = np.argmax(variances)
            best_angle = angles[best_angle_idx]
            
            return [best_angle]
        
        # Method 3: Contour-based skew detection
        def contour_skew_detection(img):
            # Apply adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find text-like contours
            text_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 2000:  # Reasonable size for text/bubbles
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                        text_contours.append(contour)
            
            # Calculate angles from contours
            angles = []
            for contour in text_contours:
                if len(contour) >= 5:  # Need at least 5 points for fitEllipse
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        angle = ellipse[2]  # Angle of the ellipse
                        
                        # Convert angle to standard format
                        if angle > 90:
                            angle = angle - 180
                        elif angle < -90:
                            angle = angle + 180
                        
                        if -15 <= angle <= 15:
                            angles.append(angle)
                    except:
                        continue
            
            return angles
        
        # Apply all three methods
        all_detected_angles = []
        
        # Method 1: Hough lines
        hough_angles = hough_skew_detection(gray)
        if hough_angles:
            all_detected_angles.extend(hough_angles)
            logger.info(f"Hough method found {len(hough_angles)} angle measurements")
        
        # Method 2: Projection profile
        try:
            proj_angles = projection_profile_skew(gray)
            if proj_angles:
                all_detected_angles.extend(proj_angles)
                logger.info(f"Projection method found angle: {proj_angles[0]:.2f}°")
        except Exception as e:
            logger.warning(f"Projection profile method failed: {e}")
        
        # Method 3: Contour analysis
        try:
            contour_angles = contour_skew_detection(gray)
            if contour_angles:
                all_detected_angles.extend(contour_angles)
                logger.info(f"Contour method found {len(contour_angles)} angle measurements")
        except Exception as e:
            logger.warning(f"Contour analysis method failed: {e}")
        
        if not all_detected_angles:
            logger.info("No skew detected - image appears to be straight")
            return image
        
        # Use robust statistics to find the best angle
        angles_array = np.array(all_detected_angles)
        
        # Remove outliers using IQR method
        Q1 = np.percentile(angles_array, 25)
        Q3 = np.percentile(angles_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_angles = angles_array[(angles_array >= lower_bound) & (angles_array <= upper_bound)]
        
        if len(filtered_angles) == 0:
            filtered_angles = angles_array
        
        # Use median for robust estimation
        final_angle = np.median(filtered_angles)
        
        logger.info(f"Final skew angle detected: {final_angle:.2f}° (from {len(all_detected_angles)} measurements)")
        
        # Apply correction if angle is significant
        if abs(final_angle) > 0.1:
            logger.info(f"Applying skew correction: {final_angle:.2f}°")
            
            # Calculate rotation matrix
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
            
            # Calculate new image dimensions to prevent cropping
            cos_angle = np.abs(M[0, 0])
            sin_angle = np.abs(M[0, 1])
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            # Adjust translation to center the image
            M[0, 2] += (new_width / 2) - center[0]
            M[1, 2] += (new_height / 2) - center[1]
            
            # Apply rotation with white background
            corrected = cv2.warpAffine(image, M, (new_width, new_height), 
                                      flags=cv2.INTER_CUBIC, 
                                      borderMode=cv2.BORDER_CONSTANT, 
                                      borderValue=(255, 255, 255))
            
            logger.info(f"Image corrected: {width}x{height} → {new_width}x{new_height}")
            return corrected
        else:
            logger.info("Skew angle too small - no correction needed")
            return image
    
    except Exception as e:
        logger.error(f"Error in skew detection: {e}")
        return image

def enhanced_bubble_detection(image):
    """
    Enhanced bubble detection to ensure all 180 questions are found,
    now with QR region filter to ignore circles inside QR code area.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        all_circles = []
        
        # Try multiple HoughCircles parameter combinations
        param_sets = [
            {'dp': 1, 'minDist': 20, 'param1': 50, 'param2': 30, 'minRadius': 8, 'maxRadius': 25},
            {'dp': 1.2, 'minDist': 15, 'param1': 40, 'param2': 25, 'minRadius': 6, 'maxRadius': 22},
            {'dp': 1.5, 'minDist': 25, 'param1': 60, 'param2': 35, 'minRadius': 10, 'maxRadius': 20},
        ]
        
        for params in param_sets:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                **params
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for x, y, r in circles:
                    all_circles.append((x, y, r))
        
        # Remove duplicate circles
        unique_circles = []
        for circle in all_circles:
            x, y, r = circle
            is_duplicate = False
            for existing in unique_circles:
                ex, ey, er = existing
                distance = np.sqrt((x - ex)**2 + (y - ey)**2)
                if distance < 15:  # Merge nearby circles
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_circles.append(circle)

        # === QR FILTER LOGIC STARTS HERE ===
        # Define QR region (top-right box). Adjust if your QR moves/size changes.
        height, width = image.shape[:2]
        qr_box_width = 230  # width of QR area in px (tune as needed)
        qr_box_height = 230 # height of QR area in px (tune as needed)
        qr_left = width - qr_box_width
        qr_top = 0
        qr_right = width
        qr_bottom = qr_box_height

        filtered_circles = []
        for x, y, r in unique_circles:
            # Keep if outside QR area
            if not (qr_left <= x <= qr_right and qr_top <= y <= qr_bottom):
                filtered_circles.append((x, y, r))
            # else: print(f"Filtered: {x},{y} (possible QR bubble)")
        logger.info(f"Enhanced detection found {len(filtered_circles)} unique circles (after QR filter)")
        return filtered_circles
    
    except Exception as e:
        logger.error(f"Error in bubble detection: {e}")
        return []

def organize_questions_1_to_180(centers, image_shape):
    """
    Organize bubbles into questions 1-180 with proper numbering
    """
    if len(centers) < 150:
        raise ValueError(f"Insufficient bubbles detected: {len(centers)}. Expected ~180 questions.")
    
    # Sort centers by x-coordinate to identify columns
    sorted_centers = sorted(centers, key=lambda c: c[0])
    
    # Find column boundaries by analyzing x-coordinate gaps
    x_coords = [c[0] for c in sorted_centers]
    
    # Calculate gaps between consecutive x-coordinates
    gaps = []
    for i in range(1, len(x_coords)):
        gap = x_coords[i] - x_coords[i-1]
        gaps.append((gap, i))
    
    # Sort gaps by size and find the 3 largest gaps (for 4 columns)
    gaps.sort(reverse=True)
    column_breaks = sorted([gap[1] for gap in gaps[:3]])
    
    # Create 4 columns
    columns = [[] for _ in range(4)]
    
    # Assign bubbles to columns
    start_idx = 0
    for col_idx, break_idx in enumerate(column_breaks):
        for i in range(start_idx, break_idx):
            columns[col_idx].append(sorted_centers[i])
        start_idx = break_idx
    
    # Add remaining bubbles to last column
    for i in range(start_idx, len(sorted_centers)):
        columns[3].append(sorted_centers[i])
    
    logger.info(f"Column distribution: {[len(col) for col in columns]}")
    
    # Process each column to create questions 1-180
    all_questions = []
    
    for col_idx, column_bubbles in enumerate(columns):
        if len(column_bubbles) < 20:
            continue
        
        # Sort by y-coordinate within each column
        column_bubbles.sort(key=lambda c: c[1])
        
        # Group bubbles into rows (each question should have 4 options)
        rows = []
        i = 0
        
        while i < len(column_bubbles):
            current_y = column_bubbles[i][1]
            
            # Find all bubbles in the same horizontal row
            row_bubbles = []
            j = i
            while j < len(column_bubbles) and abs(column_bubbles[j][1] - current_y) <= 20:
                row_bubbles.append(column_bubbles[j])
                j += 1
            
            # If we have exactly 4 bubbles, it's a valid question
            if len(row_bubbles) == 4:
                # Sort by x-coordinate to get A, B, C, D order
                row_bubbles.sort(key=lambda c: c[0])
                rows.append([(c[0], c[1]) for c in row_bubbles])
            
            i = j if j > i else i + 1
        
        # Each column should have 45 questions (180/4 = 45)
        rows = rows[:45]
        
        # Add questions with proper numbering: 1-45, 46-90, 91-135, 136-180
        start_question = col_idx * 45 + 1
        
        for row_idx, row_bubbles in enumerate(rows):
            question_number = start_question + row_idx
            if question_number <= 180:
                all_questions.append((question_number, row_bubbles))
    
    # Sort by question number to ensure proper order
    all_questions.sort(key=lambda x: x[0])
    
    logger.info(f"Successfully organized {len(all_questions)} questions")
    return all_questions

def is_bubble_filled(image, center, radius=12):
    """
    Check if a bubble is filled by analyzing pixel intensity
    """
    try:
        x, y = center
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ensure coordinates are within image bounds
        h, w = gray.shape
        if x < radius or y < radius or x >= w - radius or y >= h - radius:
            return False
        
        # Create a circular mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius-2, 255, -1)
        
        # Calculate mean intensity within the bubble
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        
        # Calculate background intensity (around the bubble)
        outer_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), radius+5, 255, -1)
        cv2.circle(outer_mask, (x, y), radius, 0, -1)
        background_intensity = cv2.mean(gray, mask=outer_mask)[0]
        
        # A bubble is filled if it's significantly darker than the background
        intensity_diff = background_intensity - mean_intensity
        
        return intensity_diff > 20 and mean_intensity < 150
    
    except Exception as e:
        logger.error(f"Error checking bubble fill: {e}")
        return False

def process_omr_enhanced(image_bytes):
    """
    Process OMR with skew correction and enhanced bubble detection
    """
    try:
        # Load image
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Cannot decode image")
        
        logger.info(f"Original image shape: {image.shape}")
        
        # STEP 1: Apply skew correction
        corrected_image = detect_and_correct_skew(image)
        logger.info("Skew correction applied")
        
        # STEP 2: Enhanced bubble detection
        bubble_centers = enhanced_bubble_detection(corrected_image)
        logger.info(f"Total bubbles detected: {len(bubble_centers)}")
        
        if len(bubble_centers) < 120:
            raise ValueError(f"Insufficient bubbles detected: {len(bubble_centers)}. Expected ~180 questions.")
        
        # STEP 3: Organize bubbles into questions 1-180
        organized_questions = organize_questions_1_to_180(bubble_centers, corrected_image.shape)
        logger.info(f"Questions organized: {len(organized_questions)}")
        
        # STEP 4: Process each question and detect filled bubbles
        output_img = corrected_image.copy()
        results = []
        
        for question_number, question_bubbles in organized_questions:
            if len(question_bubbles) != 4:
                continue
            
            for option_idx, (cx, cy) in enumerate(question_bubbles):
                # Find the actual radius for this bubble
                radius = 12
                for bx, by, br in bubble_centers:
                    if abs(bx - cx) < 5 and abs(by - cy) < 5:
                        radius = br
                        break
                
                # Check if bubble is filled
                is_filled = is_bubble_filled(corrected_image, (cx, cy), radius)
                
                # Draw visualization
                color = (0, 255, 0) if is_filled else (0, 0, 255)
                cv2.circle(output_img, (cx, cy), radius, color, 2)
                
                if is_filled:
                    cv2.circle(output_img, (cx, cy), radius-3, (0, 255, 0), -1)
                
                # Add labels
                option_letter = chr(65 + option_idx)  # A, B, C, D
                cv2.putText(output_img, f"{question_number}{option_letter}", 
                           (cx-15, cy-radius-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Store results for filled bubbles
                if is_filled:
                    results.append({
                        "question": int(question_number),
                        "option": option_letter,
                        "marked": 1,
                        "coordinates": [int(cx), int(cy)]
                    })
        
        # Convert output image to bytes
        output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(output_img_rgb)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return results, img_byte_arr, len(organized_questions)
        
    except Exception as e:
        logger.error(f"Error processing OMR: {str(e)}")
        raise ValueError(f"Error processing OMR: {str(e)}")

# API Routes
# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'healthy', 'service': 'OMR Processing API'}), 200


@app.route('/api/scanners', methods=['GET'])
def list_scanners():
    try:
        scanners = detect_scanners()
        return jsonify({'success': True, 'scanners': scanners}), 200
    except Exception as e:
        logger.error(f"Error detecting scanners: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/scan_qr', methods=['POST'])
def scan_qr():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if the file is a PDF
        if file.filename.lower().endswith('.pdf'):
            # Save PDF to temporary file first
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file.read())
                temp_file.flush()
                
                from pdf2image import convert_from_path
                images = convert_from_path(temp_file.name)
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            # Process each page of the PDF (each page is treated as a separate image)
            all_qr_results = []
            for page_idx, image in enumerate(images):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

                # Process QR code on this page
                results = process_qr_on_image(img_bytes)
                all_qr_results.append({
                    'page': page_idx + 1,
                    'qr_data': results['qr_data'],
                    'detection_info': results['detection_info'],
                    'confidence': results['confidence']
                })

            return jsonify({
                'success': True,
                'results': all_qr_results,
                'message': 'QR codes processed successfully from PDF'
            }), 200
        
        # If it's not a PDF, process it as an image file
        else:
            # Read file bytes for image processing
            file.seek(0)  # Reset file pointer since we might have read it above
            file_bytes = file.read()
            
            # Convert to numpy array for cv2
            file_bytes_np = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({'success': False, 'error': 'Failed to decode image'}), 400

            # Process QR code on the image
            results = process_qr_on_image(file_bytes)

            return jsonify({
                'success': True,
                'qr_data': results['qr_data'],
                'detection_info': results['detection_info'],
                'confidence': results['confidence']
            }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': f'Processing error: {str(e)}'}), 500

def process_qr_on_image(image_bytes):
    # QR processing logic for individual images (QR detection, enhancements)
    detector = cv2.QRCodeDetector()
    qr_data = None
    best_confidence = 0
    detection_info = {
        'angle': 0,
        'scale': 1.0,
        'region': 'unknown',
        'enhancement': 'none'
    }

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {'qr_data': None, 'detection_info': detection_info, 'confidence': 0}

    # Process QR code as done in the scan_qr function (scales, enhancements, angles)
    qr_regions = get_qr_regions(img)
    qr_regions.append(img)
    scales = [1.0, 1.5, 2.0, 2.5, 3.0, 0.8, 1.2]
    angles = [0, -1, 1, -2, 2, -3, 3, -5, 5, -7, 7, -10, 10, -15, 15, 90, 180, 270]

    # Try different scales, angles, etc.
    for region in qr_regions:
        for scale in scales:
            if scale != 1.0:
                region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            enhanced_images = enhance_qr_image(region)
            for enhanced in enhanced_images:
                for angle in angles:
                    if angle != 0:
                        center = (enhanced.shape[1] // 2, enhanced.shape[0] // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(enhanced, M, (enhanced.shape[1], enhanced.shape[0]), borderMode=cv2.BORDER_CONSTANT)
                    else:
                        rotated = enhanced.copy()

                    data, bbox, straight_qrcode = detector.detectAndDecode(rotated)
                    if data:
                        confidence = cv2.contourArea(bbox) if bbox is not None else len(data)
                        if confidence > best_confidence:
                            qr_data = data.strip()
                            best_confidence = confidence
                            detection_info = {'angle': angle, 'scale': scale}

    return {'qr_data': qr_data, 'detection_info': detection_info, 'confidence': best_confidence}

@app.route('/api/process-omr-key', methods=['POST'])
def process_omr_key():
    """
    API endpoint for answer key OMR (the correct answers)
    """
    try:
        if 'omrfile' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['omrfile']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload an image file or PDF.'}), 400
        
        # Read file bytes first
        file_bytes = file.read()
        
        # Check if the file is a PDF
        if file.filename.lower().endswith('.pdf'):
            # Convert PDF to images using PyMuPDF
            image_list = convert_pdf_to_images(file_bytes, max_pages=500)
            
            all_results = []
            all_processed_images = []
            
            # Process each page
            for page_idx, image_bytes in enumerate(image_list):
                try:
                    results, img_bytes, total_questions_processed = process_omr_enhanced(image_bytes)
                    
                    # Build answer key dict: {question_number: correct_option}
                    answer_key = {}
                    for r in results:
                        q = r['question']
                        if r['marked']:
                            answer_key[q] = r['option']
                    
                    all_results.append({
                        'page': page_idx + 1,
                        'answer_key': answer_key,
                        'raw_results': results,
                        'total_questions_processed': total_questions_processed
                    })
                    
                    processed_image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    all_processed_images.append(processed_image_base64)
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_idx + 1}: {e}")
                    continue
            
            return jsonify({
                'success': True,
                'results': all_results,
                'processed_images': all_processed_images,
                'total_pages_processed': len(all_results),
                'message': 'OMR answer key processed successfully from PDF'
            }), 200
        
        # If it's an image file, process it as usual
        else:
            results, img_bytes, total_questions_processed = process_omr_enhanced(file_bytes)

            # Build answer key dict: {question_number: correct_option}
            answer_key = {}
            for r in results:
                q = r['question']
                if r['marked']:
                    answer_key[q] = r['option']

            processed_image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return jsonify({
                'success': True,
                'answer_key': answer_key,
                'raw_results': results,
                'questions_detected': total_questions_processed,
                'processed_image': processed_image_base64,
                'message': 'Answer key OMR processed successfully'
            }), 200

    except Exception as e:
        logger.error(f"Error processing answer key OMR: {str(e)}")
        return jsonify({'success': False, 'error': str(e), 'message': 'Failed to process answer key OMR'}), 500

@app.route('/api/process-omr-sheet', methods=['POST'])
def process_omr_sheet():
    """
    API endpoint for student/question OMR (the responses to be checked)
    """
    try:
        if 'omrfile' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['omrfile']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload an image file or PDF.'}), 400
        
        # Read file bytes first
        file_bytes = file.read()
        
        # Check if the file is a PDF
        if file.filename.lower().endswith('.pdf'):
            # Convert PDF to images using PyMuPDF
            image_list = convert_pdf_to_images(file_bytes, max_pages=500)
            
            all_results = []
            all_processed_images = []
            
            # Process each page
            for page_idx, image_bytes in enumerate(image_list):
                try:
                    results, img_bytes, total_questions_processed = process_omr_enhanced(image_bytes)
                    responses = {r['question']: r['option'] for r in results if r['marked']}
                    all_results.append({
                        'page': page_idx + 1,
                        'responses': responses,
                        'total_questions_processed': total_questions_processed
                    })
                    
                    processed_image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    all_processed_images.append(processed_image_base64)
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_idx + 1}: {e}")
                    continue
            
            return jsonify({
                'success': True,
                'results': all_results,
                'processed_images': all_processed_images,
                'total_pages_processed': len(all_results),
                'message': 'OMR sheet processed successfully from PDF'
            }), 200
        
        # If it's an image file, process it as usual
        else:
            results, img_bytes, total_questions_processed = process_omr_enhanced(file_bytes)

            # Build response dict: {question_number: selected_option}
            responses = {r['question']: r['option'] for r in results if r['marked']}
            processed_image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return jsonify({
                'success': True,
                'responses': responses,
                'raw_results': results,
                'questions_detected': total_questions_processed,
                'processed_image': processed_image_base64,
                'message': 'Student/question OMR processed successfully'
            }), 200

    except Exception as e:
        logger.error(f"Error processing student/question OMR: {str(e)}")
        return jsonify({'success': False, 'error': str(e), 'message': 'Failed to process student/question OMR'}), 500



@app.route('/api/process-omr', methods=['POST'])
def process_omr_api():
    """
    API endpoint to process OMR sheets
    Expects: multipart/form-data with 'omrfile' field (could be a PDF or image)
    Returns: JSON with results, summary, and base64 encoded processed images for all sheets
    """
    try:
        # Check if file is in request
        if 'omrfile' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['omrfile']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the file bytes
        file_bytes = file.read()

        # If the file is a PDF, convert it to images
        if file.filename.lower().endswith('.pdf'):
            # Convert the PDF to images in-memory
            from pdf2image import convert_from_path
            images = convert_from_path(BytesIO(file_bytes))  # Using BytesIO to read in-memory file
            all_results = []
            all_processed_images = []
            
            # Process each page of the PDF (each OMR sheet)
            for page_idx, image in enumerate(images):
                # Save the page as bytes for further processing
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()
                
                # Process the OMR sheet (bubble detection, skew correction, etc.)
                results, img_bytes, total_questions_processed = process_omr_enhanced(image_bytes)
                
                # Append the results for each sheet
                all_results.append({
                    'page': page_idx + 1,
                    'results': results,
                    'total_questions_processed': total_questions_processed
                })
                
                # Collect base64 encoded processed image
                processed_image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                all_processed_images.append(processed_image_base64)
            
            # Return results for all pages (OMR sheets)
            return jsonify({
                'success': True,
                'results': all_results,
                'processed_images': all_processed_images,
                'message': 'OMR sheets processed successfully from PDF'
            }), 200
        
        # If it's an image file, process it as usual (same as before)
        else:
            results, img_bytes, total_questions_processed = process_omr_enhanced(file_bytes)
            
            # Generate summary
            marked_bubbles = len(results)
            questions_answered = len(set(r['question'] for r in results))
            expected_questions = 180
            
            completion_rate = (questions_answered / expected_questions) * 100
            detection_rate = (total_questions_processed / expected_questions) * 100
            
            summary = {
                'total_questions_processed': int(total_questions_processed),
                'marked_bubbles': int(marked_bubbles),
                'questions_answered': int(questions_answered),
                'expected_questions': expected_questions,
                'unanswered_questions': int(expected_questions - questions_answered),
                'completion_rate': f"{completion_rate:.1f}%",
                'detection_rate': f"{detection_rate:.1f}%",
                'processing_method': "Enhanced Detection with Skew Correction",
                'accuracy_score': f"{min(100, detection_rate):.1f}%"
            }
            
            # Convert processed image to base64
            processed_image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            logger.info(f"Successfully processed OMR API request: {summary}")
            
            return jsonify({
                'success': True,
                'results': results,
                'summary': summary,
                'processed_image': processed_image_base64,
                'message': 'OMR sheet processed successfully'
            }), 200

    except Exception as e:
        logger.error(f"Error processing OMR API request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process OMR sheet'
        }), 500

@app.route('/api/process-omr-base64', methods=['POST'])
def process_omr_base64():
    """
    API endpoint to process OMR sheets from base64 encoded images
    Expects: JSON with 'image' field containing base64 encoded image
    Returns: JSON with results, summary, and base64 encoded processed image
    """
    try:
        # Check if JSON data is provided
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        try:
            image_data = data['image']
            # Remove data URL prefix if present
            if 'data:' in image_data and ';base64,' in image_data:
                image_data = image_data.split(';base64,')[1]
            
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'error': 'Invalid base64 image data'}), 400
        
        # Process with enhanced detection
        results, img_bytes, total_questions_processed = process_omr_enhanced(image_bytes)
        
        # Generate summary
        marked_bubbles = len(results)
        questions_answered = len(set(r['question'] for r in results))
        expected_questions = 180
        
        completion_rate = (questions_answered / expected_questions) * 100
        detection_rate = (total_questions_processed / expected_questions) * 100
        
        summary = {
            'total_questions_processed': int(total_questions_processed),
            'marked_bubbles': int(marked_bubbles),
            'questions_answered': int(questions_answered),
            'expected_questions': expected_questions,
            'unanswered_questions': int(expected_questions - questions_answered),
            'completion_rate': f"{completion_rate:.1f}%",
            'detection_rate': f"{detection_rate:.1f}%",
            'processing_method': "Enhanced Detection with Skew Correction",
            'accuracy_score': f"{min(100, detection_rate):.1f}%"
        }
        
        # Convert processed image to base64
        processed_image_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        logger.info(f"Successfully processed OMR base64 API request: {summary}")
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary,
            'processed_image': processed_image_base64,
            'message': 'OMR sheet processed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing OMR base64 API request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process OMR sheet'
        }), 500


if __name__ == '__main__':
    # Print detected scanners on startup
    scanners = detect_scanners()
    print("Detected scanners:", scanners)

    # Or, to test scanning a document with scanner ID 0
    # (Be careful: this actually triggers a scan if a scanner is present!)
    # scan_bytes = scan_document(0)
    # with open("scanned_image.jpg", "wb") as f:
    #     f.write(scan_bytes)

    # Usual Flask app startup
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 6001))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"Starting Flask OMR Application on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

    # Use Flask's built-in server (works on all platforms)
    # app.run(host=host, port=port, debug=debug)
