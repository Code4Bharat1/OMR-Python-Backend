from flask import Flask, request, render_template, redirect, url_for, send_file, flash
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Create upload directory
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup logging for production
if not app.debug:
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
    Enhanced bubble detection to ensure all 180 questions are found
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
        
        logger.info(f"Enhanced detection found {len(unique_circles)} unique circles")
        return unique_circles
    
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

# Error handlers
@app.errorhandler(413)
def too_large(e):
    flash('File too large. Please upload a file smaller than 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal server error: {e}")
    flash('An internal error occurred. Please try again.')
    return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return {'status': 'healthy', 'service': 'OMR Processing'}, 200

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'omrfile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['omrfile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)
        
        try:
            image_bytes = file.read()
            
            # Process with enhanced detection
            results, img_bytes, total_questions_processed = process_omr_enhanced(image_bytes)
            
            # Save files with unique names to prevent conflicts
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            
            out_path = os.path.join(UPLOAD_FOLDER, f'annotated_{unique_id}.png')
            with open(out_path, 'wb') as f:
                f.write(img_bytes)
            
            json_path = os.path.join(UPLOAD_FOLDER, f'results_{unique_id}.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
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
            
            logger.info(f"Successfully processed OMR: {summary}")
            
            return render_template('result.html',
                                 results=results,
                                 summary=summary,
                                 image_url=url_for('uploaded_file', filename=f'annotated_{unique_id}.png'),
                                 json_url=url_for('uploaded_file', filename=f'results_{unique_id}.json'))
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            flash(f"Error processing file: {str(e)}")
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return "File not found", 404
            
        if filename.endswith('.json'):
            return send_file(file_path, mimetype='application/json')
        else:
            return send_file(file_path, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        return "Error serving file", 500

if __name__ == '__main__':
    # Production server should use a proper WSGI server like Gunicorn
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])