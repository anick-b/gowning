"""
Gown Button Detection Component
==============================

Detects and assesses gown button alignment to determine if gown is worn properly.
Uses MediaPipe for face detection and YOLO for button detection.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import os
from pathlib import Path

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Gown button detection will be disabled.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Gown button detection will be disabled.")


class GownButtonDetector:
    def __init__(self, yolo_model_path: str = "model_yolo/best.pt"):
        """
        Initialize the Gown Button Detector
        
        Args:
            yolo_model_path: Path to the trained YOLO model for button detection
        """
        # Check if YOLO is available (required for button detection)
        if not YOLO_AVAILABLE:
            self.available = False
            print("Gown button detection is not available due to missing YOLO dependency.")
            return
        
        self.available = True
        
        # MediaPipe is optional - we'll use face detection from main system
        if MEDIAPIPE_AVAILABLE:
            # Initialize MediaPipe Face Detection
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
        else:
            print("MediaPipe not available, will use face detection from main system")
            self.use_mediapipe = False
        
        # Initialize YOLO model for button detection
        if os.path.exists(yolo_model_path):
            self.yolo_model = YOLO(yolo_model_path)
        else:
            print(f"Warning: YOLO model not found at {yolo_model_path}. Using dummy model.")
            self.yolo_model = None
        
        # Configuration parameters
        self.chin_offset_y = 20  # Pixels below chin to start cropping
        self.crop_height = 200   # Height of crop area
        self.crop_width = 150    # Width of crop area
        self.min_confidence_threshold = 0.3
        self.min_buttons_required = 3
        self.max_buttons_expected = 5
        self.max_horizontal_offset = 30  # Maximum horizontal deviation for alignment
        
        print("Gown Button Detector initialized successfully")
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in the image and return chin coordinates
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (chin_x, chin_y, face_width, face_height) or None if no face detected
        """
        if not self.available:
            return None
            
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_image)
        
        if not results.detections:
            return None
            
        # Get the first detected face
        detection = results.detections[0]
        
        # Get face bounding box
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        
        # Convert relative coordinates to absolute
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Get chin landmark (bottom center of face)
        chin_x = x + width // 2
        chin_y = y + height
        
        return chin_x, chin_y, width, height
    
    def crop_chin_area(self, image: np.ndarray, chin_x: int, chin_y: int, 
                       face_width: int, face_height: int) -> Tuple[np.ndarray, Tuple]:
        """
        Crop the area below the chin where gown buttons should be
        
        Args:
            image: Input image
            chin_x, chin_y: Chin coordinates
            face_width, face_height: Face dimensions
            
        Returns:
            Tuple of (cropped_image, crop_coordinates)
        """
        h, w = image.shape[:2]
        
        # Calculate crop boundaries
        crop_x1 = max(0, chin_x - self.crop_width // 2)
        crop_x2 = min(w, chin_x + self.crop_width // 2)
        crop_y1 = min(h, chin_y + self.chin_offset_y)
        crop_y2 = min(h, crop_y1 + self.crop_height)
        
        # Ensure we have a valid crop area
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            # Fallback: use face dimensions
            crop_x1 = max(0, chin_x - face_width // 2)
            crop_x2 = min(w, chin_x + face_width // 2)
            crop_y1 = min(h, chin_y + 10)  # Small offset from chin
            crop_y2 = min(h, crop_y1 + face_height)
        
        # Crop the image
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        return cropped, (crop_x1, crop_y1, crop_x2, crop_y2)
    
    def detect_buttons(self, cropped_image: np.ndarray) -> List[dict]:
        """
        Detect buttons in the cropped chin area using YOLO
        
        Args:
            cropped_image: Cropped image of the chin area
            
        Returns:
            List of detected buttons with coordinates and confidence
        """
        if not self.available or self.yolo_model is None:
            return []
            
        # Run YOLO inference
        results = self.yolo_model(cropped_image)
        
        buttons = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    buttons.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id
                    })
        
        return buttons
    
    def assess_gown_wearing(self, buttons: List[dict]) -> dict:
        """
        Assess whether the gown is worn properly based on button detection and alignment
        
        Args:
            buttons: List of detected buttons
            
        Returns:
            Dictionary with assessment results
        """
        # Filter buttons by confidence
        valid_buttons = [btn for btn in buttons if btn['confidence'] >= self.min_confidence_threshold]
        
        # Count valid buttons
        button_count = len(valid_buttons)
        
        # Check if we have enough buttons
        has_enough_buttons = button_count >= self.min_buttons_required
        
        # Check vertical alignment if we have enough buttons
        alignment_score = 0
        alignment_details = {}
        
        if has_enough_buttons and button_count >= 3:
            alignment_score, alignment_details = self._check_vertical_alignment(valid_buttons)
        
        # Determine if gown is worn properly
        is_proper = has_enough_buttons and alignment_score >= 0.7
        
        # Additional validation
        is_valid_count = button_count <= self.max_buttons_expected
        
        assessment = {
            'is_properly_worn': is_proper and is_valid_count,
            'button_count': button_count,
            'required_buttons': self.min_buttons_required,
            'max_expected_buttons': self.max_buttons_expected,
            'confidence_threshold': self.min_confidence_threshold,
            'buttons': valid_buttons,
            'is_valid_count': is_valid_count,
            'alignment_score': alignment_score,
            'alignment_details': alignment_details,
            'message': self._get_assessment_message(button_count, is_valid_count, alignment_score, alignment_details)
        }
        print(assessment)
        return assessment
    
    def _check_vertical_alignment(self, buttons: List[dict]) -> Tuple[float, dict]:
        """
        Check if buttons are arranged in a vertical straight line
        
        Args:
            buttons: List of detected buttons with bbox coordinates
            
        Returns:
            Tuple of (alignment_score, alignment_details)
        """
        if len(buttons) < 3:
            return 0.0, {'error': 'Need at least 3 buttons to check alignment'}
        
        # Sort buttons by Y coordinate (top to bottom)
        sorted_buttons = sorted(buttons, key=lambda btn: btn['bbox'][1])
        
        # Get center X coordinates of each button
        button_centers = []
        for btn in sorted_buttons:
            x1, y1, x2, y2 = btn['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            button_centers.append((center_x, center_y, btn['confidence']))
        
        # Calculate the ideal vertical line (average X coordinate)
        ideal_x = sum(center[0] for center in button_centers) / len(button_centers)
        
        # Calculate maximum allowed horizontal offset
        max_offset = self.max_horizontal_offset
        
        # Check each button's horizontal deviation from the ideal line
        deviations = []
        total_deviation = 0
        
        for center_x, center_y, confidence in button_centers:
            deviation = abs(center_x - ideal_x)
            deviations.append(deviation)
            total_deviation += deviation
        
        # Calculate alignment score (0.0 = perfect alignment, 1.0 = poor alignment)
        avg_deviation = total_deviation / len(button_centers)
        alignment_score = max(0.0, 1.0 - (avg_deviation / max_offset))
        
        # Check if all buttons are within the offset limit
        all_within_limit = all(dev <= max_offset for dev in deviations)
        
        # Calculate vertical spacing consistency
        vertical_spacings = []
        for i in range(1, len(button_centers)):
            spacing = button_centers[i][1] - button_centers[i-1][1]
            vertical_spacings.append(spacing)
        
        # Check if vertical spacings are reasonably consistent
        if len(vertical_spacings) >= 2:
            avg_spacing = sum(vertical_spacings) / len(vertical_spacings)
            spacing_variance = sum((sp - avg_spacing) ** 2 for sp in vertical_spacings) / len(vertical_spacings)
            spacing_consistency = max(0.0, 1.0 - (spacing_variance / (avg_spacing ** 2)))
        else:
            spacing_consistency = 1.0
        
        # Combine alignment score with spacing consistency
        final_score = (alignment_score * 0.7) + (spacing_consistency * 0.3)
        
        alignment_details = {
            'ideal_vertical_line_x': ideal_x,
            'button_centers': button_centers,
            'horizontal_deviations': deviations,
            'max_allowed_offset': max_offset,
            'all_within_offset_limit': all_within_limit,
            'vertical_spacings': vertical_spacings,
            'spacing_consistency': spacing_consistency,
            'alignment_score': alignment_score,
            'final_score': final_score
        }
        
        return final_score, alignment_details
    
    def _get_assessment_message(self, button_count: int, is_valid_count: bool, 
                               alignment_score: float, alignment_details: dict) -> str:
        """Generate assessment message with alignment information"""
        if not is_valid_count:
            return f"Warning: Detected {button_count} buttons, which is more than expected. Check for false positives."
        
        if button_count < self.min_buttons_required:
            return f"Gown may not be worn properly. Only detected {button_count} buttons (minimum {self.min_buttons_required} required)."
        
        if button_count >= 3 and alignment_score > 0:
            if alignment_score >= 0.8:
                return f"Gown appears to be worn properly. Detected {button_count} buttons in good vertical alignment (score: {alignment_score:.2f})."
            elif alignment_score >= 0.6:
                return f"Gown may be worn properly. Detected {button_count} buttons with acceptable vertical alignment (score: {alignment_score:.2f})."
            else:
                return f"Gown may not be worn properly. Detected {button_count} buttons but poor vertical alignment (score: {alignment_score:.2f})."
        
        return f"Gown appears to be worn properly. Detected {button_count} buttons (minimum {self.min_buttons_required} required)."
    
    def process_gown_buttons(self, image: np.ndarray, face_bboxes: List[Tuple]) -> Dict[str, Any]:
        """
        Process gown button detection for a detected gown
        
        Args:
            image: Input image
            face_bboxes: List of face bounding boxes from main detection
            
        Returns:
            Dictionary with button detection results
        """
        if not self.available:
            return {
                'success': False,
                'error': 'Gown button detection not available (missing dependencies)',
                'assessment': {
                    'is_properly_worn': False,
                    'message': 'Button detection unavailable'
                }
            }
        
        if not face_bboxes:
            return {
                'success': False,
                'error': 'No face detected for gown button analysis',
                'assessment': {
                    'is_properly_worn': False,
                    'message': 'No face detected'
                }
            }
        
        # Use the first face for analysis
        face_bbox = face_bboxes[0]
        fx, fy, fw, fh = face_bbox
        
        # Calculate chin coordinates from face bbox
        chin_x = fx + fw // 2
        chin_y = fy + fh
        
        # Crop chin area
        cropped_image, crop_coords = self.crop_chin_area(
            image, chin_x, chin_y, fw, fh
        )
        
        # Detect buttons
        buttons = self.detect_buttons(cropped_image)
        
        # Assess gown wearing
        assessment = self.assess_gown_wearing(buttons)
        
        # Prepare result
        result = {
            'success': True,
            'face_used': face_bbox,
            'chin_coordinates': (chin_x, chin_y),
            'crop_coordinates': crop_coords,
            'buttons_detected': len(buttons),
            'assessment': assessment,
            'cropped_image': cropped_image
        }
        
        return result 