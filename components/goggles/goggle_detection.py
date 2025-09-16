"""
PPE Detection Component
=======================

Detects proper goggles and shoes by comparing SAM-generated masks with reference templates.
Approach: SAM masks -> Embedding comparison -> 90% threshold -> Color-coded visualization
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import tempfile
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from components.face_detection.face_detection import FaceDetectorDNN, FaceDetectorMTCNN

# SAM imports
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class GogglesDetector:
    """
    PPE detector for goggles and shoes using embedding comparison with SAM segmentation
    """
    
    def __init__(self, reference_path: str = None, sam_checkpoint: str = None):
        """
        Initialize PPE detector
        
        Args:
            reference_path: Path to reference templates (defaults to goggles)
            sam_checkpoint: Path to SAM model checkpoint
        """
        self.face_detector = FaceDetectorMTCNN()
        self.reference_path = Path(reference_path) if reference_path else Path("reference_data/goggles")
        self.sam_checkpoint = sam_checkpoint or "sam_vit_h_4b8939.pth"

        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Detection parameters
        self.similarity_threshold = 0.90  # 90% similarity threshold (for goggles and shoes)
        self.cuff_similarity_threshold = 0.50  # 50% similarity threshold for cuffs (more lenient)
        self.min_mask_area = 100
        self.max_mask_area = 50000
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pretrained ResNet50 (feature extractor only)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # remove fc layer
        self.feature_extractor.eval().to(self.device)

        # Normalization & resizing for ResNet
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Load reference data for goggles, shoes, and cuffs (left/right separately)
        self.goggles_data = self._load_reference_embeddings("reference_data/goggles")
        self.left_shoes_data = self._load_reference_embeddings("reference_data/shoes/left")
        self.right_shoes_data = self._load_reference_embeddings("reference_data/shoes/right")
        self.shoes_data = self.left_shoes_data + self.right_shoes_data  # Combined for compatibility
        self.left_cuffs_data = self._load_reference_embeddings("reference_data/cuffs/left")
        self.right_cuffs_data = self._load_reference_embeddings("reference_data/cuffs/right")
        self.cuffs_data = self.left_cuffs_data + self.right_cuffs_data  # Combined for compatibility
        
        # Also load traditional reference masks for compatibility
        self.reference_masks = self._load_reference_masks()
        
        # Initialize SAM (lazy loading)
        self.sam_model = None
        self.mask_generator = None
        
        total_refs = len(self.goggles_data) + len(self.shoes_data) + len(self.cuffs_data)
        print(f"PPE Detector initialized with {len(self.goggles_data)} goggles, {len(self.left_shoes_data)} left shoes, {len(self.right_shoes_data)} right shoes, {len(self.left_cuffs_data)} left cuffs, and {len(self.right_cuffs_data)} right cuffs references (total: {total_refs})")

    def _load_reference_embeddings(self, reference_path: str):
        """
        Load reference images and compute their embeddings for shoes and cuffs (color images from SAM crops)
        """
        ref_data = []
        ref_dir = Path(reference_path)
        
        if not ref_dir.exists():
            print(f"Warning: Reference path {ref_dir} does not exist")
            return ref_data

        # Determine PPE type and side from path
        if "shoes" in str(reference_path):
            ppe_type = "shoes"
            side = "left" if "left" in str(reference_path) else "right"
        elif "cuffs" in str(reference_path):
            ppe_type = "cuffs"
            side = "left" if "left" in str(reference_path) else "right"
        else:
            ppe_type = "goggles"
            side = None

        # Load PNG files
        for ref_img_path in ref_dir.glob("*.png"):
            ref_img = cv2.imread(str(ref_img_path))
            if ref_img is None:
                continue

            # Convert to RGB PIL for transform
            ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(ref_img_rgb)

            embedding = self._get_embedding(pil_img)
            ref_data.append({
                "name": ref_img_path.name,
                "embedding": embedding,
                "type": ppe_type,
                "side": side
            })

        # Also load JPG files
        for ref_img_path in ref_dir.glob("*.jpg"):
            ref_img = cv2.imread(str(ref_img_path))
            if ref_img is None:
                continue

            ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(ref_img_rgb)

            embedding = self._get_embedding(pil_img)
            ref_data.append({
                "name": ref_img_path.name,
                "embedding": embedding,
                "type": ppe_type,
                "side": side
            })

        print(f"Loaded {len(ref_data)} reference embeddings from {reference_path}")
        return ref_data

    def _get_embedding(self, pil_img: Image.Image):
        """
        Run an image through ResNet → embedding vector.
        """
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(x)
        return feat.view(-1)  # flatten to 2048-dim
    
    def _binary_mask_similarity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate similarity between two binary masks using IoU (Intersection over Union)
        """
        # Ensure both masks are binary (0 or 255)
        mask1_bin = (mask1 > 127).astype(np.uint8)
        mask2_bin = (mask2 > 127).astype(np.uint8)
        
        # Resize masks to same size if needed
        if mask1_bin.shape != mask2_bin.shape:
            mask2_bin = cv2.resize(mask2_bin, (mask1_bin.shape[1], mask1_bin.shape[0]))
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1_bin, mask2_bin).sum()
        union = np.logical_or(mask1_bin, mask2_bin).sum()
        
        # Avoid division by zero
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _determine_shoe_side(self, shoe_bbox, face_bboxes, image_width):
        """
        Determine if shoe is left or right based on position relative to face center
        Note: Box on left side of face = right shoe (from person's perspective)
        Box on right side of face = left shoe (from person's perspective)
        """
        shoe_x, shoe_y, shoe_w, shoe_h = shoe_bbox
        shoe_center_x = shoe_x + shoe_w // 2
        
        if not face_bboxes:
            # Fallback: use image center as reference if no face detected
            return "right" if shoe_center_x < image_width // 2 else "left"
        
        # Use the closest face as reference
        closest_face = min(face_bboxes, key=lambda f: abs((f[0] + f[2]//2) - shoe_center_x))
        face_x, face_y, face_w, face_h = closest_face
        face_center_x = face_x + face_w // 2
        
        # Box on left side of face = right shoe, box on right side = left shoe
        return "right" if shoe_center_x < face_center_x else "left"
    
    def _load_reference_masks(self) -> List[Dict[str, Any]]:
        """Load reference binary masks from the reference directory (for compatibility)"""
        reference_masks = []
        
        if not self.reference_path.exists():
            print(f"Warning: Reference path {self.reference_path} does not exist")
            return reference_masks
        
        # Load all image files as binary masks
        for mask_file in self.reference_path.glob("*.png"):
            try:
                # Load as grayscale
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Convert to binary (0 or 255)
                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    
                    reference_masks.append({
                        'name': mask_file.stem,
                        'path': str(mask_file),
                        'mask': binary_mask,
                        'shape': binary_mask.shape
                    })
                    
            except Exception as e:
                print(f"Error loading reference mask {mask_file}: {e}")
        
        return reference_masks
    
    def _initialize_sam(self):
        """Initialize SAM model (lazy loading) - ONLY ONCE"""
        if not SAM_AVAILABLE:
            raise RuntimeError("SAM not available. Please install segment-anything.")
        
        if self.sam_model is None:
            print("🔄 Loading SAM model (this happens only once)...")
            if not os.path.exists(self.sam_checkpoint):
                raise FileNotFoundError(f"SAM checkpoint not found: {self.sam_checkpoint}")
            
            self.sam_model = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
            self.sam_model.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=self.min_mask_area,
            )
            print("✅ SAM model loaded successfully and cached in memory")
        else:
            print("♻️ Using cached SAM model (no reloading needed)")
    
    def _generate_sam_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks using SAM"""
        self._initialize_sam()
        
        print("Generating SAM masks...")
        masks = self.mask_generator.generate(image)
        print(f"Generated {len(masks)} masks")
        
        return masks

    def _find_best_matches(self, sam_masks, full_image):
        """Find best matches using different approaches for goggles vs shoes"""
        matches = []
        all_references = self.goggles_data + self.shoes_data + self.cuffs_data

        print(f"Processing {len(sam_masks)} SAM masks against {len(all_references)} references...")

        for i, sam_mask_data in enumerate(sam_masks):
            sam_mask = sam_mask_data["segmentation"]  # This is a boolean numpy array
            bbox = sam_mask_data["bbox"]
            area = sam_mask_data["area"]

            print(f"Processing mask {i}: area={area}, min_area={self.min_mask_area}, max_area={self.max_mask_area}")

            # For GOGGLES: Use full masked image (keep region, black background)
            mask_uint8 = sam_mask.astype(np.uint8)
            masked = cv2.bitwise_and(full_image, full_image, mask=mask_uint8)
            pil_masked = Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
            goggle_embedding = self._get_embedding(pil_masked)

            # For SHOES and CUFFS: Crop original image using SAM mask region
            x, y, w, h = bbox
            cropped_region = full_image[y:y+h, x:x+w]
            shoe_embedding = None
            cuff_embedding = None
            if cropped_region.size > 0:
                cropped_rgb = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
                pil_cropped = Image.fromarray(cropped_rgb)
                shoe_embedding = self._get_embedding(pil_cropped)
                cuff_embedding = self._get_embedding(pil_cropped)  # Same embedding for both

            # Find best match for GOGGLES using full masked image
            best_goggle_sim = 0.0
            best_goggle_ref = None
            for ref in self.goggles_data:
                sim = F.cosine_similarity(
                    goggle_embedding.unsqueeze(0),
                    ref["embedding"].unsqueeze(0)
                ).item()
                if sim > best_goggle_sim:
                    best_goggle_sim = sim
                    best_goggle_ref = ref

            # Find best match for SHOES using cropped image
            best_shoe_sim = 0.0
            best_shoe_ref = None
            shoe_side = "unknown"
            if shoe_embedding is not None:
                # Try both left and right shoe references
                for ref in self.shoes_data:
                    sim = F.cosine_similarity(
                        shoe_embedding.unsqueeze(0),
                        ref["embedding"].unsqueeze(0)
                    ).item()
                    if sim > best_shoe_sim:
                        best_shoe_sim = sim
                        best_shoe_ref = ref
                        # Extract side from reference name
                        shoe_side = ref.get("side", "unknown")

            # Find best match for CUFFS using cropped image
            best_cuff_sim = 0.0
            best_cuff_ref = None
            cuff_side = "unknown"
            if cuff_embedding is not None:
                # Try both left and right cuff references
                for ref in self.cuffs_data:
                    sim = F.cosine_similarity(
                        cuff_embedding.unsqueeze(0),
                        ref["embedding"].unsqueeze(0)
                    ).item()
                    if sim > best_cuff_sim:
                        best_cuff_sim = sim
                        best_cuff_ref = ref
                        # Extract side from reference name
                        cuff_side = ref.get("side", "unknown")

            # Determine which PPE type has better match
            if best_goggle_sim > best_shoe_sim and best_goggle_sim > best_cuff_sim:
                best_sim = best_goggle_sim
                best_ref = best_goggle_ref
                ppe_type = "goggles"
            elif best_shoe_sim > best_cuff_sim:
                best_sim = best_shoe_sim
                best_ref = best_shoe_ref
                ppe_type = "shoes"
            else:
                best_sim = best_cuff_sim
                best_ref = best_cuff_ref
                ppe_type = "cuffs"

            # Use different thresholds for different PPE types
            if ppe_type == "cuffs":
                is_detected = best_sim >= self.cuff_similarity_threshold
            else:
                is_detected = best_sim >= self.similarity_threshold
            
            # Create the match data
            match_data = {
                "sam_mask_id": i,
                "sam_mask": sam_mask,  # raw boolean mask
                "full_binary_mask": (sam_mask.astype(np.uint8) * 255),  # uint8 mask for saving
                "bbox": bbox,
                "area": area,
                "similarity": best_sim,
                "ppe_type": ppe_type,
                "is_detected": is_detected,
                "is_goggle": is_detected and ppe_type == "goggles",
                "is_shoe": is_detected and ppe_type == "shoes",
                "is_cuff": is_detected and ppe_type == "cuffs",
                "shoe_side": shoe_side if ppe_type == "shoes" else None,
                "cuff_side": cuff_side if ppe_type == "cuffs" else None,
                "reference_match": {
                    "name": best_ref["name"] if best_ref else None,
                    "type": ppe_type,
                    "side": shoe_side if ppe_type == "shoes" else (cuff_side if ppe_type == "cuffs" else None)
                },
                "predicted_iou": sam_mask_data.get("predicted_iou", 0.0),
                "stability_score": sam_mask_data.get("stability_score", 0.0),
                "area_filtered": area < self.min_mask_area or area > self.max_mask_area,
                "goggle_similarity": best_goggle_sim,
                "shoe_similarity": best_shoe_sim,
                "cuff_similarity": best_cuff_sim
            }

            matches.append(match_data)
            
            # Debug print
            status = f"{ppe_type.upper()}" if is_detected else "NO_MATCH"
            if ppe_type == "shoes" and is_detected:
                status = f"{ppe_type.upper()}_{shoe_side.upper()}"
            elif ppe_type == "cuffs" and is_detected:
                status = f"{ppe_type.upper()}_{cuff_side.upper()}"
            filtered = " [AREA_FILTERED]" if match_data['area_filtered'] else ""
            print(f"  Match {i}: {status}, sim={best_sim:.3f} (goggle={best_goggle_sim:.3f}, shoe={best_shoe_sim:.3f}, cuff={best_cuff_sim:.3f}), area={area}{filtered}")

        print(f"Created {len(matches)} total matches")
        return sorted(matches, key=lambda x: x["similarity"], reverse=True)
    
    def _update_shoe_sides_with_face_info(self, matches, faces, image_width):
        """
        Update shoe and cuff sides based on face positions after face detection
        This handles cases where only one shoe or cuff is detected
        """
        for match in matches:
            if match['ppe_type'] == 'shoes' and match['is_detected']:
                bbox = match['bbox']
                updated_side = self._determine_shoe_side(bbox, faces, image_width)
                match['shoe_side'] = updated_side
                if 'reference_match' in match:
                    match['reference_match']['side'] = updated_side
            elif match['ppe_type'] == 'cuffs' and match['is_detected']:
                bbox = match['bbox']
                updated_side = self._determine_shoe_side(bbox, faces, image_width)  # Same logic for cuffs
                match['cuff_side'] = updated_side
                if 'reference_match' in match:
                    match['reference_match']['side'] = updated_side
        return matches
    
    def detect_goggles(self, image_path: str, max_size: int = 1024) -> Dict[str, Any]:
        """
        Main detection function - now detects both goggles and shoes using one SAM run
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"[DEBUG] Original image shape: {image.shape}")
        
        # Store original dimensions for coordinate scaling
        original_height, original_width = image.shape[:2]
        
        # Resize image if too large (to prevent memory overflow)
        scale_factor = 1.0
        if max(original_height, original_width) > max_size:
            if original_width > original_height:
                new_width = max_size
                new_height = int((original_height * max_size) / original_width)
            else:
                new_height = max_size
                new_width = int((original_width * max_size) / original_height)
            
            scale_factor = max(original_width / new_width, original_height / new_height)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"[DEBUG] Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
            print(f"[DEBUG] Scale factor: {scale_factor:.3f}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate SAM masks ONCE for both goggles and shoes
        sam_masks = self._generate_sam_masks(image_rgb)
        
        # Find best matches for both PPE types
        matches = self._find_best_matches(sam_masks, image)
        
        # Separate matches by type
        goggle_matches = [m for m in matches if m['is_goggle']]
        shoe_matches = [m for m in matches if m['is_shoe']]
        cuff_matches = [m for m in matches if m['is_cuff']]
        
        # Scale bounding boxes back to original image dimensions if image was resized
        if scale_factor != 1.0:
            print(f"[DEBUG] Scaling bounding boxes back to original dimensions (factor: {scale_factor:.3f})")
            for match in matches:
                bbox = match['bbox']
                x, y, w, h = bbox
                scaled_bbox = [
                    int(x * scale_factor),
                    int(y * scale_factor), 
                    int(w * scale_factor),
                    int(h * scale_factor)
                ]
                match['bbox'] = scaled_bbox
                match['original_bbox'] = bbox
        
        # Run face detection
        faces = self.face_detector.detect_faces(image)
        scaled_faces = []
        for (fx, fy, fw, fh) in faces:
            if scale_factor != 1.0:
                scaled_faces.append([
                    int(fx * scale_factor),
                    int(fy * scale_factor),
                    int(fw * scale_factor),
                    int(fh * scale_factor),
                ])
            else:
                scaled_faces.append([int(fx), int(fy), int(fw), int(fh)])
        
        # Update shoe sides based on face positions (handles single shoe detection)
        matches = self._update_shoe_sides_with_face_info(matches, scaled_faces, original_width)

        # Filter goggles that are inside face bounding boxes
        valid_goggle_matches = []
        for match in goggle_matches:
            gx, gy, gw, gh = match['bbox']
            g_center = (gx + gw // 2, gy + gh // 2)
            inside_face = any(
                (fx <= g_center[0] <= fx + fw) and (fy <= g_center[1] <= fy + fh)
                for (fx, fy, fw, fh) in scaled_faces
            )
            match['inside_face'] = inside_face
            if inside_face:
                valid_goggle_matches.append(match)
        
        # For shoes and cuffs, we don't filter by face location
        valid_shoe_matches = shoe_matches
        valid_cuff_matches = cuff_matches
        
        # Create result
        result = {
            'goggles_detected': len(valid_goggle_matches) > 0,
            'shoes_detected': len(valid_shoe_matches) > 0,
            'cuffs_detected': len(valid_cuff_matches) > 0,
            'goggle_matches': len(valid_goggle_matches),
            'shoe_matches': len(valid_shoe_matches),
            'cuff_matches': len(valid_cuff_matches),
            'confidence': valid_goggle_matches[0]['similarity'] if valid_goggle_matches else 0.0,
            'shoe_confidence': valid_shoe_matches[0]['similarity'] if valid_shoe_matches else 0.0,
            'cuff_confidence': valid_cuff_matches[0]['similarity'] if valid_cuff_matches else 0.0,
            'threshold': self.similarity_threshold,
            'total_sam_masks': len(sam_masks),
            'best_match': valid_goggle_matches[0] if valid_goggle_matches else None,
            'best_shoe_match': valid_shoe_matches[0] if valid_shoe_matches else None,
            'best_cuff_match': valid_cuff_matches[0] if valid_cuff_matches else None,
            'all_matches': matches,
            'processing_time': time.time() - start_time,
            'original_image_shape': (original_height, original_width),
            'processed_image_shape': image.shape,
            'scale_factor': scale_factor,
            'faces': scaled_faces,
            'reference_masks_count': len(self.goggles_data)
        }
        
        return result
    
    def create_visualization(self, image_path: str, detection_result: Dict[str, Any], output_path: str = None) -> str:
        """
        Create visualization with green boxes for goggles and blue boxes for shoes
        """
        # Load original image
        image = cv2.imread(image_path)
        annotated = image.copy()

        # Draw face bounding boxes (light blue)
        for face in detection_result['faces']:
            fx, fy, fw, fh = face
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (255, 200, 0), 2)
        
        # Draw bounding boxes for detected PPE
        for match in detection_result['all_matches']:
            if not match['is_detected']:
                continue
                
            bbox = match['bbox']
            x, y, w, h = bbox
            similarity = match['similarity']
            ppe_type = match['ppe_type']
            
            # Color coding: Green for goggles, Blue for shoes, Purple for cuffs
            if match['is_goggle'] and match.get('inside_face', False):
                color = (0, 255, 0)  # Green for goggles
                label = f"Goggles: {similarity:.2f}"
            elif match['is_shoe']:
                color = (255, 0, 0)  # Blue for shoes  
                label = f"Shoes: {similarity:.2f}"
            elif match['is_cuff']:
                color = (128, 0, 128)  # Purple for cuffs
                side = match.get('cuff_side', 'Unknown')
                label = f"Cuffs ({side}): {similarity:.2f}"
            else:
                continue  # Skip if not a valid detection
            
            thickness = 3
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            
            # Add label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(annotated, (x, y - label_size[1] - 10), 
                        (x + label_size[0], y), color, -1)
            
            # Label text
            cv2.putText(annotated, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add overall status
        status_lines = []
        if detection_result['goggles_detected']:
            status_lines.append(f"GOGGLES DETECTED ({detection_result['goggle_matches']} found)")
        else:
            status_lines.append("NO GOGGLES DETECTED")
            
        if detection_result['shoes_detected']:
            status_lines.append(f"SHOES DETECTED ({detection_result['shoe_matches']} found)")
        else:
            status_lines.append("NO SHOES DETECTED")
            
        if detection_result['cuffs_detected']:
            status_lines.append(f"CUFFS DETECTED ({detection_result['cuff_matches']} found)")
        else:
            status_lines.append("NO CUFFS DETECTED")
        
        # Draw status
        for i, status in enumerate(status_lines):
            color = (0, 255, 0) if "DETECTED" in status and "NO" not in status else (0, 0, 255)
            cv2.putText(annotated, status, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add confidence info
        if detection_result['goggles_detected'] or detection_result['shoes_detected'] or detection_result['cuffs_detected']:
            conf_y = 30 + len(status_lines) * 30 + 20
            if detection_result['goggles_detected']:
                conf_text = f"Goggles Confidence: {detection_result['confidence']:.2f}"
                cv2.putText(annotated, conf_text, (10, conf_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                conf_y += 25
            if detection_result['shoes_detected']:
                conf_text = f"Shoes Confidence: {detection_result['shoe_confidence']:.2f}"
                cv2.putText(annotated, conf_text, (10, conf_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                conf_y += 25
            if detection_result['cuffs_detected']:
                conf_text = f"Cuffs Confidence: {detection_result['cuff_confidence']:.2f}"
                cv2.putText(annotated, conf_text, (10, conf_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save visualization
        if output_path is None:
            output_path = str(self.temp_dir / f"ppe_detection_{int(time.time())}.jpg")
        
        cv2.imwrite(output_path, annotated)
        print(f"Visualization saved to: {output_path}")
        
        return output_path

    def save_results(self, detection_result: Dict[str, Any], output_path: str = None) -> str:
        """Save detection results to JSON file"""
        if output_path is None:
            output_path = str(self.temp_dir / f"ppe_results_{int(time.time())}.json")
        
        # Make results JSON serializable - use deep copy to avoid modifying original
        import copy
        serializable_result = copy.deepcopy(detection_result)

        # Convert numpy arrays to lists for all matches
        if 'all_matches' in serializable_result:
            for match in serializable_result['all_matches']:
                keys_to_remove = []
                for key, value in match.items():
                    if isinstance(value, np.ndarray):
                        keys_to_remove.append(key)
                    elif key == 'bbox' and isinstance(value, list):
                        match[key] = [int(x) for x in value]
                
                for key in keys_to_remove:
                    del match[key]
        
        # Handle best matches similarly
        for match_key in ['best_match', 'best_shoe_match']:
            if match_key in serializable_result and serializable_result[match_key]:
                best_match = serializable_result[match_key]
                keys_to_remove = []
                for key, value in best_match.items():
                    if isinstance(value, np.ndarray):
                        keys_to_remove.append(key)
                    elif key == 'bbox' and isinstance(value, list):
                        best_match[key] = [int(x) for x in value]
                
                for key in keys_to_remove:
                    del best_match[key]
        
        # Convert any remaining numpy types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_result = convert_numpy_types(serializable_result)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        return output_path

    def save_debug_masks(self, detection_result: Dict[str, Any], output_dir: str = None) -> str:
        """Save debug masks for analysis"""
        if output_dir is None:
            output_dir = str(self.temp_dir / f"debug_masks_{int(time.time())}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save masks organized by PPE type
        for i, match in enumerate(detection_result.get('all_matches', [])):
            if 'full_binary_mask' in match:
                ppe_type = match.get('ppe_type', 'unknown')
                similarity = match['similarity']
                detected = "DETECTED" if match['is_detected'] else "NO_MATCH"
                
                mask_filename = f"sam_mask_{i:03d}_{ppe_type}_sim_{similarity:.3f}_{detected}.png"
                mask_path = os.path.join(output_dir, mask_filename)
                cv2.imwrite(mask_path, match['full_binary_mask'])
        
        print(f"Debug masks saved to: {output_dir}")
        return output_dir

    # Keep existing methods for compatibility
    def save_all_masks_and_comparisons(self, image_path: str, detection_result: Dict[str, Any], output_dir: str = None) -> str:
        """Save all SAM masks and create comparison visualizations"""
        if output_dir is None:
            output_dir = str(self.temp_dir / f"all_masks_comparisons_{int(time.time())}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        all_masks_dir = Path(output_dir) / "all_masks"
        comparisons_dir = Path(output_dir) / "comparisons"
        all_masks_dir.mkdir(exist_ok=True)
        comparisons_dir.mkdir(exist_ok=True)
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not load original image from {image_path}")
            return output_dir
        
        # Process matches (should now have numpy arrays due to deep copy fix)
        matches = detection_result.get('all_matches', [])
        print(f"Saving {len(matches)} SAM masks...")
        
        for i, match in enumerate(matches):
            if 'full_binary_mask' not in match:
                print(f"Warning: Match {i} missing full_binary_mask, skipping")
                continue
                
            ppe_type = match.get('ppe_type', 'unknown')
            similarity = match.get('similarity', 0.0)
            detected = "DETECTED" if match.get('is_detected', False) else "NO_MATCH"
            
            # Save binary mask
            mask_filename = f"sam_mask_{i:03d}_{ppe_type}_sim_{similarity:.3f}_{detected}.png"
            mask_path = all_masks_dir / mask_filename
            cv2.imwrite(str(mask_path), match['full_binary_mask'])
            
            # Save color cropped region for ALL masks (both goggles and shoes)
            bbox = match.get('bbox', [0, 0, 100, 100])
            x, y, w, h = bbox
            
            # Crop the region from original image
            cropped_region = original_image[y:y+h, x:x+w]
            if cropped_region.size > 0:
                # Save color cropped region
                cropped_filename = f"cropped_color_{i:03d}_{ppe_type}_sim_{similarity:.3f}_{detected}.png"
                cropped_path = all_masks_dir / cropped_filename
                cv2.imwrite(str(cropped_path), cropped_region)
                print(f"Saved color cropped region: {cropped_path}")
            
            # Create comparison based on PPE type
            print(f"Match {i}: similarity={similarity:.3f}, ppe_type={ppe_type}")
            try:
                # Get the reference image that had highest similarity
                ref_match = match.get('reference_match', {})
                ref_name = ref_match.get('name', 'unknown')
                ref_type = ref_match.get('type', 'unknown')
                
                print(f"Creating comparison for match {i}: {ppe_type} vs {ref_name} (similarity: {similarity:.3f})")
                
                # Load the reference image
                ref_path = f"reference_data/{ref_type}/{ref_name}"
                if not os.path.exists(ref_path):
                    print(f"Warning: Reference image not found: {ref_path}")
                    continue
                
                ref_image = cv2.imread(ref_path)
                if ref_image is None:
                    print(f"Warning: Could not load reference image: {ref_path}")
                    continue
                
                # Different logic for shoes vs goggles
                if ppe_type == 'shoes':
                    # For shoes: crop original image using bbox and compare with reference color image
                    bbox = match.get('bbox', [0, 0, 100, 100])
                    x, y, w, h = bbox
                    
                    # Crop the region from original image
                    cropped_region = original_image[y:y+h, x:x+w]
                    if cropped_region.size == 0:
                        print(f"Warning: Empty region for match {i}, skipping comparison")
                        continue
                    
                    # Resize both images to same height for side-by-side display
                    target_height = 200
                    
                    # Resize cropped region
                    cropped_resized = cv2.resize(cropped_region, (int(cropped_region.shape[1] * target_height / cropped_region.shape[0]), target_height))
                    
                    # Resize reference image
                    ref_resized = cv2.resize(ref_image, (int(ref_image.shape[1] * target_height / ref_image.shape[0]), target_height))
                    
                    # Create side-by-side comparison: cropped color image vs reference color image
                    comparison = np.hstack([
                        cropped_resized,  # Left: cropped color image from original
                        ref_resized       # Right: reference color image
                    ])
                    
                else:
                    # For goggles: keep existing binary mask comparison (unchanged)
                    sam_mask = match['full_binary_mask']
                    
                    # Resize both images to same height for side-by-side display
                    target_height = 200
                    
                    # Resize SAM mask
                    sam_resized = cv2.resize(sam_mask, (int(sam_mask.shape[1] * target_height / sam_mask.shape[0]), target_height))
                    sam_3ch = cv2.cvtColor(sam_resized, cv2.COLOR_GRAY2BGR)
                    
                    # Resize reference image
                    ref_resized = cv2.resize(ref_image, (int(ref_image.shape[1] * target_height / ref_image.shape[0]), target_height))
                    
                    # Create side-by-side comparison: SAM mask vs reference image
                    comparison = np.hstack([
                        sam_3ch,        # Left: SAM mask
                        ref_resized     # Right: Reference image
                    ])
                
                # Create filename with similarity and reference info
                status = "DETECTED" if match.get('is_detected', False) else "NO_MATCH"
                comp_filename = f"comparison_{i:03d}_{ppe_type}_sim_{similarity:.3f}_{status}_{ref_name}"
                comp_path = comparisons_dir / comp_filename
                cv2.imwrite(str(comp_path), comparison)
                print(f"Saved comparison: {comp_path}")
            except Exception as e:
                print(f"Warning: Could not create comparison for match {i}: {e}")
        
        # Create summary image
        self._create_summary_image(detection_result, output_dir, original_image)
        
        print(f"All masks and comparisons saved to: {output_dir}")
        return output_dir
    

    
    def _create_summary_image(self, detection_result: Dict[str, Any], output_dir: str, original_image: np.ndarray):
        """Create a summary image showing all detections"""
        summary_path = Path(output_dir) / "00_SUMMARY_top_matches.png"
        
        # Get top matches (detected ones) - filter for matches that have full_binary_mask
        detected_matches = [m for m in detection_result.get('all_matches', []) 
                           if m.get('is_detected', False) and 'full_binary_mask' in m]
        top_matches = sorted(detected_matches, key=lambda x: x['similarity'], reverse=True)[:6]  # Top 6
        
        if not top_matches:
            # If no detections, create a simple message image
            summary_img = np.ones((300, 600, 3), dtype=np.uint8) * 255
            cv2.putText(summary_img, "NO PPE DETECTED", (150, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(str(summary_path), summary_img)
            return
        
        # Create grid of top matches
        rows = 2
        cols = 3
        cell_height = 200
        cell_width = 200
        
        summary_img = np.ones((rows * cell_height, cols * cell_width, 3), dtype=np.uint8) * 255
        
        for idx, match in enumerate(top_matches):
            if idx >= rows * cols:
                break
                
            row = idx // cols
            col = idx % cols
            
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width
            
            # Check if full_binary_mask exists before using it
            if 'full_binary_mask' not in match:
                print(f"Warning: Match {idx} missing full_binary_mask, skipping")
                continue
                
            # Resize mask to fit cell
            mask_resized = cv2.resize(match['full_binary_mask'], (cell_width, cell_height))
            
            # Apply mask to original image region
            bbox = match.get('bbox', [0, 0, 100, 100])  # Default bbox if missing
            x, y, w, h = bbox
            region = original_image[y:y+h, x:x+w]
            if region.size > 0:
                region_resized = cv2.resize(region, (cell_width, cell_height))
                mask_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR) / 255.0
                masked_region = (region_resized * mask_3ch).astype(np.uint8)
            else:
                masked_region = mask_resized
            
            # Place in summary grid
            summary_img[y_start:y_end, x_start:x_end] = masked_region
            
            # Add label
            ppe_type = match.get('ppe_type', 'unknown')
            similarity = match.get('similarity', 0.0)
            label = f"{ppe_type}: {similarity:.2f}"
            cv2.putText(summary_img, label, (x_start + 5, y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(str(summary_path), summary_img)
        print(f"Summary image saved to: {summary_path}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = GogglesDetector()
    
    print("PPE Detector ready for use!")
    print(f"Goggles references: {len(detector.goggles_data)}")
    print(f"Shoes references: {len(detector.shoes_data)}")
    print(f"Similarity threshold: {detector.similarity_threshold}")