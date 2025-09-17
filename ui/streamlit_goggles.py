"""
Streamlit UI for PPE Detection System
====================================

Simple interface for testing goggles and shoes detection with SAM + embedding comparison.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tempfile
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Ellipse
import matplotlib.patches as mpatches

# Add parent directory to path to import the detector
sys.path.append(str(Path(__file__).parent.parent))

try:
    from components.goggles.goggle_detection import GogglesDetector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    DETECTOR_AVAILABLE = False
    import_error = str(e)

# Page configuration
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🥽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .ppe-type-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_human_figure_compliance(detection_result):
    """
    Create a human figure showing PPE compliance status with predefined areas
    
    Args:
        detection_result: Dictionary containing detection results
        
    Returns:
        matplotlib figure with human figure and compliance visualization
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Define colors
    GREEN = '#2E8B57'  # Green for compliant
    RED = '#DC143C'    # Red for non-compliant
    GRAY = '#D3D3D3'   # Gray for neutral/not detected
    
    # Check detection results
    goggles_detected = detection_result.get('goggles_detected', False)
    left_shoe_detected = any(match.get('is_shoe', False) and match.get('shoe_side') == 'left' 
                            for match in detection_result.get('all_matches', []))
    right_shoe_detected = any(match.get('is_shoe', False) and match.get('shoe_side') == 'right' 
                            for match in detection_result.get('all_matches', []))
    gown_detected = detection_result.get('gown_detected', False)
    hairnet_detected = detection_result.get('hairnet_detected', False)
    
    # Check if gown is properly worn (if detected)
    gown_properly_worn = False
    if gown_detected:
        gown_matches = [m for m in detection_result.get('all_matches', []) if m.get('is_gown', False)]
        print(f"DEBUG: Found {len(gown_matches)} gown matches")
        if gown_matches:
            for gown_match in gown_matches:
                print(f"DEBUG: Gown match: {gown_match}")
                if gown_match.get('gown_assessment'):
                    gown_properly_worn = gown_match['gown_assessment'].get('is_properly_worn', False)
                    print(f"DEBUG: Gown properly worn: {gown_properly_worn}")
                    break
                else:
                    print("DEBUG: No gown_assessment found in match")
        else:
            print("DEBUG: No gown matches found")
    else:
        print("DEBUG: No gown detected")
    
    print(f"DEBUG: Final gown_detected: {gown_detected}, gown_properly_worn: {gown_properly_worn}")
    
    # Define PPE areas with coordinates (x, y, width, height)
    ppe_areas = {
        'hairnet': {'center': (5, 14.5), 'width': 1.5, 'height': 1.0, 'shape': 'ellipse'},
        'goggles': {'center': (5, 13.5), 'width': 1.2, 'height': 0.6, 'shape': 'rectangle'},
        'gown': {'center': (5, 10), 'width': 3, 'height': 4, 'shape': 'rectangle'},
        'buttons': {'center': (5, 9), 'width': 0.8, 'height': 1.5, 'shape': 'rectangle'},
        'left_shoe': {'center': (3.5, 2), 'width': 1, 'height': 1.5, 'shape': 'ellipse'},
        'right_shoe': {'center': (6.5, 2), 'width': 1, 'height': 1.5, 'shape': 'ellipse'}
    }
    
    # Draw human figure outline
    # Head
    head = Circle((5, 14), 1.2, fill=False, color='black', linewidth=2)
    ax.add_patch(head)
    
    # Body
    body = Rectangle((3.5, 8), 3, 6, fill=False, color='black', linewidth=2)
    ax.add_patch(body)
    
    # Arms
    left_arm = Rectangle((2, 9), 1.5, 4, fill=False, color='black', linewidth=2)
    right_arm = Rectangle((6.5, 9), 1.5, 4, fill=False, color='black', linewidth=2)
    ax.add_patch(left_arm)
    ax.add_patch(right_arm)
    
    # Legs
    left_leg = Rectangle((3.5, 2), 1.5, 6, fill=False, color='black', linewidth=2)
    right_leg = Rectangle((5, 2), 1.5, 6, fill=False, color='black', linewidth=2)
    ax.add_patch(left_leg)
    ax.add_patch(right_leg)
    
    # Draw PPE compliance areas
    compliance_status = {
        'hairnet': hairnet_detected,
        'goggles': goggles_detected,
        'gown': gown_detected,  # Gown area is green if gown is detected
        'buttons': gown_detected and gown_properly_worn,  # Buttons are green only if gown is properly worn
        'left_shoe': left_shoe_detected,
        'right_shoe': right_shoe_detected
    }
    
    for area_name, area_info in ppe_areas.items():
        center = area_info['center']
        width = area_info['width']
        height = area_info['height']
        shape = area_info['shape']
        
        # Determine color based on compliance
        is_compliant = compliance_status[area_name]
        color = GREEN if is_compliant else RED
        
        # Draw the area
        if shape == 'ellipse':
            area_patch = Ellipse(center, width, height, 
                               facecolor=color, alpha=0.6, edgecolor='black', linewidth=1)
        else:  # rectangle
            x = center[0] - width/2
            y = center[1] - height/2
            area_patch = Rectangle((x, y), width, height, 
                                 facecolor=color, alpha=0.6, edgecolor='black', linewidth=1)
        
        ax.add_patch(area_patch)
        
        # Add label
        ax.text(center[0], center[1], area_name.replace('_', ' ').title(), 
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add title
    ax.set_title('PPE Compliance Status', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=GREEN, label='Compliant'),
        mpatches.Patch(color=RED, label='Non-Compliant')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Calculate compliance percentage
    total_areas = len(compliance_status)
    compliant_areas = sum(compliance_status.values())
    compliance_percentage = (compliant_areas / total_areas) * 100
    
    # Add compliance summary
    summary_text = f"Compliance: {compliant_areas}/{total_areas} ({compliance_percentage:.0f}%)"
    ax.text(5, 0.5, summary_text, ha='center', va='center', 
           fontsize=12, fontweight='bold', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🥽 PPE Detection System</h1>', unsafe_allow_html=True)
    
    # Check if detector is available
    if not DETECTOR_AVAILABLE:
        st.error(f"❌ PPE detector not available: {import_error}")
        st.info("""
        **Setup Instructions:**
        1. Ensure you're running from the correct directory
        2. Check that `components/goggles/goggle_detection.py` exists
        3. Install required dependencies: `pip install opencv-python numpy segment-anything`
        """)
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # SAM checkpoint path
        sam_checkpoint = st.text_input(
            "SAM Checkpoint Path",
            value="sam_vit_h_4b8939.pth",
            help="Path to SAM model checkpoint file"
        )
        
        # Reference masks path
        reference_path = st.text_input(
            "Reference Masks Path", 
            value="reference_data/goggles",
            help="Path to reference binary masks directory"
        )
        
        # Similarity thresholds
        st.subheader("🔧 Detection Thresholds")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            similarity_threshold = st.slider(
                "Goggles & Shoes Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.90,
                step=0.05,
                help="Minimum similarity score for goggles and shoes detection"
            )
        
        with col2:
            cuff_similarity_threshold = st.slider(
                "Cuffs Threshold",
                min_value=0.3,
                max_value=0.8,
                value=0.50,
                step=0.05,
                help="Minimum similarity score for cuffs detection (typically lower than goggles/shoes)"
            )
        
        with col3:
            gown_similarity_threshold = st.slider(
                "Gowns Threshold",
                min_value=0.4,
                max_value=0.9,
                value=0.60,
                step=0.05,
                help="Minimum similarity score for gowns detection"
            )
        
        with col4:
            hairnet_similarity_threshold = st.slider(
                "Hairnet Threshold",
                min_value=0.4,
                max_value=0.9,
                value=0.75,
                step=0.05,
                help="Minimum similarity score for hairnet detection"
            )
        
        # Maximum image size for SAM processing
        max_image_size = st.selectbox(
            "Max Image Size (SAM Processing)",
            options=[512, 768, 1024, 1280, 1536],
            index=2,  # Default to 1024
            help="Maximum dimension for SAM processing. Smaller = faster but less accurate, Larger = slower but more accurate"
        )
        
        # Initialize detector button
        if st.button("🔄 Initialize Detector"):
            st.session_state.detector = None  # Force re-initialization
        
        st.markdown("---")
        st.header("📋 System Info")
        
        # Check paths
        sam_exists = os.path.exists(sam_checkpoint)
        ref_exists = os.path.exists(reference_path)
        shoes_ref_exists = os.path.exists("reference_data/shoes")
        
        st.write(f"**SAM Checkpoint**: {'✅' if sam_exists else '❌'}")
        st.write(f"**Goggles Reference**: {'✅' if ref_exists else '❌'}")
        st.write(f"**Shoes Reference**: {'✅' if shoes_ref_exists else '❌'}")
        
        if ref_exists:
            ref_files = list(Path(reference_path).glob("*.png")) + list(Path(reference_path).glob("*.jpg"))
            st.write(f"**Goggles References**: {len(ref_files)}")
        
        if shoes_ref_exists:
            left_shoes = list(Path("reference_data/shoes/left").glob("*.png")) + list(Path("reference_data/shoes/left").glob("*.jpg"))
            right_shoes = list(Path("reference_data/shoes/right").glob("*.png")) + list(Path("reference_data/shoes/right").glob("*.jpg"))
            total_shoes = len(left_shoes) + len(right_shoes)
            st.write(f"**Shoes References**: {total_shoes} (Left: {len(left_shoes)}, Right: {len(right_shoes)})")
        
        # Check cuff references
        cuff_ref_exists = Path("reference_data/cuffs").exists()
        st.write(f"**Cuffs Reference**: {'✅' if cuff_ref_exists else '❌'}")
        
        if cuff_ref_exists:
            left_cuffs = list(Path("reference_data/cuffs/left").glob("*.png")) + list(Path("reference_data/cuffs/left").glob("*.jpg"))
            right_cuffs = list(Path("reference_data/cuffs/right").glob("*.png")) + list(Path("reference_data/cuffs/right").glob("*.jpg"))
            total_cuffs = len(left_cuffs) + len(right_cuffs)
            st.write(f"**Cuffs References**: {total_cuffs} (Left: {len(left_cuffs)}, Right: {len(right_cuffs)})")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        **Detection Process:**
        1. Upload image with person wearing PPE
        2. SAM generates all possible masks
        3. Each mask compared with reference embeddings
        4. Similarity ≥ 90% → ✅ Green box (goggles) / Blue box (shoes)
        5. Similarity < 90% → No box shown
        6. No matches → "NO PPE DETECTED" message
        """)
    
    # Initialize detector (but don't block UI if it fails)
    detector_ready = False
    if 'detector' not in st.session_state or st.session_state.detector is None:
        if sam_exists and ref_exists:
            try:
                with st.spinner("Initializing PPE detector..."):
                    detector = GogglesDetector(
                        reference_path=reference_path,
                        sam_checkpoint=sam_checkpoint
                    )
                    detector.similarity_threshold = similarity_threshold
                    detector.cuff_similarity_threshold = cuff_similarity_threshold
                    detector.gown_similarity_threshold = gown_similarity_threshold
                    detector.hairnet_similarity_threshold = hairnet_similarity_threshold
                    st.session_state.detector = detector
                    detector_ready = True
                st.success("✅ Detector initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize detector: {e}")
                st.info("You can still upload images, but detection won't work until paths are fixed.")
                detector_ready = False
        else:
            st.warning("⚠️ Please check SAM checkpoint and reference paths in sidebar")
            st.info("You can still upload images, but detection won't work until paths are fixed.")
            detector_ready = False
    else:
        # Update thresholds if changed
        st.session_state.detector.similarity_threshold = similarity_threshold
        st.session_state.detector.cuff_similarity_threshold = cuff_similarity_threshold
        st.session_state.detector.gown_similarity_threshold = gown_similarity_threshold
        st.session_state.detector.hairnet_similarity_threshold = hairnet_similarity_threshold
        detector_ready = True
    
    detector = st.session_state.detector if detector_ready else None
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a person wearing (or not wearing) PPE"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🔍 Detect PPE", type="primary"):
                if not detector_ready or detector is None:
                    st.error("❌ Detector not initialized. Please fix the SAM checkpoint and reference paths in the sidebar.")
                    return
                
                # Save uploaded image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    temp_image_path = tmp_file.name
                
                try:
                    # Run detection with configured max size
                    with st.spinner("Running PPE detection..."):
                        result = detector.detect_goggles(temp_image_path, max_size=max_image_size)
                    
                    # Store results in session state
                    st.session_state.detection_result = result
                    st.session_state.temp_image_path = temp_image_path
                    
                    # Create and store visualization
                    with st.spinner("Creating visualization..."):
                        viz_path = detector.create_visualization(temp_image_path, result)
                        st.session_state.visualization_path = viz_path
                    
                    st.success("✅ Detection completed!")
                    
                except Exception as e:
                    st.error(f"❌ Detection failed: {e}")
                    # Cleanup
                    try:
                        os.unlink(temp_image_path)
                    except:
                        pass
    
    with col2:
        st.header("📊 Detection Results")
        
        if 'detection_result' in st.session_state:
            result = st.session_state.detection_result
            
            # Overall result
            if result['goggles_detected'] or result['shoes_detected'] or result.get('cuffs_detected', False):
                st.markdown('<div class="success-box"><h3>✅ PPE DETECTED</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box"><h3>❌ NO PPE DETECTED</h3></div>', unsafe_allow_html=True)
            
            # PPE Type specific results
            col_goggles, col_shoes, col_cuffs, col_gowns = st.columns(4)
            
            with col_goggles:
                st.markdown('<div class="ppe-type-box">', unsafe_allow_html=True)
                if result['goggles_detected']:
                    st.markdown('<h4>🥽 GOGGLES DETECTED</h4>', unsafe_allow_html=True)
                    st.write(f"**Matches**: {result['goggle_matches']}")
                    st.write(f"**Confidence**: {result['confidence']:.2f}")
                else:
                    st.markdown('<h4>🥽 NO GOGGLES</h4>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_shoes:
                st.markdown('<div class="ppe-type-box">', unsafe_allow_html=True)
                if result['shoes_detected']:
                    st.markdown('<h4>👟 SHOES DETECTED</h4>', unsafe_allow_html=True)
                    st.write(f"**Matches**: {result['shoe_matches']}")
                    st.write(f"**Confidence**: {result['shoe_confidence']:.2f}")
                    
                    # Show left/right shoe breakdown
                    left_shoes = [m for m in result.get('all_matches', []) if m.get('is_shoe') and m.get('shoe_side') == 'left']
                    right_shoes = [m for m in result.get('all_matches', []) if m.get('is_shoe') and m.get('shoe_side') == 'right']
                    
                    if left_shoes or right_shoes:
                        st.write("**Shoe Types:**")
                        if left_shoes:
                            st.write(f"  - Left Shoes: {len(left_shoes)}")
                        if right_shoes:
                            st.write(f"  - Right Shoes: {len(right_shoes)}")
                else:
                    st.markdown('<h4>👟 NO SHOES</h4>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_cuffs:
                st.markdown('<div class="ppe-type-box">', unsafe_allow_html=True)
                if result.get('cuffs_detected', False):
                    st.markdown('<h4>🧤 CUFFS DETECTED</h4>', unsafe_allow_html=True)
                    st.write(f"**Matches**: {result.get('cuff_matches', 0)}")
                    st.write(f"**Confidence**: {result.get('cuff_confidence', 0.0):.2f}")
                    
                    # Show left/right cuff breakdown
                    left_cuffs = [m for m in result.get('all_matches', []) if m.get('is_cuff') and m.get('cuff_side') == 'left']
                    right_cuffs = [m for m in result.get('all_matches', []) if m.get('is_cuff') and m.get('cuff_side') == 'right']
                    
                    if left_cuffs or right_cuffs:
                        st.write("**Cuff Types:**")
                        if left_cuffs:
                            st.write(f"  - Left Cuffs: {len(left_cuffs)}")
                        if right_cuffs:
                            st.write(f"  - Right Cuffs: {len(right_cuffs)}")
                else:
                    st.markdown('<h4>🧤 NO CUFFS</h4>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_gowns:
                st.markdown('<div class="ppe-type-box">', unsafe_allow_html=True)
                if result.get('gown_detected', False):
                    st.markdown('<h4>👗 GOWNS DETECTED</h4>', unsafe_allow_html=True)
                    st.write(f"**Matches**: {result.get('gown_matches', 0)}")
                    st.write(f"**Confidence**: {result.get('gown_confidence', 0.0):.2f}")
                else:
                    st.markdown('<h4>👗 NO GOWNS</h4>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Human Figure Compliance Visualization
            st.markdown("---")
            st.subheader("👤 PPE Compliance Status")
            
            # Create and display human figure
            human_fig = create_human_figure_compliance(result)
            st.pyplot(human_fig)
            
            # Key metrics
            col1_metrics, col2_metrics, col3_metrics = st.columns(3)
            
            with col1_metrics:
                st.metric(
                    "Total SAM Masks",
                    result['total_sam_masks']
                )
            
            with col2_metrics:
                st.metric(
                    "Goggle Matches",
                    result['goggle_matches']
                )
            
            with col3_metrics:
                st.metric(
                    "Shoe Matches",
                    result['shoe_matches']
                )
            
            # Processing info
            st.markdown("---")
            st.subheader("🔍 Processing Details")
            
            col1_info, col2_info = st.columns(2)
            
            with col1_info:
                st.write(f"**Processing Time**: {result['processing_time']:.2f}s")
                st.write(f"**Original Image**: {result['original_image_shape'][1]}×{result['original_image_shape'][0]}")
                st.write(f"**Processed Image**: {result['processed_image_shape'][1]}×{result['processed_image_shape'][0]}")
                if result.get('scale_factor', 1.0) != 1.0:
                    st.write(f"**Scale Factor**: {result['scale_factor']:.3f}")
                st.write(f"**Reference Masks**: {result['reference_masks_count']}")
            
            with col2_info:
                if result['best_match']:
                    best = result['best_match']
                    st.write(f"**Best Goggle Similarity**: {best['similarity']:.3f}")
                    st.write(f"**Best Goggle Area**: {best['area']} pixels")
                    st.write(f"**Reference Match**: {best['reference_match']['name'] if best['reference_match'] else 'None'}")
                
                if result['best_shoe_match']:
                    best_shoe = result['best_shoe_match']
                    st.write(f"**Best Shoe Similarity**: {best_shoe['similarity']:.3f}")
                    st.write(f"**Best Shoe Area**: {best_shoe['area']} pixels")
                    st.write(f"**Reference Match**: {best_shoe['reference_match']['name'] if best_shoe['reference_match'] else 'None'}")
            
            # Show visualization
            if 'visualization_path' in st.session_state:
                st.markdown("---")
                st.subheader("🎨 Detection Visualization")
                
                try:
                    viz_image = Image.open(st.session_state.visualization_path)
                    st.image(viz_image, caption="Detection Results (Green boxes = Goggles, Blue boxes = Shoes)", use_container_width=True)
                    
                    # Download buttons
                    col1_dl, col2_dl = st.columns(2)
                    
                    with col1_dl:
                        with open(st.session_state.visualization_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Visualization",
                                data=file,
                                file_name=f"ppe_detection_{uploaded_file.name}",
                                mime="image/jpeg"
                            )

                    with col2_dl:
                        # Save and download JSON results
                        json_path = detector.save_results(result)
                        with open(json_path, "r") as file:
                            st.download_button(
                                label="📄 Download Results JSON",
                                data=file.read(),
                                file_name=f"ppe_results_{uploaded_file.name.split('.')[0]}.json",
                                mime="application/json"
                            )
                
                except Exception as e:
                    st.error(f"Error displaying visualization: {e}")
            
            # Enhanced debug section - ALWAYS show after detection
            st.markdown("---")
            st.subheader("🔍 Debug: View All Masks & Comparisons")
            st.write("**Click the buttons below to see what SAM detected and how it compared with your reference masks:**")
            
            col1_debug, col2_debug = st.columns(2)
            
            with col1_debug:
                if st.button("💾 Save All Masks & Comparisons", type="primary"):
                    if 'temp_image_path' in st.session_state and detector_ready:
                        with st.spinner("Creating comparison visualizations..."):
                            comparison_dir = detector.save_all_masks_and_comparisons(st.session_state.temp_image_path, result)
                            st.session_state.comparison_dir = comparison_dir
                        st.success(f"✅ All masks and comparisons saved!")
                        st.info(f"📁 **Location**: `{comparison_dir}`")
                        st.info("""
                        **What's saved:**
                        - `all_masks/` - All {0} SAM masks (cropped binary)
                        - `comparisons/` - Side-by-side SAM vs Reference  
                        - `00_SUMMARY_top_matches.png` - Overview of top matches
                        """.format(result.get('total_sam_masks', 0)))
                    else:
                        st.error("No detection data available. Please run detection first.")
            
            with col2_debug:
                if st.button("💾 Save Simple Debug Masks"):
                    if detector_ready:
                        debug_dir = detector.save_debug_masks(result)
                        st.success(f"Debug masks saved to: {debug_dir}")
                        st.info("""
                        **Simple debug files:**
                        - `sam_cropped_mask_XXX.png` - Individual SAM masks
                        - `reference_masks/` - Reference binary masks
                        """)
                    else:
                        st.error("Detector not ready. Please check configuration.")
            
            # Detailed matches (expandable)
            with st.expander("🔍 Detailed Match Analysis"):
                if result['all_matches']:
                    for i, match in enumerate(result['all_matches'][:10]):  # Show top 10
                        if match['is_goggle']:
                            status = "✅ GOGGLE"
                        elif match['is_shoe']:
                            shoe_side = match.get('shoe_side', 'unknown')
                            status = f"✅ SHOE ({shoe_side.upper()})"
                        elif match['is_cuff']:
                            cuff_side = match.get('cuff_side', 'unknown')
                            status = f"✅ CUFF ({cuff_side.upper()})"
                        elif match['is_gown']:
                            status = "✅ GOWN"
                        else:
                            status = "❌ NO MATCH"
                        
                        st.write(f"**Match {i+1}**: {status}")
                        st.write(f"  - Similarity: {match['similarity']:.3f}")
                        st.write(f"  - Area: {match['area']} pixels")
                        st.write(f"  - SAM IoU: {match['predicted_iou']:.3f}")
                        st.write(f"  - Stability: {match['stability_score']:.3f}")
                        if match['is_shoe'] and match.get('shoe_side'):
                            st.write(f"  - Shoe Side: {match['shoe_side'].upper()}")
                        if match['is_cuff'] and match.get('cuff_side'):
                            st.write(f"  - Cuff Side: {match['cuff_side'].upper()}")
                        if match['reference_match']:
                            st.write(f"  - Best Reference: {match['reference_match']['name']}")
                        st.write("---")
                else:
                    st.write("No matches found")
        
        else:
            st.info("👆 Upload an image and click 'Detect PPE' to see results")
    
    # Cleanup section
    if st.sidebar.button("🧹 Clear Results"):
        # Clear session state
        for key in ['detection_result', 'temp_image_path', 'visualization_path']:
            if key in st.session_state:
                if key.endswith('_path'):
                    try:
                        os.unlink(st.session_state[key])
                    except:
                        pass
                del st.session_state[key]
        st.success("✅ Results cleared!")
        st.rerun()

if __name__ == "__main__":
    main()
