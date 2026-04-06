import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial import distance as scipy_distance
from scipy.stats import beta as beta_dist

# Voxel generation improvements:
# - Direct computation approach: only processes voxels that contain points (vs. full grid generation)
# - Inner cube filtering: uses Chebyshev distance with 80% core for better boundary detection
# - Voxel grid creation: uses combined grofund truth + projected points for comprehensive coverage
# - Voxel scoring: uses ground truth points only for accurate distance-based scoring
# - Memory efficient: O(num_points) vs O(total_possible_voxels)

# Global configuration
VOXEL_SIZE = 8  # Size of voxels for visualization and intersection calculation (mm)
INPUT_DOWNSAMPLE_SIZE = 2  # Voxel size for downsampling input point clouds (mm)
                              # Larger = faster but less detail, Smaller = slower but more detail
                              # Recommended range: 0.5 to 2.0 mm
expansion_rate = 3.0 # Rate at which the ellipsoid width is expanded to determine the number of layers

# Enhanced global voxel occupancy table with confidence tracking
# Maps (x,y,z) voxel coordinates to a dict: {rock_idx: {'score': float, 'confidence': float, 'view_count': int}}
voxel_occupancy = {}

# Store min_dist for each voxel for debugging
voxel_min_dist = {}

# Store which rock gives the highest score for each voxel (with confidence threshold)
voxel_best_rock = {}

# When True, RockSegment skips per-segment print (e.g. batch/test set this for faster, quiet loading).
# Why "loading" is slow: np.load() is fast. The cost is creating one RockSegment per (rock, view).
# Each RockSegment does: voxel_down_sample, outlier removal, _create_projected_points() (O(points×layers)
# with layer-by-layer erosion), _create_optimized_voxel_grid(), _calculate_enhanced_voxel_scores().
# With 50+ segments per scene that is a lot of work and I/O; set _quiet=True when used as a library.
_quiet = False
def _log(*args, **kwargs):
    if not _quiet:
        print(*args, **kwargs)

class RockSegment:
    def __init__(self, points, camera_transform, complete_rock_idx):
        """
        Initialize a rock segment with its points, camera position, and rock index.
        
        Args:
            points: numpy array of points (Nx3)
            camera_transform: 4x4 transformation matrix for camera position
            complete_rock_idx: index of the complete rock this segment belongs to
        """
        self.complete_rock_idx = complete_rock_idx
        self.camera_transform = camera_transform
        
        # Create and preprocess point cloud
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        
        # Preprocess points - VERY gentle preprocessing for small rock segments
        # Only downsample larger point clouds
        # if len(self.point_cloud.points) > 500:
        self.point_cloud = self.point_cloud.voxel_down_sample(voxel_size=INPUT_DOWNSAMPLE_SIZE)
        
        # Use very gentle outlier removal, only for larger segments
        if len(self.point_cloud.points) > 20:  # Only remove outliers if we have enough points
            nb_neighbors = max(2, min(10, len(self.point_cloud.points) // 10))  # At least 2, max 10 neighbors
            self.point_cloud, _ = self.point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=3.0)  # Very permissive
        
        # Store ground truth points
        self.ground_truth_points = np.asarray(self.point_cloud.points)
        
        # Check if we still have points after preprocessing
        self.is_valid = len(self.ground_truth_points) > 0
        if not self.is_valid:
            _log(f"Warning: Rock {complete_rock_idx} segment became empty after preprocessing, skipping...")
            self.projected_point_cloud = o3d.geometry.PointCloud()
            self.voxel_grid = None
            self.voxel_scores = {}
            return
        
        # Calculate projected dots
        self.projected_point_cloud = self._create_projected_points()
        
        # Calculate optimized voxel grid from both ground truth and projected points
        if len(self.projected_point_cloud.points) > 0:
            # Combine ground truth and projected points for voxel grid creation
            combined_points = np.vstack([self.ground_truth_points, np.asarray(self.projected_point_cloud.points)])
            _log(f"Rock {complete_rock_idx}: {len(self.ground_truth_points)} ground truth + {len(self.projected_point_cloud.points)} projected = {len(combined_points)} total points")
            self.voxel_grid = self._create_optimized_voxel_grid(combined_points)
        else:
            # Fallback to just ground truth points if no projected points
            _log(f"Rock {complete_rock_idx}: No projected points generated, using only ground truth points!")
            self.voxel_grid = self._create_optimized_voxel_grid(self.ground_truth_points)
            
        # Calculate enhanced voxel scores with directional awareness
        self.voxel_scores = {}  # Maps voxel position to {'score': float, 'confidence': float}
        if self.voxel_grid is not None:
            self._calculate_enhanced_voxel_scores()
    
    def _create_projected_points(self):
        """
        Create points behind rock segment using raycasting and layer-by-layer erosion.
        
        The projection direction is from the camera to the segment's error ellipsoid center.
        The ellipsoid center (3D mean of ground truth points) is projected onto the camera's
        view plane to determine the 2D center for erosion calculations.
        
        Erosion uses Euclidean distance (L2 norm) to create circular/elliptical patterns
        that naturally taper the volumetric envelope at segment boundaries.
        """
        # Extract camera position and orientation
        camera_pos = self.camera_transform[:3, 3]
        camera_rot = self.camera_transform[:3, :3]
        
        # Calculate view plane basis vectors (assuming camera looks along -Z)
        view_z = -camera_rot[:, 2]  # Forward direction (-Z in camera frame)
        view_x = camera_rot[:, 0]   # Right direction (X in camera frame)
        view_y = camera_rot[:, 1]   # Up direction (Y in camera frame)
        
        # Calculate error ellipsoid center first (3D mean of ground truth points)
        if self.ground_truth_points.shape[0] >= 2:
            # Compute covariance and eigenvalues
            mean = np.mean(self.ground_truth_points, axis=0)
            covariance = np.cov(self.ground_truth_points.T)
            eigval, eigvec = np.linalg.eig(covariance)
            order = eigval.argsort()[::-1]
            eigval = eigval[order]
            # radii[0] = length, radii[1] = width, radii[2] = thickness
            radii = np.sqrt(np.abs(eigval)) * 1.765  # alpha=1.75
            if len(radii) > 1:
                ellipsoid_width = radii[1]  # Use width (second largest radius)
            else:
                ellipsoid_width = radii[0]
            if ellipsoid_width == 0:
                ellipsoid_width = 1e-6
        else:
            mean = np.mean(self.ground_truth_points, axis=0)
            ellipsoid_width = 1e-6
        
        # Project ellipsoid center onto camera view plane
        ellipsoid_center_to_camera = mean - camera_pos
        center_x = np.dot(ellipsoid_center_to_camera, view_x)
        center_y = np.dot(ellipsoid_center_to_camera, view_y)
        
        # Calculate bounding box in view plane (for width/height reference)
        points_to_camera = self.ground_truth_points - camera_pos
        x_coords = np.dot(points_to_camera, view_x)
        y_coords = np.dot(points_to_camera, view_y)
        view_points = np.column_stack((x_coords, y_coords)).astype(np.float64)
        min_x, min_y = np.min(view_points, axis=0)
        max_x, max_y = np.max(view_points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        
        # Use ellipsoid width to determine number of layers (FIXED!)
        surface_diameter = ellipsoid_width * expansion_rate # Use ellipsoid width instead of 2D bounding box
        num_layers = int(surface_diameter / VOXEL_SIZE)
        erosion_rate = 1.0 / num_layers if num_layers > 0 else 1.0
        
        # Debug output
        _log(f"Rock {self.complete_rock_idx}: surface_diameter={surface_diameter:.2f}, VOXEL_SIZE={VOXEL_SIZE}, num_layers={num_layers}")
        _log(f"  ellipsoid_width={ellipsoid_width:.2f}, 2D_bbox_width={width:.2f}, 2D_bbox_height={height:.2f}")
        _log(f"  ellipsoid_center_3D={mean}, projected_2D=({center_x:.2f}, {center_y:.2f})")
        _log(f"  num_ground_truth_points={len(self.ground_truth_points)}")
        
        # Initialize output points
        volume_points = []
        volume_colors = []
        
        # Ensure we have at least some layers
        if num_layers == 0:
            num_layers = 1
            erosion_rate = 1.0
            _log(f"  Warning: Forced num_layers to 1")
        
        # For each point, project back into layers
        for point in self.ground_truth_points:
            point_to_camera = point - camera_pos
            x = np.dot(point_to_camera, view_x)
            y = np.dot(point_to_camera, view_y)
            
            # Calculate normalized position using Euclidean distance (circular erosion)
            norm_x = abs(x - center_x) / (width * 0.5) if width > 0 else 0
            norm_y = abs(y - center_y) / (height * 0.5) if height > 0 else 0
            # Scale by 1.1 to match rectangular erosion rate
            norm_dist = np.sqrt(norm_x**2 + norm_y**2) / 1.0
            
            # Generate points along the ray for each layer
            for layer in range(num_layers):
                survival_threshold = 1.0 - (layer + 1) * erosion_rate
                if norm_dist > survival_threshold:
                    continue
                    
                layer_distance = (layer + 1) * VOXEL_SIZE
                layer_point = point + view_z * layer_distance
                volume_points.append(layer_point)
                
                # Safe color calculation with division by zero protection
                if survival_threshold > 0:
                    color_factor = (1.0 - layer/num_layers) * (1.0 - norm_dist/survival_threshold)
                else:
                    color_factor = 1.0 - layer/num_layers
                volume_colors.append([0.1, 0.5, color_factor])
        
        _log(f"  Generated {len(volume_points)} projected points")
        
        # Create final volume point cloud
        volume_pcd = o3d.geometry.PointCloud()
        if volume_points:
            volume_pcd.points = o3d.utility.Vector3dVector(np.array(volume_points))
            volume_pcd.colors = o3d.utility.Vector3dVector(np.array(volume_colors))
        
        return volume_pcd
    
    def _create_optimized_voxel_grid(self, points):
        """
        Create an optimized voxel grid by directly computing only the voxels that contain points.
        This is much more efficient than creating a full grid and filtering.
        Uses inner-cube filtering with Chebyshev distance for better boundary detection.
        
        Args:
            points: numpy array of points (Nx3)
        Returns:
            optimized voxel grid with only non-empty voxels
        """
        if len(points) == 0:
            return None
        
        # Directly compute voxel centers for each point and apply inner-cube filter
        # Keep a voxel only if at least one point lies within the inner cube (80% of VOXEL_SIZE)
        core_half = 0.8 * VOXEL_SIZE / 2.0  # 40% of VOXEL_SIZE from center along each axis
        center_has_core_point = {}
        
        for point in points:
            # Find the voxel center that contains this point
            voxel_center = np.round(point / VOXEL_SIZE) * VOXEL_SIZE
            center_key = tuple(voxel_center)
            
            # Initialize flag
            if center_key not in center_has_core_point:
                center_has_core_point[center_key] = False
            
            # Check if the point is inside the inner cube (Chebyshev distance to center)
            if not center_has_core_point[center_key]:
                if np.max(np.abs(point - voxel_center)) <= core_half:
                    center_has_core_point[center_key] = True
        
        # Keep only voxel centers with at least one inner-cube point
        non_empty_voxel_centers = np.array([c for c, keep in center_has_core_point.items() if keep])
        
        _log(f"  Direct voxel computation with inner-cube filter: kept {len(non_empty_voxel_centers)} / {len(center_has_core_point)} voxels from {len(points)} points")
        
        if len(non_empty_voxel_centers) == 0:
            _log(f"  WARNING: No voxels with inner-cube points found! Consider reducing filter strictness.")
            return None
        
        # Create point cloud from non-empty voxel centers
        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(non_empty_voxel_centers)
        
        # Create voxel grid from these centers
        optimized_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            voxel_pcd, 
            voxel_size=VOXEL_SIZE
        )
        
        return optimized_voxel_grid
    
    def _calculate_enhanced_voxel_scores(self):
        """
        Calculate voxel confidence scores using Beta distribution.
        
        Scores are based on normalized distance to ground truth surface points.
        Beta(1.41, 1.94) models asymmetric uncertainty: rapid confidence increase
        near observed surface, gradual decay into volumetric interior.
        """
        # Score each voxel based on distance to ground truth points only
        scored_voxels = 0
        total_voxels = len(self.voxel_grid.get_voxels())
        
        _log(f"Rock {self.complete_rock_idx}: Using {len(self.ground_truth_points)} ground truth points for scoring")
        
        # First pass: calculate all minimum distances to find max_depth
        all_min_dists = []
        for voxel in self.voxel_grid.get_voxels():
            center = self.voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            dists = np.linalg.norm(self.ground_truth_points - center, axis=1)
            min_dist = np.min(dists)
            all_min_dists.append(min_dist)
        
        # Calculate max_depth as largest minimum distance times 1.01
        if all_min_dists:
            max_depth = max(all_min_dists) * 1.01
        else:
            max_depth = 1e-6
        
        # Second pass: calculate scores using the dynamic max_depth
        for voxel in self.voxel_grid.get_voxels():
            center = self.voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            # Distance-based score with custom parabolic curve
            dists = np.linalg.norm(self.ground_truth_points - center, axis=1)
            min_dist = np.min(dists)
            norm_dist = min_dist / max_depth
            
            # Beta distribution score: peaks at ~0.30, naturally zero at boundaries
            # Parameters: alpha=1.41, beta=1.94 based on uncertainty quantification principles
            # Reflects asymmetric uncertainty: rapid confidence increase near surface,
            # gradual decay into volumetric interior
            alpha, beta_param = 1.41, 1.94
            norm_dist_clipped = np.clip(norm_dist, 1e-10, 1 - 1e-10)
            distance_score = beta_dist.pdf(norm_dist_clipped, alpha, beta_param)
            # Normalize to max = 1.0
            peak_location = (alpha - 1) / (alpha + beta_param - 2)
            max_score = beta_dist.pdf(peak_location, alpha, beta_param)
            distance_score = distance_score / max_score
            
            # Simple confidence based on distance
            confidence = distance_score
            
            if distance_score > 0.05:  # Lower threshold to be more permissive
                pos_key = tuple(np.round(center / VOXEL_SIZE) * VOXEL_SIZE)
                self.voxel_scores[pos_key] = {
                    'score': distance_score,
                    'confidence': confidence,
                    'distance': min_dist
                }
                scored_voxels += 1
        _log(f"Rock {self.complete_rock_idx}: scored {scored_voxels}/{total_voxels} voxels (max_depth={max_depth:.2f})")

def simple_voxel_assignment(voxel_occupancy):
    """
    Simple voxel assignment - just assign to the rock with highest score.
    """
    voxel_best_rock = {}
    
    for pos_key, rock_scores in voxel_occupancy.items():
        if not rock_scores:
            continue
            
        # Get the rock with highest score
        best_rock_idx = max(rock_scores.items(), key=lambda x: x[1]['score'])[0]
        voxel_best_rock[pos_key] = best_rock_idx
    
    return voxel_best_rock

# Removed detect_and_preserve_gaps function - will be replaced with new edge detection methods 

def create_voxel_grid_from_points(points, voxel_size=VOXEL_SIZE):
    """
    Convert point cloud to voxel grid using Open3D's built-in function.
    Args:
        points: numpy array of points (Nx3)
        voxel_size: size of each voxel
    Returns:
        voxel_grid: Open3D voxel grid
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

def visualize_simple_results(rock_segments, voxel_best_rock, 
                              segment_to_rock, global_voxel_grid=None):
    """Visualize the simple voxel assignment results."""
    # Create visualizers
    vis1 = o3d.visualization.Visualizer()  # Original segments
    vis2 = o3d.visualization.Visualizer()  # Segments + projected dots (no voxels)

    vis1.create_window(window_name="Original Rock Segments", width=1280, height=720, left=0)
    vis2.create_window(window_name="Segments + Projected Dots", width=1280, height=720, left=1280)

    # Add coordinate frames
    global_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    global_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    vis1.add_geometry(global_frame1)
    vis2.add_geometry(global_frame2)

    # Generate colors for rocks
    np.random.seed(42)
    num_rocks = max(segment_to_rock) + 1 if segment_to_rock else len(rock_segments)
    rock_colors = {i: np.random.rand(3) * 0.5 + 0.5 for i in range(num_rocks)}

    # Store geometries to prevent garbage collection
    geometries = []
    
    # Extract camera positions from rock segments
    camera_positions = [seg.camera_transform for seg in rock_segments]

    # Visualize original segments
    for i, segment in enumerate(rock_segments):
        rock_idx = segment_to_rock[i] if segment_to_rock else i
        color = rock_colors.get(rock_idx, [0.5, 0.5, 0.5])

        segment_copy = o3d.geometry.PointCloud()
        segment_copy.points = o3d.utility.Vector3dVector(np.asarray(segment.point_cloud.points).copy())
        segment_copy.paint_uniform_color(color)
        vis1.add_geometry(segment_copy)
        geometries.append(segment_copy)

    # Add camera frames to vis1
    for camera_pos in camera_positions:
        camera_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        camera_frame1.transform(camera_pos)
        vis1.add_geometry(camera_frame1)
        geometries.append(camera_frame1)

    # Visualize segments and projected dots in vis2
    for i, segment in enumerate(rock_segments):
        rock_idx = segment_to_rock[i] if segment_to_rock else i
        color = rock_colors.get(rock_idx, [0.5, 0.5, 0.5])

        # Show original segment
        segment_copy = o3d.geometry.PointCloud()
        segment_copy.points = o3d.utility.Vector3dVector(np.asarray(segment.point_cloud.points).copy())
        segment_copy.paint_uniform_color(color)
        vis2.add_geometry(segment_copy)
        geometries.append(segment_copy)

        # Show projected dots (if available) - use same color as rock segments
        if hasattr(segment, 'projected_point_cloud') and len(segment.projected_point_cloud.points) > 0:
            projected_copy = o3d.geometry.PointCloud()
            projected_copy.points = o3d.utility.Vector3dVector(np.asarray(segment.projected_point_cloud.points).copy())
            projected_copy.paint_uniform_color(color)  # Same color as rock segment
            vis2.add_geometry(projected_copy)
            geometries.append(projected_copy)
            print(f"Added {len(segment.projected_point_cloud.points)} projected dots for rock {rock_idx} (same color)")
        else:
            print(f"No projected dots for rock {rock_idx}")

    # Add camera frames to vis2
    for camera_pos in camera_positions:
        camera_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        camera_frame2.transform(camera_pos)
        vis2.add_geometry(camera_frame2)
        geometries.append(camera_frame2)

    # Set rendering options
    for vis in [vis1, vis2]:
        opt = vis.get_render_option()
        opt.point_size = 2
        opt.background_color = np.array([1, 1, 1])
        opt.light_on = True
        opt.mesh_show_back_face = True
        opt.mesh_show_wireframe = True

    # Run visualizers
    while vis1.poll_events() and vis2.poll_events():
        vis1.update_renderer()
        vis2.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()

def save_simple_results(voxel_best_rock, segment_to_rock, 
                         output_filename, voxel_occupancy=None, voxel_to_segments=None):
    """Save simple voxel assignment results to NPZ file and CSV file with segment information."""
    import pandas as pd

    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Prepare assigned voxel data
    assigned_positions = []
    assigned_labels = []
    assigned_confidences = []
    assigned_segment_labels = []  # NEW: Track which segments contributed to each voxel
    
    for pos_key, rock_idx in voxel_best_rock.items():
        assigned_positions.append(pos_key)
        assigned_labels.append(rock_idx)
        
        # Get confidence - voxel must have a score since it was assigned
        if voxel_occupancy and pos_key in voxel_occupancy and rock_idx in voxel_occupancy[pos_key]:
            confidence = voxel_occupancy[pos_key][rock_idx]['confidence']
        else:
            # This should not happen - assigned voxels must have scores
            print(f"Warning: Assigned voxel at {pos_key} for rock {rock_idx} has no score data!")
            confidence = 0.0
        assigned_confidences.append(confidence)
        
        # NEW: Get segment information for this voxel - ENCODE ALL CONTRIBUTING SEGMENTS
        if voxel_to_segments and pos_key in voxel_to_segments:
            contributing_segments = voxel_to_segments[pos_key]
            if contributing_segments:
                # Encode ALL contributing segments as a sorted list converted to string
                # Example: segments {2, 5, 7} -> "2,5,7"
                segment_list = sorted(list(contributing_segments))
                segment_label = ",".join(map(str, segment_list))
            else:
                segment_label = "-1"  # Unknown segment
        else:
            segment_label = "-1"  # Unknown segment
        assigned_segment_labels.append(segment_label)
    
    # Convert to numpy arrays
    assigned_positions = np.array(assigned_positions) if assigned_positions else np.array([]).reshape(0, 3)
    assigned_labels = np.array(assigned_labels) if assigned_labels else np.array([])
    assigned_confidences = np.array(assigned_confidences) if assigned_confidences else np.array([])
    assigned_segment_labels = np.array(assigned_segment_labels) if assigned_segment_labels else np.array([])
    
    # Save to NPZ file with segment information
    np.savez(output_filename,
             voxel_positions=assigned_positions,
             voxel_labels=assigned_labels,
             voxel_confidences=assigned_confidences,
             voxel_segment_labels=assigned_segment_labels,  # NEW: Segment labels for training
             metadata={'voxel_size': VOXEL_SIZE, 'num_rocks': max(segment_to_rock) + 1, 'num_segments': len(segment_to_rock)})
    
    # Also save to CSV file with segment information
    csv_filename = output_filename.replace('.npz', '.csv')
    save_voxel_csv(voxel_best_rock, segment_to_rock, voxel_occupancy, csv_filename, voxel_to_segments)
    
    print(f"Saved simple results:")
    print(f"  - {len(assigned_positions)} voxels with scores")
    print(f"  - Avg score: {np.mean(assigned_confidences):.3f}")
    print(f"  - NPZ file: {output_filename}")
    print(f"  - CSV file: {csv_filename}")

def save_voxel_csv(voxel_best_rock, segment_to_rock, voxel_occupancy, csv_filename, voxel_to_segments=None):
    """
    Save voxel data to CSV file with segment information included.
    Format: x,y,z,score_rock_1,score_rock_2,...,score_sum,best_rock,segment_label
    """
    import pandas as pd
    
    # Determine maximum number of rocks
    num_rocks = max(segment_to_rock) + 1 if segment_to_rock else 1
    
    # Collect all voxel data
    csv_data = []
    
    # Process ALL voxels that have any rock scores (not just assigned ones)
    for pos_key, rock_scores_dict in voxel_occupancy.items():
        x, y, z = pos_key
        
        # Initialize score columns for all rocks
        rock_scores = [0.0] * num_rocks
        
        # Get scores for this voxel
        for rock_id, score_data in rock_scores_dict.items():
            if 0 <= rock_id < num_rocks:
                rock_scores[rock_id] = score_data.get('score', 0.0)
        
        # Calculate total score
        score_sum = sum(rock_scores)
        
        # Determine best rock (assigned rock or -1 if not assigned)
        if pos_key in voxel_best_rock:
            best_rock = voxel_best_rock[pos_key] + 1  # +1 for 1-based indexing
        else:
            best_rock = -1  # Not assigned to any rock
        
        # NEW: Get segment information for this voxel - ENCODE ALL CONTRIBUTING SEGMENTS
        if voxel_to_segments and pos_key in voxel_to_segments:
            contributing_segments = voxel_to_segments[pos_key]
            if contributing_segments:
                # Encode ALL contributing segments as a sorted list converted to string
                # Example: segments {2, 5, 7} -> "2,5,7"
                segment_list = sorted(list(contributing_segments))
                segment_label = ",".join(map(str, segment_list))
            else:
                segment_label = "-1"  # Unknown segment
        else:
            segment_label = "-1"  # No segment information available
        
        # Create row with segment information
        row = [x, y, z] + rock_scores + [score_sum, best_rock, segment_label]
        csv_data.append(row)
    
    # Create column names with segment label
    columns = ['x', 'y', 'z']
    for i in range(num_rocks):
        columns.append(f'score_rock_{i+1}')  # 1-based indexing for rock names
    columns.extend(['score_sum', 'best_rock', 'segment_label'])  # NEW: Added segment_label
    
    # Create DataFrame and save
    if csv_data:
        df = pd.DataFrame(csv_data, columns=columns)
        # Sort by coordinates for better readability
        df = df.sort_values(['x', 'y', 'z']).reset_index(drop=True)
        df.to_csv(csv_filename, index=False)
        
        # Calculate statistics
        assigned_voxels = len(df[df['best_rock'] > 0])
        unassigned_voxels = len(df[df['best_rock'] == -1])
        total_voxels = len(df)
        
        # NEW: Segment statistics - handle comma-separated segment lists
        voxels_with_segments = len(df[df['segment_label'] != '-1'])
        
        # Count unique segments from comma-separated lists
        all_segments = set()
        multi_segment_voxels = 0
        if voxels_with_segments > 0:
            for segment_str in df[df['segment_label'] != '-1']['segment_label']:
                segments = segment_str.split(',')
                if len(segments) > 1:
                    multi_segment_voxels += 1
                for seg in segments:
                    if seg.strip() and seg.strip() != '-1':
                        all_segments.add(int(seg.strip()))
        
        unique_segments = len(all_segments)
        
        print(f"CSV Details:")
        print(f"  - Total voxels with scores: {total_voxels}")
        print(f"  - Assigned voxels (best_rock > 0): {assigned_voxels}")
        print(f"  - Unassigned voxels (best_rock = -1): {unassigned_voxels}")
        print(f"  - Rock columns: {num_rocks} (score_rock_1 to score_rock_{num_rocks})")
        print(f"  - Voxels with segment info: {voxels_with_segments} ({voxels_with_segments/total_voxels*100:.1f}%)")
        print(f"  - Unique segments identified: {unique_segments}")
        print(f"  - Multi-segment voxels: {multi_segment_voxels} ({multi_segment_voxels/voxels_with_segments*100:.1f}% of segmented voxels)")
        
        # Show some examples of conflicts
        conflict_voxels = df[df['best_rock'] == -1]
        if len(conflict_voxels) > 0:
            print(f"  - Example unassigned voxel scores:")
            for i, row in conflict_voxels.head(3).iterrows():
                scores = [row[f'score_rock_{j+1}'] for j in range(num_rocks)]
                non_zero_scores = [(j+1, score) for j, score in enumerate(scores) if score > 0]
                segment_info = f", segments=[{row['segment_label']}]" if row['segment_label'] != '-1' else ""
                print(f"    Voxel ({row['x']:.1f}, {row['y']:.1f}, {row['z']:.1f}): {non_zero_scores}{segment_info}")
        
        # NEW: Show segment distribution examples including multi-segment voxels
        if voxels_with_segments > 0:
            print(f"  - Example segment assignments:")
            segment_examples = df[df['segment_label'] != '-1'].head(5)
            for i, row in segment_examples.iterrows():
                rock_info = f"rock={int(row['best_rock'])}" if row['best_rock'] != -1 else "unassigned"
                segments = row['segment_label']
                is_multi = ',' in segments
                multi_indicator = " (MULTI-SEGMENT)" if is_multi else ""
                print(f"    Voxel ({row['x']:.1f}, {row['y']:.1f}, {row['z']:.1f}): {rock_info}, segments=[{segments}]{multi_indicator}")
            
            # Show specifically multi-segment examples if any exist
            multi_examples = df[df['segment_label'].str.contains(',', na=False)]
            if len(multi_examples) > 0:
                print(f"  - Multi-segment voxel examples:")
                for i, row in multi_examples.head(3).iterrows():
                    rock_info = f"rock={int(row['best_rock'])}" if row['best_rock'] != -1 else "unassigned"
                    segments = row['segment_label'].split(',')
                    print(f"    Voxel ({row['x']:.1f}, {row['y']:.1f}, {row['z']:.1f}): {rock_info}, {len(segments)} segments=[{row['segment_label']}]")
    else:
        print(f"Warning: No voxel data to save to CSV: {csv_filename}")

def print_simple_analysis(voxel_occupancy, voxel_best_rock):
    """Print analysis of simple voxel assignment results."""
    print("\n=== SIMPLE ANALYSIS ===")
    
    # Count conflicts (voxels claimed by multiple rocks)
    conflict_voxels = 0
    high_conflict_voxels = 0
    total_voxels = len(voxel_occupancy)
    
    if total_voxels == 0:
        print("WARNING: No voxels generated! Check scoring parameters.")
        print("This might indicate the scoring is too strict or there's an issue with the data.")
        return
    
    for pos_key, rock_scores in voxel_occupancy.items():
        if len(rock_scores) > 1:
            conflict_voxels += 1
            scores = [data['score'] for data in rock_scores.values()]
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2 and sorted_scores[1] / sorted_scores[0] > 0.7:
                high_conflict_voxels += 1
    
    assigned_voxels = len(voxel_best_rock)
    
    print(f"Total voxels with scores: {total_voxels}")
    print(f"Conflict voxels (multiple rocks): {conflict_voxels} ({conflict_voxels/total_voxels*100:.1f}%)")
    print(f"High conflict voxels (scores close): {high_conflict_voxels} ({high_conflict_voxels/total_voxels*100:.1f}%)")
    print(f"All voxels assigned: {assigned_voxels} ({assigned_voxels/total_voxels*100:.1f}%)")
    
    # Calculate score distribution
    if voxel_occupancy:
        all_scores = []
        for pos_key in voxel_best_rock:
            if pos_key in voxel_occupancy:
                rock_idx = voxel_best_rock[pos_key]
                if rock_idx in voxel_occupancy[pos_key]:
                    score = voxel_occupancy[pos_key][rock_idx].get('score', 0)
                    all_scores.append(score)
        
        if all_scores:
            print(f"Average score: {np.mean(all_scores):.3f}")
            print(f"High score (>0.8): {sum(1 for s in all_scores if s > 0.8)} ({sum(1 for s in all_scores if s > 0.8)/len(all_scores)*100:.1f}%)")

def visualize_simple_heatmap_gui(voxel_best_rock, segment_to_rock, 
                                  voxel_size=VOXEL_SIZE, camera_positions=None, voxel_occupancy=None,
                                  app=None, auto_run=True):
    """
    Visualize the simple voxel assignment using Open3D's O3DVisualizer GUI.
    Different colors for each rock with score-based transparency.
    """
    import open3d as o3d
    import numpy as np

    # Initialize or reuse Open3D GUI application
    owns_app = False
    if app is None:
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        owns_app = True
    vis = o3d.visualization.O3DVisualizer("Simple Voxel Assignment", 1280, 720)

    # Turn off skybox
    vis.show_skybox(False)

    # Generate consistent colors for each rock
    np.random.seed(42)  # For reproducibility
    num_rocks = max(segment_to_rock) + 1 if segment_to_rock else 1
    rock_colors = {}
    for i in range(num_rocks):
        # Generate bright, distinct colors
        hue = i / num_rocks
        saturation = 0.8
        value = 0.9
        # Convert HSV to RGB
        if hue < 1/6:
            r, g, b = value, value * (1 - saturation * (1 - 6 * hue)), value * (1 - saturation)
        elif hue < 2/6:
            r, g, b = value * (1 - saturation * (6 * hue - 1)), value, value * (1 - saturation)
        elif hue < 3/6:
            r, g, b = value * (1 - saturation), value, value * (1 - saturation * (3 - 6 * hue))
        elif hue < 4/6:
            r, g, b = value * (1 - saturation), value * (1 - saturation * (6 * hue - 3)), value
        elif hue < 5/6:
            r, g, b = value * (1 - saturation * (5 - 6 * hue)), value * (1 - saturation), value
        else:
            r, g, b = value, value * (1 - saturation), value * (1 - saturation * (6 * hue - 5))
        rock_colors[i] = [r, g, b]  # RGB format, alpha will be calculated per voxel

    # Normalize scores for alpha mapping
    if voxel_occupancy:
        # Get accumulated scores (sum of all rock scores) for each voxel for alpha
        voxel_accumulated_scores = {}
        # Get maximum individual rock scores for color assignment (already handled by voxel_best_rock)
        
        for pos_key in voxel_best_rock.keys():
            if pos_key in voxel_occupancy:
                rock_scores = voxel_occupancy[pos_key]
                # Calculate accumulated score (sum of all rock scores for this voxel)
                accumulated_score = sum(rock_scores[rock_idx]['score'] for rock_idx in rock_scores.keys())
                voxel_accumulated_scores[pos_key] = accumulated_score
            else:
                voxel_accumulated_scores[pos_key] = 0.0
        
        # Find global min and max accumulated scores for normalization
        if voxel_accumulated_scores:
            max_accumulated = max(voxel_accumulated_scores.values())
            min_accumulated = min(voxel_accumulated_scores.values())
            accumulated_range = max(max_accumulated - min_accumulated, 1e-6)  # Avoid division by zero
        else:
            max_accumulated = 1.0
            min_accumulated = 0.0
            accumulated_range = 1.0
    else:
        voxel_accumulated_scores = {}
        max_accumulated = 1.0
        min_accumulated = 0.0
        accumulated_range = 1.0

    # Create and add assigned rock voxels
    for pos_key, rock_idx in voxel_best_rock.items():
        # Create a box for each voxel
        box = o3d.geometry.TriangleMesh.create_box(
            width=voxel_size, height=voxel_size, depth=voxel_size
        )
        
        # Position the box
        center = np.array(pos_key)
        box.translate(center - np.array([voxel_size/2, voxel_size/2, voxel_size/2]))
        box.compute_vertex_normals()  # This fixes the normals warning
        
        # Color by rock with score-based alpha
        base_color = rock_colors.get(rock_idx, [0.5, 0.5, 0.5])
        
        # Calculate alpha based on accumulated score (sum of all rock scores)
        if pos_key in voxel_accumulated_scores:
            accumulated_score = voxel_accumulated_scores[pos_key]
            # Map accumulated score to alpha: higher accumulated score = less transparent
            # Alpha in [0.1, 0.95]
            alpha = 0.1 + 0.85 * ((accumulated_score - min_accumulated) / accumulated_range)
        else:
            alpha = 0.8  # Default alpha if no score available
        
        color = base_color + [alpha]  # RGBA format
        
        # Set up material using the working approach from original
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"  # This is the key!
        material.base_color = color
        vis.add_geometry(f"rock_voxel_{pos_key}", box, material)

    # Add a global coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    material_frame = o3d.visualization.rendering.MaterialRecord()
    material_frame.shader = "defaultLit"
    vis.add_geometry("frame", frame, material_frame)

    # Add a coordinate frame for each camera position if provided
    if camera_positions is not None:
        for i, camera_pos in enumerate(camera_positions):
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
            cam_frame.transform(camera_pos)
            cam_material = o3d.visualization.rendering.MaterialRecord()
            cam_material.shader = "defaultLit"
            vis.add_geometry(f"camera_frame_{i}", cam_frame, cam_material)

    # Reset camera after all geometry is loaded
    vis.reset_camera_to_default()

    # Show statistics in the title
    assigned_count = len(voxel_best_rock)
    
    print(f"\n=== SIMPLE VISUALIZATION STATS ===")
    print(f"Voxels with scores: {assigned_count}")
    print(f"Total rocks: {num_rocks}")
    print(f"Colors: Each rock has a unique color")
    print(f"Alpha: Based on accumulated score (sum of all rock scores) with transparency range [0.10, 0.95]")
    
    # Show accumulated score statistics
    if voxel_occupancy and voxel_accumulated_scores:
        accumulated_scores = list(voxel_accumulated_scores.values())
        print(f"Accumulated scores: min={min(accumulated_scores):.3f}, max={max(accumulated_scores):.3f}, avg={np.mean(accumulated_scores):.3f}")

    # Show the visualizer window
    app.add_window(vis)
    if auto_run and owns_app:
        app.run()
    return vis

def visualize_gray_voxels_gui(voxel_best_rock, segment_to_rock, 
                              voxel_size=VOXEL_SIZE, camera_positions=None, voxel_occupancy=None,
                              app=None, auto_run=True):
    """
    Visualize all voxels in gray using Open3D's GUI with transparency.
    All voxels are shown in uniform gray color with score-based transparency.
    Includes a slider to filter voxels by score threshold.
    """
    import open3d as o3d
    import open3d.visualization.gui as gui  # type: ignore[import-untyped]
    import numpy as np

    # Get individual voxel scores (best rock score for each voxel)
    voxel_individual_scores = {}
    for pos_key, rock_idx in voxel_best_rock.items():
        if voxel_occupancy and pos_key in voxel_occupancy and rock_idx in voxel_occupancy[pos_key]:
            voxel_individual_scores[pos_key] = voxel_occupancy[pos_key][rock_idx]['score']
        else:
            voxel_individual_scores[pos_key] = 0.0
    
    # Normalize scores for alpha mapping
    if voxel_occupancy:
        # Get accumulated scores (sum of all rock scores) for each voxel for alpha
        voxel_accumulated_scores = {}
        
        for pos_key in voxel_best_rock.keys():
            if pos_key in voxel_occupancy:
                rock_scores = voxel_occupancy[pos_key]
                # Calculate accumulated score (sum of all rock scores for this voxel)
                accumulated_score = sum(rock_scores[rock_idx]['score'] for rock_idx in rock_scores.keys())
                voxel_accumulated_scores[pos_key] = accumulated_score
            else:
                voxel_accumulated_scores[pos_key] = 0.0
        
        # Find global min and max accumulated scores for normalization
        if voxel_accumulated_scores:
            max_accumulated = max(voxel_accumulated_scores.values())
            min_accumulated = min(voxel_accumulated_scores.values())
            accumulated_range = max(max_accumulated - min_accumulated, 1e-6)  # Avoid division by zero
        else:
            max_accumulated = 1.0
            min_accumulated = 0.0
            accumulated_range = 1.0
    else:
        voxel_accumulated_scores = {}
        max_accumulated = 1.0
        min_accumulated = 0.0
        accumulated_range = 1.0
    
    # Find min and max individual scores for slider range
    if voxel_individual_scores:
        min_score = min(voxel_individual_scores.values())
        max_score = max(voxel_individual_scores.values())
    else:
        min_score = 0.0
        max_score = 1.0

    # Initialize Open3D GUI application (or reuse caller's instance)
    owns_app = False
    if app is None:
        app = gui.Application.instance
        app.initialize()
        owns_app = True
    
    window = app.create_window("Gray Voxels with Score Filter", 1280, 800)
    em = window.theme.font_size
    
    # Control panel at the top with margins
    panel = gui.Horiz(0, gui.Margins(0.5*em, 0.5*em, 0.5*em, 0.5*em))
    panel.add_fixed(em)
    
    # Add slider label
    slider_label = gui.Label("Score Threshold: ")
    panel.add_child(slider_label)
    
    # Add slider
    threshold_slider = gui.Slider(gui.Slider.DOUBLE)
    threshold_slider.set_limits(min_score, max_score)
    threshold_slider.double_value = min_score
    panel.add_child(threshold_slider)
    panel.add_fixed(em)
    
    # Add threshold value label
    threshold_value_label = gui.Label(f"{min_score:.3f}")
    panel.add_child(threshold_value_label)
    panel.add_fixed(em)
    
    # Add voxel count label
    voxel_count_label = gui.Label(f"Showing: {len(voxel_best_rock)}/{len(voxel_best_rock)}")
    panel.add_child(voxel_count_label)
    panel.add_stretch()
    
    # Add score range info
    range_label = gui.Label(f"Range: [{min_score:.3f}, {max_score:.3f}]")
    panel.add_child(range_label)
    panel.add_fixed(em)
    # Create scene widget
    scene = gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    scene.scene.set_background([1, 1, 1, 1])
    
    # Add widgets directly to the window and control layout manually
    window.add_child(panel)
    window.add_child(scene)
    
    def on_layout(layout_context):
        content_rect = window.content_rect
        panel_height = int(2.5 * em)
        panel.frame = gui.Rect(content_rect.x,
                               content_rect.y,
                               content_rect.width,
                               panel_height)
        scene.frame = gui.Rect(content_rect.x,
                               content_rect.y + panel_height,
                               content_rect.width,
                               max(0, content_rect.height - panel_height))
    window.set_on_layout(on_layout)
    
    # Function to update voxels based on threshold
    def update_voxels(threshold):
        # Clear existing voxels
        scene.scene.clear_geometry()
        
        # Add coordinate frames
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        mat_frame = o3d.visualization.rendering.MaterialRecord()
        mat_frame.shader = "defaultLit"
        scene.scene.add_geometry("frame", frame, mat_frame)
        
        if camera_positions is not None:
            for i, camera_pos in enumerate(camera_positions):
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
                cam_frame.transform(camera_pos)
                cam_mat = o3d.visualization.rendering.MaterialRecord()
                cam_mat.shader = "defaultLit"
                scene.scene.add_geometry(f"camera_frame_{i}", cam_frame, cam_mat)
        
        # Add voxels that pass threshold
        voxel_count = 0
        for pos_key, rock_idx in voxel_best_rock.items():
            # Check if voxel score is above threshold
            if pos_key in voxel_individual_scores:
                if voxel_individual_scores[pos_key] < threshold:
                    continue
            
            voxel_count += 1
            
            # Create a box for each voxel
            box = o3d.geometry.TriangleMesh.create_box(
                width=voxel_size, height=voxel_size, depth=voxel_size
            )
            
            # Position the box
            center = np.array(pos_key)
            box.translate(center - np.array([voxel_size/2, voxel_size/2, voxel_size/2]))
            box.compute_vertex_normals()
            
            # Gray color with score-based alpha
            gray_color = [0.5, 0.5, 0.5]
            
            # Calculate alpha based on accumulated score
            if pos_key in voxel_accumulated_scores:
                accumulated_score = voxel_accumulated_scores[pos_key]
                alpha = 0.1 + 0.85 * ((accumulated_score - min_accumulated) / accumulated_range)
            else:
                alpha = 0.8
            
            color = gray_color + [alpha]
            
            # Set up material
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultLitTransparency"
            material.base_color = color
            scene.scene.add_geometry(f"gray_voxel_{pos_key}", box, material)
        
        # Update count label
        voxel_count_label.text = f"Showing: {voxel_count}/{len(voxel_best_rock)}"
    
    # Callback for slider change
    def on_slider_changed(value):
        threshold_value_label.text = f"{value:.3f}"
        update_voxels(value)
    
    threshold_slider.set_on_value_changed(on_slider_changed)
    
    # Initial update with minimum threshold (show all voxels)
    update_voxels(min_score)
    
    # Setup bounds and camera
    scene.setup_camera(60, scene.scene.bounding_box, scene.scene.bounding_box.get_center())
    
    # Show statistics
    assigned_count = len(voxel_best_rock)
    print(f"\n=== GRAY VOXELS VISUALIZATION WITH THRESHOLD FILTER ===")
    print(f"Total gray voxels: {assigned_count}")
    print(f"Score range: [{min_score:.3f}, {max_score:.3f}]")
    print(f"Color: Uniform gray for all voxels")
    print(f"Alpha: Based on accumulated score with transparency range [0.10, 0.95]")
    print(f"Use slider to filter voxels by score threshold")
    
    if voxel_occupancy and voxel_accumulated_scores:
        accumulated_scores = list(voxel_accumulated_scores.values())
        print(f"Accumulated scores: min={min(accumulated_scores):.3f}, max={max(accumulated_scores):.3f}, avg={np.mean(accumulated_scores):.3f}")

    # Run application if this function owns the GUI lifecycle
    if auto_run and owns_app:
        app.run()
    return window

def _default_rock_npz_dir() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str((repo_root / "data" / "in_segments").resolve())


def main():
    # Load rock segment data from the npz file
    npz_name = '1_stacked_segment.npz'
    data_dir = os.environ.get("ROCK_NPZ_DATA_DIR", _default_rock_npz_dir())
    data_path = os.path.join(data_dir, npz_name)
    data = np.load(data_path, allow_pickle=True)

    # Support both new format (cameras + cam_idx_list) and old format (camera_angles)
    rock_pcd_list = data['rock_pcd_list']
    if 'cameras' in data.files:
        cameras  = data['cameras']       # (N_cams, 4, 4)
        cam_ids  = data['cam_idx_list']  # (N_rocks,) of lists of int
        raw_segs = [
            (rock_idx, pts, np.array(cameras[int(cam_idx)], dtype=np.float64))
            for rock_idx, (pcd_views, view_ids) in enumerate(zip(rock_pcd_list, cam_ids))
            for pts, cam_idx in zip(pcd_views, view_ids)
        ]
    else:
        camera_angles = data['camera_angles']
        raw_segs = [
            (rock_idx, pts, np.array(transform, dtype=np.float64))
            for rock_idx, (pcd_views, cam_views) in enumerate(zip(rock_pcd_list, camera_angles))
            for pts, transform in zip(pcd_views, cam_views)
        ]

    # Create RockSegment objects for each segment
    rock_segments = []
    skipped_segments = 0
    for rock_idx, points, transform in raw_segs:
        segment = RockSegment(points, transform, rock_idx)
        if segment.is_valid:
            rock_segments.append(segment)
        else:
            skipped_segments += 1

    print(f"Created {len(rock_segments)} valid rock segments")
    if skipped_segments > 0:
        print(f"Skipped {skipped_segments} invalid segments (too few points after preprocessing)")
    print(f"\n=== CONFIGURATION ===")
    print(f"VOXEL_SIZE: {VOXEL_SIZE} mm (output voxel grid)")
    print(f"INPUT_DOWNSAMPLE_SIZE: {INPUT_DOWNSAMPLE_SIZE} mm (input point cloud downsampling)")
    print(f"=====================\n")

    # Clear global dictionaries
    voxel_occupancy.clear()
    voxel_min_dist.clear()
    voxel_best_rock.clear()

    # NEW: Track which segments contribute to each voxel for segment-aware training
    voxel_to_segments = {}  # Maps voxel position to set of segment IDs that contributed to it
    
    # Combine enhanced voxel scores from all segments
    all_projected_points = []
    for segment_idx, segment in enumerate(rock_segments):
        # Add projected points to global list
        if len(segment.projected_point_cloud.points) > 0:
            all_projected_points.extend(np.asarray(segment.projected_point_cloud.points))
        
        # Add enhanced scores to global voxel occupancy
        for pos_key, score_data in segment.voxel_scores.items():
            # Track which segment contributed to this voxel
            if pos_key not in voxel_to_segments:
                voxel_to_segments[pos_key] = set()
            voxel_to_segments[pos_key].add(segment_idx)  # Add this segment's ID
            
            if pos_key not in voxel_occupancy:
                voxel_occupancy[pos_key] = {}
            if segment.complete_rock_idx not in voxel_occupancy[pos_key]:
                voxel_occupancy[pos_key][segment.complete_rock_idx] = score_data.copy()
                voxel_occupancy[pos_key][segment.complete_rock_idx]['view_count'] = 1
            else:
                # Accumulate scores from multiple views
                existing = voxel_occupancy[pos_key][segment.complete_rock_idx]
                existing['score'] += score_data['score']
                existing['confidence'] = max(existing['confidence'], score_data['confidence'])
                existing['view_count'] += 1

    print(f"Total voxels before filtering: {len(voxel_occupancy)}")
    print(f"Total segments tracked: {len(rock_segments)}")
    
    # Print segment statistics for debugging
    segment_contributions = {}
    for pos_key, segments in voxel_to_segments.items():
        num_segments = len(segments)
        if num_segments not in segment_contributions:
            segment_contributions[num_segments] = 0
        segment_contributions[num_segments] += 1
    
    print(f"Segment contribution statistics:")
    for num_segs, count in sorted(segment_contributions.items()):
        print(f"  {count} voxels contributed by {num_segs} segment(s)")

    # Apply simple voxel assignment - just highest score wins
    simple_voxel_best_rock = simple_voxel_assignment(voxel_occupancy)

    # Create global voxel grid from all projected points
    if all_projected_points:
        all_projected_points = np.array(all_projected_points)
        global_voxel_grid = create_voxel_grid_from_points(all_projected_points)
    else:
        global_voxel_grid = None

    # Print analysis
    print_simple_analysis(voxel_occupancy, simple_voxel_best_rock)

    # Prepare visualization data
    segments = [seg.point_cloud for seg in rock_segments]
    camera_positions = [seg.camera_transform for seg in rock_segments]
    segment_to_rock = [seg.complete_rock_idx for seg in rock_segments]

    # Visualize simple results
    if global_voxel_grid is not None:
        visualize_simple_results(
            rock_segments, 
            simple_voxel_best_rock, 
            segment_to_rock,
            global_voxel_grid
        )

    # Save simple results with segment information
    output_filename = f'voxel_npz/{npz_name}_labels.npz'
    save_simple_results(
        simple_voxel_best_rock, 
        segment_to_rock,
        output_filename,
        voxel_occupancy,
        voxel_to_segments  # NEW: Pass segment tracking information
    )

    # Only visualize if we have voxels
    if len(voxel_occupancy) > 0:
        gui_app = o3d.visualization.gui.Application.instance
        gui_app.initialize()
        visualize_simple_heatmap_gui(
            simple_voxel_best_rock,
            segment_to_rock,
            VOXEL_SIZE,
            camera_positions,
            voxel_occupancy,
            app=gui_app,
            auto_run=False
        )
        visualize_gray_voxels_gui(
            simple_voxel_best_rock,
            segment_to_rock,
            VOXEL_SIZE,
            camera_positions,
            voxel_occupancy,
            app=gui_app,
            auto_run=False
        )
        gui_app.run()


    else:
        print("Skipping visualization - no voxels generated")

    print(f"\n=== SUMMARY ===")
    print(f"Simple voxel assignment completed:")
    print(f"1. Distance-based scoring only")
    print(f"2. All voxels with scores included")
    print(f"3. Highest score wins assignment")
    print(f"4. Ready for new edge detection implementation")

if __name__ == "__main__":
    main() 