#!/usr/bin/env python3
"""
Standalone script to evaluate chamfer distances between ground truth and predictions.
Generates a LaTeX table with results organized by category.
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

# Import required functions from the main metric script
from artvip_metric import (
    evaluate_chamfer_distance,
    load_vertices,
    extract_mesh_vertices_from_urdf,
    find_best_prediction,
    calculate_bounding_sphere_scale
)


def find_matching_objects(gt_dir, result_dir, debug=False):
    """
    Find objects that exist in both ground truth and result directories.
    
    Args:
        gt_dir: Path to processed ground truth directory
        result_dir: Path to results directory
        debug: Enable debug output
    
    Returns:
        List of matching objects with their paths
    """
    matching_objects = []
    
    if not os.path.exists(gt_dir):
        print(f"Ground truth directory not found: {gt_dir}")
        return []
    
    if not os.path.exists(result_dir):
        print(f"Results directory not found: {result_dir}")
        return []
    
    # Walk through processed GT directory
    for category_path in Path(gt_dir).iterdir():
        if not category_path.is_dir():
            continue
            
        category = category_path.name
        
        for object_dir in category_path.iterdir():
            if not object_dir.is_dir():
                continue
                
            for object_number_dir in object_dir.iterdir():
                if not object_number_dir.is_dir():
                    continue
                
                object_number = object_number_dir.name
                gt_path = str(object_number_dir)
                
                # Check if vertices file exists in GT
                vertices_file = os.path.join(gt_path, "vertices.json")
                if not os.path.exists(vertices_file):
                    if debug:
                        print(f"Skipping {object_number}: no vertices.json in GT")
                    continue
                
                # Look for corresponding result
                result_pattern = f"artvip_{object_number}_multi-view"
                result_path = os.path.join(result_dir, result_pattern)
                
                if os.path.exists(result_path):
                    matching_objects.append({
                        "object_number": object_number,
                        "category": category,
                        "object_name": object_dir.name,
                        "gt_path": gt_path,
                        "result_path": result_path
                    })
                elif debug:
                    print(f"No result found for {object_number} at {result_path}")
    
    if debug:
        print(f"Found {len(matching_objects)} matching objects")
    
    return matching_objects


def evaluate_all_chamfer_distances(matching_objects, use_scaling=False, use_surface_samples=False, debug=False):
    """
    Evaluate chamfer distances for all matching objects.
    
    Args:
        matching_objects: List of objects with GT and result paths
        use_scaling: Whether to apply bounding sphere scaling
        use_surface_samples: Whether to use surface-sampled vertices
        debug: Enable debug output
    
    Returns:
        Dictionary mapping object_number to (distance, category, object_name)
    """
    results = {}
    
    for i, obj in enumerate(matching_objects):
        object_number = obj["object_number"]
        category = obj["category"]
        object_name = obj["object_name"]
        gt_path = obj["gt_path"]
        result_path = obj["result_path"]
        
        print(f"[{i+1}/{len(matching_objects)}] Evaluating {category}/{object_number}")
        
        # Find the best prediction URDF
        urdf_path, iter_num, seed_num = find_best_prediction(result_path, debug=debug)
        
        if not urdf_path:
            print(f"  No valid URDF found for {object_number}")
            continue
        
        if debug:
            print(f"  Using URDF: {urdf_path} (iter {iter_num}, seed {seed_num})")
        
        # Calculate scaling factor if requested
        scaling_factor = 1.0
        if use_scaling:
            vertices_file = "vertices_surface.json" if use_surface_samples else "vertices.json"
            gt_vertices_path = os.path.join(gt_path, vertices_file)
            
            gt_vertices = load_vertices(gt_vertices_path)
            pred_vertices = extract_mesh_vertices_from_urdf(urdf_path)
            
            if gt_vertices is not None and pred_vertices is not None:
                scaling_factor = calculate_bounding_sphere_scale(pred_vertices, gt_vertices)
                if debug:
                    print(f"  Applied scaling factor: {scaling_factor:.4f}")
        
        # Evaluate chamfer distance
        chamfer_dist = evaluate_chamfer_distance(
            gt_path, urdf_path, 
            use_surface_samples=use_surface_samples, 
            debug=debug, 
            scale_pred=scaling_factor
        )
        
        if chamfer_dist != float('inf'):
            results[object_number] = (chamfer_dist, category, object_name)
            scale_note = " (scaled)" if use_scaling and scaling_factor != 1.0 else ""
            print(f"  Chamfer distance{scale_note}: {chamfer_dist:.4f}")
        else:
            print(f"  Failed to compute chamfer distance")
    
    return results


def generate_chamfer_latex_table(chamfer_results, output_path, method_name="Method"):
    """
    Generate a LaTeX table for chamfer distance results.
    
    Args:
        chamfer_results: Dictionary mapping object_number to (distance, category, object_name)
        output_path: Path to save the LaTeX table
        method_name: Name of the method for the table caption
    """
    if not chamfer_results:
        print("No results to generate table")
        return
    
    # Group results by category
    by_category = defaultdict(list)
    for obj_id, (distance, category, object_name) in chamfer_results.items():
        by_category[category].append(distance)
    
    # Calculate statistics per category
    category_stats = {}
    for category, distances in by_category.items():
        category_stats[category] = {
            'count': len(distances),
            'mean': np.mean(distances),
            'std': np.std(distances),
            'median': np.median(distances),
            'min': np.min(distances),
            'max': np.max(distances)
        }
    
    # Calculate overall statistics
    all_distances = [dist for distances in by_category.values() for dist in distances]
    overall_stats = {
        'count': len(all_distances),
        'mean': np.mean(all_distances),
        'std': np.std(all_distances),
        'median': np.median(all_distances),
        'min': np.min(all_distances),
        'max': np.max(all_distances)
    }
    
    # Generate LaTeX table
    header = f"""\\begin{{table*}}[t]
\\centering
\\caption{{{method_name}: Chamfer Distance Results}}
\\label{{tab:chamfer_distance_results}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Category}} & \\textbf{{Count}} & \\textbf{{Mean}} & \\textbf{{Std}} & \\textbf{{Median}} & \\textbf{{Min}} & \\textbf{{Max}} \\\\
\\midrule
"""
    
    footer = """\\bottomrule
\\end{tabular}%
}
\\end{table*}
"""
    
    body = ""
    
    # Add category rows (sorted by mean distance)
    sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['mean'])
    for category, stats in sorted_categories:
        category_name = category.replace('_', ' ').title()
        body += (f"{category_name} & "
                f"{stats['count']} & "
                f"{stats['mean']:.4f} & "
                f"{stats['std']:.4f} & "
                f"{stats['median']:.4f} & "
                f"{stats['min']:.4f} & "
                f"{stats['max']:.4f} \\\\\n")
    
    # Add overall row
    body += "\\midrule\n"
    body += (f"\\textbf{{Overall}} & "
            f"\\textbf{{{overall_stats['count']}}} & "
            f"\\textbf{{{overall_stats['mean']:.4f}}} & "
            f"\\textbf{{{overall_stats['std']:.4f}}} & "
            f"\\textbf{{{overall_stats['median']:.4f}}} & "
            f"\\textbf{{{overall_stats['min']:.4f}}} & "
            f"\\textbf{{{overall_stats['max']:.4f}}} \\\\\n")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(header + body + footer)
    
    print(f"LaTeX table saved to {output_path}")
    
    # Also print summary to console
    print(f"\n=== {method_name} Chamfer Distance Summary ===")
    print(f"{'Category':<15} {'Count':<8} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print("-" * 85)
    
    for category, stats in sorted_categories:
        print(f"{category:<15} {stats['count']:<8} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
              f"{stats['median']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f}")
    
    print("-" * 85)
    print(f"{'Overall':<15} {overall_stats['count']:<8} {overall_stats['mean']:<10.4f} {overall_stats['std']:<10.4f} "
          f"{overall_stats['median']:<10.4f} {overall_stats['min']:<10.4f} {overall_stats['max']:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate chamfer distances and generate LaTeX table')
    parser.add_argument('result_dir', help='Path to results directory')
    parser.add_argument('--gt_dir', default='datasets/ArtVIP/processed', 
                       help='Path to processed ground truth directory')
    parser.add_argument('--output', default='chamfer_results.tex',
                       help='Output path for LaTeX table')
    parser.add_argument('--method_name', default='Method',
                       help='Name of the method for table caption')
    parser.add_argument('--use_scaling', action='store_true',
                       help='Apply bounding sphere scaling to predictions')
    parser.add_argument('--use_surface_samples', action='store_true',
                       help='Use surface-sampled vertices instead of full mesh')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    print(f"Evaluating chamfer distances...")
    print(f"GT directory: {args.gt_dir}")
    print(f"Results directory: {args.result_dir}")
    print(f"Scaling: {'Enabled' if args.use_scaling else 'Disabled'}")
    print(f"Surface samples: {'Enabled' if args.use_surface_samples else 'Disabled'}")
    
    # Find matching objects
    matching_objects = find_matching_objects(args.gt_dir, args.result_dir, debug=args.debug)
    
    if not matching_objects:
        print("No matching objects found. Exiting.")
        return
    
    print(f"\nFound {len(matching_objects)} matching objects")
    
    # Evaluate chamfer distances
    chamfer_results = evaluate_all_chamfer_distances(
        matching_objects, 
        use_scaling=args.use_scaling,
        use_surface_samples=args.use_surface_samples,
        debug=args.debug
    )
    
    if not chamfer_results:
        print("No successful evaluations. Exiting.")
        return
    
    print(f"\nSuccessfully evaluated {len(chamfer_results)} objects")
    
    # Generate LaTeX table
    generate_chamfer_latex_table(chamfer_results, args.output, args.method_name)
    
    # Save raw results as JSON for reference
    json_output = args.output.replace('.tex', '_raw.json')
    with open(json_output, 'w') as f:
        json.dump({k: {'distance': float(v[0]), 'category': v[1], 'object_name': v[2]} 
                  for k, v in chamfer_results.items()}, f, indent=2)
    print(f"Raw results saved to {json_output}")


if __name__ == "__main__":
    main()