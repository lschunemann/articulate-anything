import os
import sys
sys.path.append('..')

from articulate_anything.api.odio_urdf import save_joint_states
import glob

def post_process_generated_joints(result_dir="results"):
    """
    Post-process generated object directories to create missing robot_joints.json files.
    Only looks in joint_actor/iter_{}/seed_{} directories.
    """
    generated_dir = os.path.join(result_dir, "generated")
    if not os.path.exists(generated_dir):
        print(f"Generated directory not found: {generated_dir}")
        return
    
    # Find all generated object directories
    pattern = os.path.join(generated_dir, "partnet_*_single-view")
    obj_dirs = glob.glob(pattern)
    
    print(f"Found {len(obj_dirs)} generated object directories")
    
    processed = 0
    errors = 0
    
    for obj_dir in obj_dirs:
        obj_name = os.path.basename(obj_dir)
        joint_actor_dir = os.path.join(obj_dir, "joint_actor")
        
        if not os.path.exists(joint_actor_dir):
            print(f"No joint_actor directory in {obj_name}")
            errors += 1
            continue
        
        # Look for iter_*/seed_* directories
        iter_pattern = os.path.join(joint_actor_dir, "iter_*")
        iter_dirs = glob.glob(iter_pattern)
        
        if not iter_dirs:
            print(f"No iter_ directories found in {obj_name}/joint_actor")
            errors += 1
            continue
        
        obj_processed = False
        
        for iter_dir in iter_dirs:
            seed_pattern = os.path.join(iter_dir, "seed_*")
            seed_dirs = glob.glob(seed_pattern)
            
            for seed_dir in seed_dirs:
                # Look for URDF files in this seed directory
                possible_urdf_files = [
                    os.path.join(seed_dir, "mobility.urdf"),
                    os.path.join(seed_dir, "robot.urdf")
                ]
                
                urdf_file = None
                for path in possible_urdf_files:
                    if os.path.exists(path):
                        urdf_file = path
                        break
                
                if urdf_file:
                    try:
                        # Check if robot_joints.json already exists
                        output_path = os.path.join(seed_dir, "robot_joints.json")
                        
                        if os.path.exists(output_path):
                            continue  # Skip if already exists
                        
                        print(f"Processing {obj_name}/{os.path.basename(iter_dir)}/{os.path.basename(seed_dir)}")
                        
                        # Generate the robot_joints.json file
                        save_joint_states(urdf_file, output_path)
                        obj_processed = True
                        
                    except Exception as e:
                        print(f"Error processing {seed_dir}: {e}")
                        continue
        
        if obj_processed:
            processed += 1
        else:
            errors += 1
    
    print(f"\nPost-processing complete:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {errors}")
    print(f"  Total directories: {len(obj_dirs)}")

if __name__ == "__main__":
    post_process_generated_joints()