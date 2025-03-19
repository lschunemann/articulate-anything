import xml.etree.ElementTree as ET
import os

"""
extract part URDFs from URDF of whole object retrieved by articulate anything
"""

def extract_links_from_urdf(input_urdf_path, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Parse the original URDF file
    tree = ET.parse(input_urdf_path)
    root = tree.getroot()

    # Iterate through each link in the URDF
    for link in root.findall('link'):
        link_name = link.get('name')
        
        # Create a new URDF structure for the individual link
        new_robot = ET.Element('robot', name=link_name)
        new_link = ET.SubElement(new_robot, 'link', name=link_name)

        # Copy visual and collision elements
        for visual in link.findall('visual'):
            new_visual = ET.SubElement(new_link, 'visual')
            new_visual.append(visual.find('geometry'))
            new_visual.append(visual.find('origin'))

        for collision in link.findall('collision'):
            new_collision = ET.SubElement(new_link, 'collision')
            new_collision.append(collision.find('geometry'))
            new_collision.append(collision.find('origin'))

        # Write the new URDF to a file
        new_urdf_path = os.path.join(output_directory, f"{link_name}.urdf")
        new_tree = ET.ElementTree(new_robot)
        new_tree.write(new_urdf_path, encoding='utf-8', xml_declaration=True)

    print(f"Extracted links to {output_directory}")

# Example usage
input_urdf = '/home/link/DreMa/third_party/articulate-anything/results/video/drawer_RL_Bench/link_placement/iter_0/seed_0/mobility.urdf'  # Replace with your URDF file path
output_dir = '/home/link/DreMa/third_party/articulate-anything/results/video/drawer_RL_Bench/urdfs'    # Replace with your desired output directory
extract_links_from_urdf(input_urdf, output_dir)