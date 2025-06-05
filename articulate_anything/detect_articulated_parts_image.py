import os
from openai import OpenAI
import base64
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
import argparse

def make_prompt(input_path):
    # Load a single image instead of video frames
    image = Image.open(input_path)
    
    content = [
            {
                "type": "text",
                "text": "Here is an image of the object: "
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(image)}"
                }
            }
        ]

    return content

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def identify_movable_parts(image_path):
    # Initialize OpenAI client
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("API_KEY"), base_url="https://ai-gateway.mytkhgroup.com/")
    
    # Updated system instruction for image analysis
    system_instruction = """
    You have a good understanding of the structure of articulated objects. Your job is to assist the user to analyze the structure of an object. Specifically, the user will give you an image of an articulated object, and your task is to recognize the main parts of that object that could potentially move or articulate.

    You should give your answer in the following format:
    ```part_list
    (1) part_name: name of the part; description: a brief description about the part, and how it might move or articulate
    (2) part_name: name of the part; description: a brief description about the part, and how it might move or articulate
    ...
    ```
    
    Remember:
    (1) Focus only on parts that could potentially move, articulate, or be manipulated (e.g., doors, drawers, lids, knobs, hinges).
    (2) Your answer should be purely based on the input image, do not imagine anything.
    (3) If there are multiple parts with the same semantic, just add one part to the list. For example, if there are four wheels, just add one part whose name is wheel.
    (4) Your answer has to be based on the object being shown. If there is a robotic or a human arm interacting with the object, ignore it and just describe the object.
    """

    content = make_prompt(image_path)
    
    # Prepare the message
    messages = [
        {
            "role": "system", 
            "content": system_instruction
        },
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Call the API
    response = client.chat.completions.create(
        model="claude-3-5-sonnet-latest",
        messages=messages,
        temperature=0.5
    )
    
    return response.choices[0].message.content

def extract_part_list(api_response):
    import re
    """Extracts the part list found between triple backticks (```) in a string."""
    pattern = r"```part_list\n([\s\S]*?)```"
    matches = re.findall(pattern, api_response, re.DOTALL)

    if matches:
        # Return the first match of the part list, removing whitespace
        return matches[0].strip()
    else:
        # If no code block found, return the full response (it might already be in the right format)
        return api_response.strip()

def parse_part_list(part_list_str):
    # Initialize lists to store part names and descriptions
    part_names = []
    part_descriptions = []
    
    # Split the input by lines
    lines = part_list_str.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove the numbering if present (e.g., "(1) ")
        if line[0] == "(" and ")" in line:
            line = line[line.find(")")+1:].strip()
        
        # Split by semicolon to separate part name and description
        sections = line.split(';')
        if len(sections) != 2:
            continue
            
        name_section = sections[0].strip()
        desc_section = sections[1].strip()
        
        # Extract the actual name and description after the colon
        if "part_name:" in name_section:
            name = name_section.split("part_name:")[1].strip()
        elif ":" in name_section:
            name = name_section.split(":")[1].strip()
        else:
            continue
            
        if "description:" in desc_section:
            description = desc_section.split("description:")[1].strip()
        elif ":" in desc_section:
            description = desc_section.split(":")[1].strip()
        else:
            continue
        
        part_names.append(name)
        part_descriptions.append(description)
    
    return part_names, part_descriptions

def detect_articulated_parts(image_path):
    response = identify_movable_parts(image_path)
    part_list_str = extract_part_list(response)
    
    part_names, part_descriptions = parse_part_list(part_list_str)
    
    # Format results as specified
    names_output = ".".join(part_names)
    descriptions_output = ".".join(part_descriptions)
    
    return names_output, descriptions_output, part_list_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('object', help='Name of the object to analyze')
    # parser.add_argument('--image_path', help='Path to the image file', required=True)
    args = parser.parse_args()

    object_ = args.object.split('_')[1]
    image_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/partnet-mobility-v0-processed/partnet-mobility-v0/dataset/{object_}/robot_frontview.png"
    
    # Create output directory structure if it doesn't exist
    out_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.object}"
    os.makedirs(out_path, exist_ok=True)
    
    # If already ran, skip repeated execution
    if os.path.isfile(f"{out_path}/segmentation_targets.txt"):
        print(f"Segmentation targets already exist for object {args.object}, skipping...")
    else:
        names_output, descriptions_output, raw_part_list = detect_articulated_parts(image_path)
        
        # Write results to the file
        with open(os.path.join(out_path, "segmentation_targets.txt"), 'w') as f:
            f.write(f"{names_output}\n{descriptions_output}")
        
        # Optional: Write the raw part list to a separate file for debugging
        with open(os.path.join(out_path, "raw_part_list.txt"), 'w') as f:
            f.write(raw_part_list)
            
        print(f"Successfully analyzed {args.object} and saved results to {out_path}/segmentation_targets.txt")