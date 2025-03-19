def extract_meshes_with_specific_vertices(obj_file_path):
    import os
    
    with open(obj_file_path, 'r') as file:
        content = file.readlines()
    
    # Parse all vertices, normals, and texture coordinates
    vertices = []
    normals = []
    texcoords = []
    
    # Store mesh sections
    meshes = {}
    current_material = None
    mesh_faces = []
    
    # Parse the file
    for line in content:
        line = line.strip()
        
        if line.startswith('v '):
            vertices.append(line)
        elif line.startswith('vn '):
            normals.append(line)
        elif line.startswith('vt '):
            texcoords.append(line)
        elif line.startswith('usemtl'):
            if current_material:
                # Store previous mesh
                meshes[current_material] = mesh_faces
            
            # Start new mesh
            current_material = line.split()[1]
            mesh_faces = [line]  # Include the material line
        elif line.startswith('f ') and current_material:
            mesh_faces.append(line)
        elif line.startswith('#') or line.startswith('mtllib'):
            # Keep comments and material library references
            if not current_material:
                # These are header lines
                if 'header' not in meshes:
                    meshes['header'] = []
                meshes['header'].append(line)
    
    # Add the last mesh if exists
    if current_material and mesh_faces:
        meshes[current_material] = mesh_faces
    
    # Process each mesh
    for material, faces in meshes.items():
        if material == 'header':
            continue
            
        # Find which vertices, normals, and texcoords are used in this mesh
        used_vertices = set()
        used_normals = set()
        used_texcoords = set()
        
        for face in faces:
            if face.startswith('f '):
                # Parse face indices
                parts = face.split()[1:]
                for part in parts:
                    indices = part.split('/')
                    if len(indices) >= 1 and indices[0]:
                        used_vertices.add(int(indices[0]))
                    if len(indices) >= 2 and indices[1]:
                        used_texcoords.add(int(indices[1]))
                    if len(indices) >= 3 and indices[2]:
                        used_normals.add(int(indices[2]))
        
        # Create mapping for new indices
        vertex_map = {old: new+1 for new, old in enumerate(sorted(used_vertices))}
        normal_map = {old: new+1 for new, old in enumerate(sorted(used_normals))}
        texcoord_map = {old: new+1 for new, old in enumerate(sorted(used_texcoords))}
        
        # Create output file
        base_path = '/'.join(obj_file_path.split('/')[:-1]) + '/separate_meshes'
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        output_file = f"{base_path}/mesh_{material.replace('$', '')}.obj"
        with open(output_file, 'w') as out_file:
            # Write header
            if 'header' in meshes:
                for line in meshes['header']:
                    out_file.write(line + '\n')
            
            # Write used vertices with new indices
            for old_idx in sorted(used_vertices):
                # OBJ indices are 1-based, so subtract 1 to get 0-based array index
                out_file.write(vertices[old_idx-1] + '\n')
            
            # Write used texture coordinates
            for old_idx in sorted(used_texcoords):
                if old_idx <= len(texcoords):
                    out_file.write(texcoords[old_idx-1] + '\n')
            
            # Write used normals
            for old_idx in sorted(used_normals):
                if old_idx <= len(normals):
                    out_file.write(normals[old_idx-1] + '\n')
            
            # Write material
            for line in faces:
                if line.startswith('usemtl'):
                    out_file.write(line + '\n')
            
            # Write faces with remapped indices
            for line in faces:
                if line.startswith('f '):
                    parts = line.split()
                    new_face = ['f']
                    
                    for part in parts[1:]:
                        indices = part.split('/')
                        new_indices = []
                        
                        # Remap vertex index
                        if len(indices) >= 1 and indices[0]:
                            new_indices.append(str(vertex_map[int(indices[0])]))
                        else:
                            new_indices.append('')
                        
                        # Remap texture coordinate index
                        if len(indices) >= 2:
                            if indices[1] and int(indices[1]) in texcoord_map:
                                new_indices.append(str(texcoord_map[int(indices[1])]))
                            else:
                                new_indices.append('')
                        
                        # Remap normal index
                        if len(indices) >= 3:
                            if indices[2] and int(indices[2]) in normal_map:
                                new_indices.append(str(normal_map[int(indices[2])]))
                            else:
                                new_indices.append('')
                        
                        new_face.append('/'.join(new_indices))
                    
                    out_file.write(' '.join(new_face) + '\n')
        
        print(f"Created {output_file} with {len(used_vertices)} vertices")

# Usage
extract_meshes_with_specific_vertices("/home/link/DreMa/third_party/articulate-anything/datasets/output_views/drawer_rodin/drawer_RLBench.obj")