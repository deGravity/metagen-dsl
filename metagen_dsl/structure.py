from enum import Enum
from .tile import Tile
from .pattern import TilingPattern


# =======================================
#   Structures
# =======================================
class Structure:
    # TODO: if custom operations like rotation are used,
    # we should probably check that there is eg something about the skeleton incident on the rotation line, otherwise it might be invalid
    # if the structure is disconnected under those operations

    def __init__(self, tile:Tile, pattern:TilingPattern):
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _tile = tile
        _pattern = pattern

        self.tile = _tile
        self.pat = _pattern

    def __call__(self, color=None, box_color=None, simulate=False, resolution=64):
        from tempfile import TemporaryDirectory, TemporaryFile
        import os
        import trimesh
        import base64
        from metagen import ProcMetaTranslator # local to avoid circular reference
        from metagen.processing import generate_and_simulate, run_render, load_voxels, validate_voxels
        from metagen.validation import validate_simulation
        output = {'valid':False}
        with TemporaryDirectory(delete=False) as tempdir:
            graph_path = os.path.join(tempdir, 'graph.json')
            ProcMetaTranslator(self).save(graph_path)
            log_info = generate_and_simulate(graph_path, resolution, tempdir, simulate=simulate)
            output['log_info'] = log_info
            mesh_path = os.path.join(tempdir, 'thickened_mc.obj')
            vox_path = os.path.join(tempdir, 'vox_active_cells.txt')
            sim_path = os.path.join(tempdir, 'structure_info.json')
            if os.path.exists(vox_path):
                with open(vox_path, 'r') as f:
                    voxels = load_voxels(f)
                valid = validate_voxels(voxels)
                output['valid'] = valid
                output['voxels'] = voxels
            if os.path.exists(mesh_path):
                run_render(mesh_path, tempdir, color, box_color)
                def read_im(p):
                    with open(p,'rb') as f:
                        return base64.b64encode(f.read()).decode()
                mesh = trimesh.load_mesh(mesh_path)
                output['mesh'] = mesh
                images = {n: read_im(os.path.join(tempdir,f'{n}.png')) for n in ['top_right','top','front','right','all']}
                output['images'] = images
            if os.path.exists(sim_path):
                import json
                with open(sim_path, 'r') as f:
                    sim_data = json.load(f)
                output['sim_data'] = sim_data
                output['sim_valid'] = validate_simulation(sim_path)
        return output
    
    def _repr_llm_(self):
        output = self()
        content = []
        
        if 'images' in output:
            if not output['valid']:
                content.append({'type':'text', 'value':'Invalid Metamaterial'})
            for (label, key) in [('Front-Upper-Right', 'top_right'), ('Top', 'top'), ('Front', 'front'), ('Right', 'right')]:
                img_data = output['images'][key]
                img_enc = f"data:image/png;base64, {img_data}"
                content.append({'type':'text', 'value':f'{label} View:'})
                content.append({'type':'image', 'value':img_enc})
        else:
            content.append({'type':'text', 'value':'Program failed to produce a material.'})

        return {
            'role': 'dsl',
            'content': content
        }


    def _repr_html_(self):
        output = self()
        tags = []
        if 'mesh' in output:
            import trimesh.viewer
            return trimesh.viewer.scene_to_notebook(trimesh.Scene(output['mesh']))._repr_html_()
        if 'images' in output:
            table = '''
            <table>
            <tr>
                <td><img src="data:image/png;base64, {top_data}" /></td>
                <td><img src="data:image/png;base64, {top_right_data}" /></td>
            </tr>
            <tr>
                <td><img src="data:image/png;base64, {front_data}" /></td>
                <td><img src="data:image/png;base64, {right_data}" /></td>
            </tr>
            </table>
            '''.format(top_data = output['images']['top'],
                top_right_data = output['images']['top_right'],
                front_data = output['images']['front'],
                right_data = output['images']['right'])
            tags.append(table)
            #for viewpoint, data in output['images'].items():
                #data = output['images']['top_right']
                #tags.append(f'<p><b>{viewpoint}</b><img src="data:image/png;base64, {data}" /></p>')
            tags.append(f'<p> {"Valid" if output["valid"] else "Invalid"} </p>')
        else:
            tags.append('<p>Failed to produce output.</p>')
        return '\n'.join(tags)
        



# =======================================
#   CSG Booleans -- also Structures since they're realizable
# =======================================
class CSGBooleanTypes(Enum):
    UNION = 0
    INTERSECT = 1
    DIFFERENCE = 2

class CSGBoolean(Structure):
    def __init__(self, _A:Structure, _B:Structure, _opType:CSGBooleanTypes):
        self.A = _A
        self.B = _B
        self.op_type = _opType

class Union(CSGBoolean):
    def __init__(self, A:Structure, B:Structure):
        super().__init__(A, B, CSGBooleanTypes.UNION)

class Intersect(CSGBoolean):
    def __init__(self, A:Structure, B:Structure):
        super().__init__(A, B, CSGBooleanTypes.INTERSECT)

class Subtract(CSGBoolean):
    def __init__(self, A:Structure, B:Structure):
        super().__init__(A, B, CSGBooleanTypes.DIFFERENCE)