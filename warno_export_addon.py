bl_info = {
    "name": "WARNO Quick Export",
    "author": "ChatGPT",
    "version": (1, 2, 0),
    "blender": (3, 0, 0),
    "location": "Topbar -> WARNO Export",
    "description": "Import P3D via Enfusion Tools (auto name + auto textures), export LODs H/M/L with WARNO FBX settings, optional Definitions.ndf append",
    "category": "Import-Export",
}

import bpy
from pathlib import Path
from bpy.types import AddonPreferences
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper


# ==========================================================
# Preferences (щоб меню було компактним)
# ==========================================================
class WARNO_AddonPreferences(AddonPreferences):
    bl_idname = __name__

    models_dir: StringProperty(
        name="WARNO Models/Fences folder",
        subtype='DIR_PATH',
        default=r"C:\Users\galin\Saved Games\EugenSystems\WARNO\DecorsSets\UkraineHouses\Models\Fences",
    )

    texture_dir: StringProperty(
        name="Texture search folder",
        subtype='DIR_PATH',
        default=r"C:\Users\galin\Saved Games\EugenSystems\WARNO\DecorsSets\UkraineHouses\Models\Fences",
    )

    definitions_path: StringProperty(
        name="Definitions.ndf",
        subtype='FILE_PATH',
        default=r"C:\Users\galin\Saved Games\EugenSystems\WARNO\DecorsSets\UkraineHouses\Definitions\Definitions.ndf",
    )

    strip_prefix: StringProperty(
        name="Strip Prefix",
        description="Що прибирати на початку назви p3d (наприклад rhs_)",
        default="rhs_",
    )

    auto_link_after_import: BoolProperty(
        name="Auto-link textures after import",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="WARNO Quick Export Settings")
        layout.prop(self, "models_dir")
        layout.prop(self, "texture_dir")
        layout.prop(self, "definitions_path")
        layout.prop(self, "strip_prefix")
        layout.prop(self, "auto_link_after_import")


def prefs(context) -> WARNO_AddonPreferences:
    return context.preferences.addons[__name__].preferences


# ==========================================================
# Helpers
# ==========================================================
def _is_mesh_obj(o):
    return o and o.type == "MESH"

def _strip_prefix(stem: str, prefix: str):
    return stem[len(prefix):] if prefix and stem.startswith(prefix) else stem

def _strip_dot_number(name: str):
    # "mat.001" -> "mat"
    if len(name) > 4 and name[-4] == "." and name[-3:].isdigit():
        return name[:-4]
    return name

def _sanitize_material_key(name: str):
    n = _strip_dot_number(name)
    n = n.replace(" ", "_")
    return n

def _collect_lod_objects():
    col0 = bpy.data.collections.get("LOD0")
    col1 = bpy.data.collections.get("LOD1")
    col2 = bpy.data.collections.get("LOD2")

    def from_collection(col):
        if not col:
            return []
        return [o for o in col.objects if _is_mesh_obj(o)]

    lod0 = from_collection(col0)
    lod1 = from_collection(col1)
    lod2 = from_collection(col2)

    if lod0 or lod1 or lod2:
        return {"LOD0": lod0, "LOD1": lod1, "LOD2": lod2}

    # fallback by object name
    lods = {"LOD0": [], "LOD1": [], "LOD2": []}
    for o in bpy.data.objects:
        if not _is_mesh_obj(o):
            continue
        n = o.name.lower()
        if "lod0" in n:
            lods["LOD0"].append(o)
        elif "lod1" in n:
            lods["LOD1"].append(o)
        elif "lod2" in n:
            lods["LOD2"].append(o)
    return lods


def _ensure_principled_setup(mat: bpy.types.Material):
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    out = None
    principled = None

    for n in nodes:
        if n.type == "OUTPUT_MATERIAL":
            out = n
        elif n.type == "BSDF_PRINCIPLED":
            principled = n

    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (500, 0)

    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.location = (200, 0)

    if not out.inputs["Surface"].is_linked:
        links.new(principled.outputs["BSDF"], out.inputs["Surface"])

    return nt, principled


def _get_or_create_img_node(nt, label: str, x: int, y: int):
    for n in nt.nodes:
        if n.type == "TEX_IMAGE" and n.label == label:
            return n
    n = nt.nodes.new("ShaderNodeTexImage")
    n.label = label
    n.location = (x, y)
    return n


def _get_or_create_normalmap_node(nt, x: int, y: int):
    for n in nt.nodes:
        if n.type == "NORMAL_MAP":
            return n
    n = nt.nodes.new("ShaderNodeNormalMap")
    n.location = (x, y)
    return n


def _load_image(path: Path):
    return bpy.data.images.load(str(path), check_existing=True)


COLOR_TOKENS = {
    "gray","grey","green","red","blue","brown","black","white","yellow","orange",
    "tan","beige","dark","light"
}

def _unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _strip_color_suffix(key: str):
    parts = key.split("_")
    if parts and parts[-1].lower() in COLOR_TOKENS:
        return "_".join(parts[:-1])
    return key

def _progressive_shorter_keys(key: str, min_parts: int = 3):
    """
    fence_type_05_gray -> [fence_type_05_gray, fence_type_05]
    fence_type_02_blue_pole_01 -> ... (обрізає по 1 токену)
    """
    parts = key.split("_")
    keys = [key]
    while len(parts) > min_parts:
        parts = parts[:-1]
        keys.append("_".join(parts))
    return keys

def _key_from_basecolor(mat):
    """
    Якщо Base Color підключений напряму з Image Texture — беремо filename,
    і якщо він закінчується на _B або _co — робимо ключ.
    """
    if not mat or not mat.use_nodes:
        return None
    nt = mat.node_tree
    principled = None
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            principled = n
            break
    if not principled:
        return None

    inp = principled.inputs.get("Base Color")
    if not inp or not inp.is_linked:
        return None

    from_node = inp.links[0].from_node
    if not from_node or from_node.type != "TEX_IMAGE":
        return None
    img = getattr(from_node, "image", None)
    if not img:
        return None

    # filename key
    if img.filepath:
        stem = Path(bpy.path.abspath(img.filepath)).stem
    else:
        stem = Path(img.name).stem

    stem = _strip_dot_number(stem)

    # прибрати суфікс BaseColor
    for suf in ("_B", "_B_", "_co", "_CO"):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break

    return stem

def _find_texture(folder: Path, keys, suffixes):
    """
    keys: список можливих базових імен
    suffixes: список можливих суфіксів, напр ["_NM", "_NM_"]
    """
    for k in keys:
        for suf in suffixes:
            p = folder / f"{k}{suf}.png"
            if p.exists():
                return p
    return None

def _auto_link_textures_for_material(mat: bpy.types.Material, tex_dir: Path):
    # базовий ключ по матеріалу
    mat_key = _sanitize_material_key(mat.name)

    # ключ з BaseColor якщо є
    bc_key = _key_from_basecolor(mat)

    # список кандидатів (спочатку найкращі)
    candidates = []
    if bc_key:
        candidates += _progressive_shorter_keys(bc_key)
        candidates += _progressive_shorter_keys(_strip_color_suffix(bc_key))
    candidates += _progressive_shorter_keys(mat_key)
    candidates += _progressive_shorter_keys(_strip_color_suffix(mat_key))

    candidates = _unique(candidates)

    nt, principled = _ensure_principled_setup(mat)
    links = nt.links

    found = False

    # BaseColor (_B) -> sRGB
    pB = _find_texture(tex_dir, candidates, ["_B", "_B_", "_co", "_CO"])
    if pB:
        img = _load_image(pB)
        img.colorspace_settings.name = "sRGB"
        n = _get_or_create_img_node(nt, "WARNO_B", -600, 200)
        n.image = img
        if principled.inputs["Base Color"].is_linked:
            for l in list(principled.inputs["Base Color"].links):
                links.remove(l)
        links.new(n.outputs["Color"], principled.inputs["Base Color"])
        found = True

    # Metallic (_M) -> Non-Color (пробуємо також _M_)
    pM = _find_texture(tex_dir, candidates, ["_M", "_M_"])
    if pM:
        img = _load_image(pM)
        img.colorspace_settings.name = "Non-Color"
        n = _get_or_create_img_node(nt, "WARNO_M", -600, 20)
        n.image = img
        if principled.inputs["Metallic"].is_linked:
            for l in list(principled.inputs["Metallic"].links):
                links.remove(l)
        links.new(n.outputs["Color"], principled.inputs["Metallic"])
        found = True

    # Roughness (_R) -> Non-Color (пробуємо також _R_)
    pR = _find_texture(tex_dir, candidates, ["_R", "_R_"])
    if pR:
        img = _load_image(pR)
        img.colorspace_settings.name = "Non-Color"
        n = _get_or_create_img_node(nt, "WARNO_R", -600, -160)
        n.image = img
        if principled.inputs["Roughness"].is_linked:
            for l in list(principled.inputs["Roughness"].links):
                links.remove(l)
        links.new(n.outputs["Color"], principled.inputs["Roughness"])
        found = True

    # Normal (_NM) -> Non-Color -> NormalMap -> Normal (пробуємо також _NM_)
    pNM = _find_texture(tex_dir, candidates, ["_NM", "_NM_"])
    if pNM:
        img = _load_image(pNM)
        img.colorspace_settings.name = "Non-Color"
        n = _get_or_create_img_node(nt, "WARNO_NM", -600, -340)
        n.image = img
        nm_node = _get_or_create_normalmap_node(nt, -250, -340)

        if nm_node.inputs["Color"].is_linked:
            for l in list(nm_node.inputs["Color"].links):
                links.remove(l)
        links.new(n.outputs["Color"], nm_node.inputs["Color"])

        if principled.inputs["Normal"].is_linked:
            for l in list(principled.inputs["Normal"].links):
                links.remove(l)
        links.new(nm_node.outputs["Normal"], principled.inputs["Normal"])
        found = True

    return found


def _detect_apply_scale_enum():
    prop = bpy.ops.export_scene.fbx.get_rna_type().properties['apply_scale_options']
    enum_ids = [e.identifier for e in prop.enum_items]
    if "FBX_SCALE_ALL" in enum_ids:
        return "FBX_SCALE_ALL"
    if "FBX_ALL" in enum_ids:
        return "FBX_ALL"
    return enum_ids[0] if enum_ids else None


def _export_fbx(objects, out_fbx: Path, export_animation: bool):
    out_fbx.parent.mkdir(parents=True, exist_ok=True)
    apply_scale = _detect_apply_scale_enum()

    bpy.ops.object.select_all(action='DESELECT')
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0] if objects else None

    bpy.ops.export_scene.fbx(
        filepath=str(out_fbx),
        use_selection=True,

        # Include
        use_custom_props=True,

        # Transform (WARNO required)
        apply_scale_options=apply_scale,
        axis_forward='Y',
        axis_up='Z',

        # Geometry (WARNO required)
        use_triangles=True,

        # Armature (WARNO required)
        add_leaf_bones=False,

        # Animation (optional)
        bake_anim=export_animation,
    )


def _append_definitions_block(defs_path: Path, name: str):
    block = f"""
unnamed SceneryDescriptor_Model
(
    RegistrationName = '{name}'
    Tags             = [ "Fences" ]
    ModelHigh = TResourceMesh
    (
        Mesh = 'DecorsSetsData:/UkraineHouses/Models/Fences/{name}_H.fbx'
    )
    ModelMid = TResourceMesh
    (
        Mesh = 'DecorsSetsData:/UkraineHouses/Models/Fences/{name}_M.fbx'
    )
    ModelLow = TResourceMesh
    (
        Mesh = 'DecorsSetsData:/UkraineHouses/Models/Fences/{name}_L.fbx'
    )
    Destroyable = true
    LodConfig = $/SceneryDB/LodConfig_Big
    BuildingType = BuildingType/None
    IsForDistrict = true
)
""".lstrip("\n")

    defs_path.parent.mkdir(parents=True, exist_ok=True)
    if defs_path.exists():
        txt = defs_path.read_text(encoding="utf-8", errors="ignore")
        if f"RegistrationName = '{name}'" in txt:
            return False
        if not txt.endswith("\n"):
            txt += "\n"
        txt += "\n" + block
        defs_path.write_text(txt, encoding="utf-8")
        return True
    defs_path.write_text(block, encoding="utf-8")
    return True


# ==========================================================
# Operator: Auto-Link Textures
# ==========================================================
class WARNO_OT_auto_link_textures(bpy.types.Operator):
    bl_idname = "warno.auto_link_textures"
    bl_label = "Auto-Link Textures"

    def execute(self, context):
        p = prefs(context)
        tex_dir = Path(bpy.path.abspath(p.texture_dir)).expanduser()
        if not tex_dir.exists():
            self.report({'ERROR'}, f"Texture folder not found: {tex_dir}")
            return {'CANCELLED'}

        lods = _collect_lod_objects()
        objs = lods.get("LOD0", []) + lods.get("LOD1", []) + lods.get("LOD2", [])
        if not objs:
            objs = [o for o in context.selected_objects if _is_mesh_obj(o)]

        mats = set()
        for o in objs:
            for slot in o.material_slots:
                if slot.material:
                    mats.add(slot.material)

        linked = 0
        missed = 0
        for m in mats:
            if _auto_link_textures_for_material(m, tex_dir):
                linked += 1
            else:
                missed += 1

        self.report({'INFO'}, f"Auto-link: linked={linked}, missed={missed}")
        return {'FINISHED'}


# ==========================================================
# Operator: Import P3D via Enfusion Tools (works!)
# ==========================================================
class WARNO_OT_import_p3d(bpy.types.Operator, ImportHelper):
    bl_idname = "warno.import_p3d_warno"
    bl_label = "Import P3D (WARNO)"

    filename_ext = ".p3d"
    filter_glob: StringProperty(default="*.p3d", options={'HIDDEN'})

    def execute(self, context):
        p = prefs(context)
        p3d_path = Path(self.filepath)
        if not p3d_path.exists():
            self.report({'ERROR'}, "File not found")
            return {'CANCELLED'}

        # 1) AutoName from filename
        base = _strip_prefix(p3d_path.stem, p.strip_prefix)
        context.scene.warno_base_name = base

        # 2) Call Enfusion Tools importer
        if not hasattr(bpy.ops.scene, "ebt_import_p3d"):
            self.report({'ERROR'}, "Enfusion Tools importer not found: bpy.ops.scene.ebt_import_p3d")
            return {'CANCELLED'}

        # Спроба викликати імпорт у CLI-режимі (без їхнього file browser)
        # Enfusion tools сам підхопить layer presets у execute якщо usingCLI=True.
        try:
            bpy.ops.scene.ebt_import_p3d(
                filepath=str(p3d_path),
                usingCLI=True,
                discardUnsupportedLods=True,
                updateGameMaterials=True,
                convertAxis=True,
            )
        except Exception as e:
            self.report({'ERROR'}, f"EBT import failed: {e}")
            return {'CANCELLED'}

        # 3) Auto textures після імпорту
        if p.auto_link_after_import:
            try:
                bpy.ops.warno.auto_link_textures()
            except Exception as e:
                self.report({'WARNING'}, f"Auto-link textures failed: {e}")

        self.report({'INFO'}, f"Imported: {p3d_path.name} -> Base Name: {base}")
        return {'FINISHED'}


# ==========================================================
# Operator: Export WARNO LODs
# ==========================================================
class WARNO_OT_export(bpy.types.Operator):
    bl_idname = "warno.export_lods"
    bl_label = "Export WARNO LODs"

    def execute(self, context):
        p = prefs(context)
        scn = context.scene

        base_name = scn.warno_base_name.strip()
        if not base_name:
            self.report({'ERROR'}, "Base Name is empty")
            return {'CANCELLED'}

        out_dir = Path(bpy.path.abspath(p.models_dir)).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        lods = _collect_lod_objects()
        lod0 = lods.get("LOD0", [])
        lod1 = lods.get("LOD1", [])
        lod2 = lods.get("LOD2", [])

        if not lod0 and not lod1 and not lod2:
            self.report({'ERROR'}, "No LOD0/LOD1/LOD2 collections found")
            return {'CANCELLED'}

        anim = scn.warno_export_animation

        if lod0:
            _export_fbx(lod0, out_dir / f"{base_name}_H.fbx", anim)
        if lod1:
            _export_fbx(lod1, out_dir / f"{base_name}_M.fbx", anim)
        if lod2:
            _export_fbx(lod2, out_dir / f"{base_name}_L.fbx", anim)

        # optional append definitions
        if scn.warno_append_definitions:
            defs = Path(bpy.path.abspath(p.definitions_path)).expanduser()
            try:
                added = _append_definitions_block(defs, base_name)
                if added:
                    self.report({'INFO'}, "Definitions.ndf appended")
                else:
                    self.report({'INFO'}, "Definitions.ndf already contains this RegistrationName")
            except Exception as e:
                self.report({'WARNING'}, f"Definitions update failed: {e}")

        self.report({'INFO'}, f"Export done -> {out_dir}")
        return {'FINISHED'}


# ==========================================================
# Compact Topbar UI
# ==========================================================
class WARNO_MT_topbar_menu(bpy.types.Menu):
    bl_label = "WARNO Export"
    bl_idname = "WARNO_MT_topbar_menu"

    def draw(self, context):
        layout = self.layout
        p = prefs(context)
        scn = context.scene

        # Compact rows
        layout.operator("warno.import_p3d_warno", icon="IMPORT")

        row = layout.row(align=True)
        row.label(text="Name:")
        row.prop(scn, "warno_base_name", text="")

        row = layout.row(align=True)
        row.prop(scn, "warno_export_animation", text="Anim")
        row.prop(scn, "warno_append_definitions", text="NDF")

        row = layout.row(align=True)
        row.operator("warno.auto_link_textures", text="Relink Tex", icon="NODE_MATERIAL")
        row.operator("warno.export_lods", text="Export", icon="EXPORT")

        layout.separator()
        # Settings shortcut
        layout.operator("warno.open_prefs", icon="PREFERENCES")


def draw_topbar(self, context):
    self.layout.menu(WARNO_MT_topbar_menu.bl_idname)


# ==========================================================
# Open Preferences Operator
# ==========================================================
class WARNO_OT_open_prefs(bpy.types.Operator):
    bl_idname = "warno.open_prefs"
    bl_label = "Open WARNO Addon Settings"

    def execute(self, context):
        try:
            bpy.ops.preferences.addon_show(module=__name__)
        except Exception:
            # fallback: open prefs window
            bpy.ops.screen.userpref_show("INVOKE_DEFAULT")
        return {'FINISHED'}


# ==========================================================
# Register
# ==========================================================
classes = (
    WARNO_AddonPreferences,
    WARNO_OT_auto_link_textures,
    WARNO_OT_import_p3d,
    WARNO_OT_export,
    WARNO_OT_open_prefs,
    WARNO_MT_topbar_menu,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.warno_base_name = StringProperty(
        name="Base Name",
        default="NAME",
        description="NAME без суфікса; аддон додасть _H/_M/_L"
    )
    bpy.types.Scene.warno_export_animation = BoolProperty(
        name="Export Animation",
        default=False
    )
    bpy.types.Scene.warno_append_definitions = BoolProperty(
        name="Append to Definitions.ndf",
        default=False
    )

    bpy.types.TOPBAR_MT_editor_menus.append(draw_topbar)

def unregister():
    try:
        bpy.types.TOPBAR_MT_editor_menus.remove(draw_topbar)
    except Exception:
        pass

    for prop in ["warno_base_name", "warno_export_animation", "warno_append_definitions"]:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    for c in reversed(classes):
        bpy.utils.unregister_class(c)
