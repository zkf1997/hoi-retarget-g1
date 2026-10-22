"""IsaacSim backend with one object asset per environment.

If a RigidObjectConfig's ``urdf_file`` is a DIRECTORY, every ``*.urdf`` in it (sorted) becomes one asset of an IsaacLab
MultiAssetSpawnerCfg with ``random_choice=False``, so environment i receives asset ``i % N``. The MultiMotionLoader sorts its
motion files the same way, and FixedMotionCommand binds environment i to motion ``i % N``: motion and object always match.
Requires ``SceneConfig.replicate_physics=False`` (heterogeneous assets).

Pitfall found 2026-10-15: holosoma spawns objects BEFORE ``scene.clone_environments(copy_from_source=False)``, and with
``copy_from_source=False`` every env_i prim INHERITS env_0's children (live). Each env therefore carried its own asset plus
env_0's object, IsaacLab's rigid-body view (path pattern built from env_0's body name) picked env_0's object in every env,
and all 60 envs effectively simulated the Box_A while the real objects were extra uncontrolled bodies.
Deep-copy cloning crashed PhysX's GPU views and deactivating the inherited copies broke the view pattern. What works:
give every object URDF the same body LINK name (``<link name="object_body">``; the importer names the body prim after
it), so env_i's own body prim overrides the inherited one by name and the rigid-body view pattern matches every env.
Files live at ``assets/objects/<tag>/hiphi_object.urdf``.
"""
from __future__ import annotations

import os
from pathlib import Path

from holosoma.config_types.scene import SceneConfig
from holosoma.simulator.isaacsim.isaacsim import IsaacSim as _BaseIsaacSim


class IsaacSim(_BaseIsaacSim):  # name must lowercase to 'isaacsim' (holosoma infers the simulator type from the class name)
    def _load_rigid_objects(self, scene_config: SceneConfig) -> None:
        self._deferred_multi_assets = {}
        for obj_name, obj in scene_config.rigid_objects.items():
            path = obj.urdf_file
            if path is not None and os.path.isdir(os.path.expanduser(path)):
                self._deferred_multi_assets[obj_name] = obj  # spawned after cloning, see _setup_scene
            else:
                from holosoma.simulator.isaacsim.object_spawner import build_standalone_rigid_object

                self.scene.rigid_objects[obj_name] = build_standalone_rigid_object(
                    obj_name, obj, self.get_supported_scene_formats(), scene_config.asset_root
                )

    def _spawn_deferred_objects(self) -> None:
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg

        from holosoma.simulator.isaacsim.object_spawner import select_spawn_cfg

        for obj_name, obj in getattr(self, "_deferred_multi_assets", {}).items():
            path = obj.urdf_file
            # layout: <dir>/<tag>/hiphi_object.urdf -- the IMPORTED PRIM IS NAMED AFTER THE FILE STEM, and it must be the same
            # in every env (see module docstring), so every asset is a file with the same name in its own folder
            urdfs = sorted(Path(os.path.expanduser(path)).glob("*/hiphi_object.urdf"))
            assert urdfs, f"no <tag>/hiphi_object.urdf files in {path}"
            links = {u.read_text().split('<link name="', 1)[1].split('"', 1)[0] for u in urdfs}
            assert links == {"object_body"}, f'all object URDFs need <link name="object_body"> (found {sorted(links)[:3]})'
            assets = [select_spawn_cfg("urdf", str(u.resolve()), fixed=obj.fixed, physics=obj.physics, source_prim_path=None) for u in urdfs]
            spawn = MultiAssetSpawnerCfg(assets_cfg=assets, random_choice=False)
            # Objects live OUTSIDE /World/envs/env_i: env_i inherits env_0 (holosoma clones with copy_from_source=False),
            # so anything under env_0 leaks into every env (mass, collision pieces). /World/hiphi_objects/env_i has an
            # identity transform; holosoma writes object poses in world coordinates (+ env origin) anyway.
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            for i in range(self.training_config.num_envs):
                UsdGeom.Xform.Define(stage, f"/World/hiphi_objects/env_{i}")
            cfg = RigidObjectCfg(
                prim_path=f"/World/hiphi_objects/env_.*/{obj_name}",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(obj.position), rot=tuple(obj.orientation)),
            )
            self.scene.rigid_objects[obj_name] = RigidObject(cfg)
            print(f"[hiphi_wbt IsaacSim multi-asset] object '{obj_name}': {len(urdfs)} assets from {path}, env i <- asset i % {len(urdfs)} (spawned after cloning)")
        self._deferred_multi_assets = {}

    def _setup_scene(self) -> None:
        """holosoma's scene setup, with the per-env objects spawned right after ``clone_environments``."""
        from isaaclab.scene import InteractiveScene

        original = InteractiveScene.clone_environments
        sim_self = self

        def hooked(scene_self, *args, **kwargs):
            result = original(scene_self, *args, **kwargs)
            sim_self._spawn_deferred_objects()
            return result

        InteractiveScene.clone_environments = hooked
        try:
            super()._setup_scene()
        finally:
            InteractiveScene.clone_environments = original
