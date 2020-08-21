template = '''<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <compiler angle="radian" coordinate="local"></compiler>
    <option timestep="0.005">
        <flag warmstart="enable"></flag>
    </option>
    <!--size nconmax="500" /!-->

    <asset>
        <material name="floor_mat" specular="0" shininess="0.5" reflectance="0" rgba="0.2 0.2 0.2 1"></material>
        <material name="table_mat" specular="0" shininess="0.5" reflectance="0" rgba="0.93 0.93 0.93 1"></material>
        <material name="block_mat" specular="0" shininess="0.5" reflectance="0" rgba="0.1 0.1 0.5 1"></material>
        <material name="stick_mat" specular="0" shininess="0.5" reflectance="0" rgba="0.5 0.2 0.0 0.5"></material>
    </asset>

    <worldbody>
        {scene}
		<body pos="0 0 0" name="masspoint">
			<geom name="masspoint" size="0.15" type="sphere" material="stick_mat" mass="1" euler="1.57 0 0" rgba="0.1 1.0 0.1 1.0"></geom>
			<joint axis="1 0 0" name="masspoint:slidex" type="slide" damping="0.1"></joint>
			<joint axis="0 1 0" name="masspoint:slidey" type="slide" damping="0.1"></joint>
			<joint axis="0 0 1" name="masspoint:slidez" type="slide" damping="10000"></joint>
			<site name="masspoint" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
		</body>

		<body name="object0" pos="0 0 0">
			<!--joint name="object0:joint" type="free" damping="0.01"></joint!-->
			<joint axis="1 0 0" name="object0:slidex" type="slide" damping="10"></joint>
			<joint axis="0 1 0" name="object0:slidey" type="slide" damping="10"></joint>
            <joint axis="0 0 1" name="object0:slidez" type="slide" damping="10000"></joint>
            <joint axis="0 0 1" name="object0:rz" type="hinge" damping="10"></joint>
			<geom size="0.15 0.15 0.15" type="box" name="object0" material="block_mat" mass="1" euler="0 0 0"></geom>
			<!--geom size="0.025 0.025 0.025" type="sphere" condim="6" name="object0" material="block_mat" mass="2"></geom!-->
			<site name="object0" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
		</body>

		{obstacles}

		<light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"></light>
	</worldbody>

	<actuator>
		<motor gear="100" joint="masspoint:slidex"></motor>
		<motor gear="100" joint="masspoint:slidey"></motor>
		<!--position gear="10" joint="masspoint:slidex"></position>
		<position gear="10" joint="masspoint:slidey"></position!-->
	</actuator>
</mujoco>
'''

scene_template = '''
<geom name="floor0" pos="2.5 2.5 0" size="30 30 1" type="plane" condim="3" material="floor_mat"></geom>
		<body name="floor0" pos="2.5 2.5 0">
			<site name="target0" pos="0.7 0.0 0.4" size="0.10 0.10 0.10" rgba="1 0 0 1" type="sphere"></site>
		</body>

		<geom name="bound0" pos="{bound0_pos}" size="{bound_v_size}" type="box" mass="2000" material="table_mat"></geom>
		<geom name="bound1" pos="{bound1_pos}" size="{bound_v_size}" type="box" mass="2000" material="table_mat"></geom>
		<geom name="bound2" pos="{bound2_pos}" size="{bound_h_size}" type="box" mass="2000" material="table_mat"></geom>
		<geom name="bound3" pos="{bound3_pos}" size="{bound_h_size}" type="box" mass="2000" material="table_mat"></geom>
		{walls}
'''


def generate_xml(num_obstacles):
    wall_template = '<geom name="wall{id}" pos="{pos}" size="0.15 1.0 0.25" type="box" condim="3" mass="100" material="table_mat" rgba="0.93 0.93 0.93 0.5"></geom>'
    walls = []
    for i in range(num_obstacles):
        walls.append(wall_template.format(**dict(id=2 * i, pos=str(1.7 * (i + 1)) + ' 1.0 0.25')))
        walls.append(wall_template.format(**dict(id=2 * i + 1, pos=str(1.7 * (i + 1)) + ' 4.0 0.25')))
    scene = scene_template.format(**dict(bound0_pos=str(1.7 * (num_obstacles + 1) / 2) + ' -0.2 0.25',
                                         bound1_pos=str(1.7 * (num_obstacles + 1) / 2) + ' 5.2 0.25',
                                         bound2_pos='-0.2 2.5 0.25',
                                         bound3_pos=str(1.7 * (num_obstacles + 1) + 0.2) + ' 2.5 0.25',
                                         bound_v_size=str(1.7 * (num_obstacles + 1) / 2) + ' 0.2 0.25',
                                         bound_h_size='0.2 2.9 0.25',
                                         walls="\n".join(walls)))
    obstacle_template = '''<body name="object{id}" pos="0 0 0">
			<joint axis="1 0 0" name="object{id}:slidex" type="slide" damping="10"></joint>
			<joint axis="0 1 0" name="object{id}:slidey" type="slide" damping="10"></joint>
            <joint axis="0 0 1" name="object{id}:slidez" type="slide" damping="10000"></joint>
            <joint axis="0 0 1" name="object{id}:rz" type="hinge" damping="10"></joint>
			<geom size="0.15 0.60 0.15" type="box" name="object{id}" material="stick_mat" mass="5" euler="0 0 0"></geom>
			<site name="object{id}" pos="0 0 0" size="0.02 0.02 0.02" rgba="0.5 0.2 0 0" type="sphere"></site>
		</body>'''
    obstacles = [obstacle_template.format(**dict(id=i + 1)) for i in range(num_obstacles)]
    xml = template.format(**dict(scene=scene, obstacles="\n".join(obstacles)))
    return xml
