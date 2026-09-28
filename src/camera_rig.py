"""Pure camera geometry for fixed overhead, flange RGB-D and observer roles."""
import numpy as np

def look_at_cv(eye,target,up=(0,0,1)):
    eye=np.asarray(eye,dtype=float);target=np.asarray(target,dtype=float);up=np.asarray(up,dtype=float)
    forward=target-eye;norm=np.linalg.norm(forward)
    if norm<1e-9:raise ValueError('Camera eye equals target')
    forward/=norm;right=np.cross(forward,up)
    if np.linalg.norm(right)<1e-6:
        up=np.array([0.,1.,0.]);right=np.cross(forward,up)
    right/=np.linalg.norm(right);down=np.cross(forward,right)
    T=np.eye(4);T[:3,:3]=np.column_stack((right,down,forward));T[:3,3]=eye
    if not np.isfinite(T).all() or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-9) or np.linalg.det(T[:3,:3])<.999:
        raise ValueError('Invalid rigid camera transform')
    return T

def camera_specs(profile='overhead_wrist_v2'):
    if profile=='legacy_v1':
        return [dict(id='overview',role='policy',root='.',resolution_hw=[240,320],fps=10,
            parent='/World',name='OverviewCamera',T_parent_camera_cv=look_at_cv((-3.2,-3.8,3.4),(-.05,0,.75)).tolist(),horizontal_aperture_mm=20.955,focal_mm=20.,seg=False),
            dict(id='source',role='policy',root='source_camera',resolution_hw=[240,320],fps=10,
            parent='/World',name='SourceCamera',T_parent_camera_cv=look_at_cv((1.65,-3.3,2.65),(0,-.78,.76)).tolist(),horizontal_aperture_mm=20.955,focal_mm=20.,seg=True)]
    if profile!='overhead_wrist_v2':raise ValueError('Unknown camera rig')
    # Wrist optical pose selected by dense CAD-ray pregrasp comparison.
    # This does not model camera housing, bracket/cables, or their collisions.
    return [dict(id='overhead',role='policy',root='.',resolution_hw=[480,640],fps=30,
        parent='/World',name='OverheadCamera',T_parent_camera_cv=look_at_cv((-.2,0,4.0),(-.15,0,.5),(0,1,0)).tolist(),horizontal_aperture_mm=32.,focal_mm=24.6,seg=True),
        dict(id='wrist',role='policy',root='wrist_camera',resolution_hw=[480,640],fps=30,
        parent='/World/H2017/link_6',name='WristRGBD',T_parent_camera_cv=look_at_cv((.22,0,.07),(0,0,.29369938),(0,1,0)).tolist(),horizontal_aperture_mm=32.,focal_mm=17.5,seg=False),
        dict(id='observer',role='observer_only',root='observer_camera',resolution_hw=[1080,1920],fps=30,
        parent='/World',name='ObserverCamera',T_parent_camera_cv=look_at_cv((3.2,-3.8,3.3),(-.25,0,1.)).tolist(),horizontal_aperture_mm=32.,focal_mm=28.,seg=False)]

def intrinsics(spec):
    height,width=spec['resolution_hw'];ha=spec['horizontal_aperture_mm'];va=ha*height/width;f=spec['focal_mm']
    return np.array([[width*f/ha,0,width/2],[0,height*f/va,height/2],[0,0,1.]])

def world_camera_pose(spec,world_flange=None):
    local=np.asarray(spec['T_parent_camera_cv'],dtype=float)
    if spec['id']=='wrist':
        if world_flange is None:raise ValueError('Wrist camera needs measured flange pose')
        return np.asarray(world_flange)@local
    return local
