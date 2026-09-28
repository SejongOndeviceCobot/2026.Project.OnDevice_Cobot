"""Named, read-only robot collision profiles. Configuration selection; physical validation is recorded per run."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import yaml

PROFILE_NAMES=('baseline','link2_hull12_margin10mm','link24_hull_margin10mm')
LINK24_REL='robot-collision-candidates-v2/h2017_link24_hull_margin10mm'
LINK24_PINS={'candidate': '145003cb16ae494c6587067a0d31e237f2e01dae191e9deb1e3ecb717b7e7be6', 'manifest': '8f71f18e71e6b16f1a33da884ec329b9940498a29e771e88b5cf3f43752d24d0', 'validation': '79fb42c6f947cfb32223671ac8a4afcfa91a656c5d576698125520743f0fea4b'}
BASELINE_REL='grasp-requests-v2/robot-slow.yml'
URDF_REL='grasp-requests-v2/h2017_vgp20_velocity_limited.urdf'
CANDIDATE_REL='robot-collision-candidates-v1/h2017_link2_hull12_margin10mm'
PINNED_SHA256={
 'baseline':'c11db2bbbf2868a2f290aa671ce26b4022594e254d673a275848226806785e05',
 'candidate':'06b51958bc7e400ebb7ab3f3fc364b0862ca7d3e3e75b263a5fa4d4e94794fa6',
 'manifest':'3c10031ac5915455275e7713f938c195445a8792e57c7e1a9d95714b77fb3034',
 'validation':'f73d16b929142b1b9622d619709aac624a69efaf1767ee6bbd252c276e6f68bf',
 'urdf':'1510e86713f806b0a493c9c45db8f7730571b113a77cef6845461b617b1d445f',
}


def _sha(raw):return hashlib.sha256(raw).hexdigest()

def _regular(path: Path, data_root: Path):
    root=data_root.resolve(strict=True)
    # The project data root itself may be the documented /DATA symlink.
    # No further symlink below that trusted root may redirect an artifact.
    relative=path.relative_to(data_root)
    current=root
    for part in relative.parts:
        current=current/part
        if current.is_symlink():raise ValueError('Profile artifact symlink is forbidden: '+str(current))
    resolved=current.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError('Profile artifact is outside the project data root or not a file')
    return current


def _read_pinned(path, data_root, key):
    raw=_regular(path,data_root).read_bytes()
    if _sha(raw)!=PINNED_SHA256[key]:raise ValueError('Profile '+key+' SHA256 mismatch')
    return raw


def _path_equals(recorded, expected):
    if not isinstance(recorded,str) or not Path(recorded).is_absolute():return False
    try:return Path(recorded).resolve(strict=True)==expected.resolve(strict=True)
    except (OSError,RuntimeError):return False


def _semantic_checks(baseline,candidate,manifest,validation,paths):
    """Defense in depth after byte pinning; separately exercised by unit tests."""
    bk=baseline.get('robot_cfg',baseline)['kinematics'];ck=candidate.get('robot_cfg',candidate)['kinematics']
    original=bk['collision_spheres']['link_2'];actual=ck['collision_spheres']['link_2']
    if len(original)!=10 or len(actual)!=22 or actual[:10]!=original:
        raise ValueError('Original link_2 spheres must remain exactly unchanged')
    reverted=copy.deepcopy(candidate)
    del reverted.get('robot_cfg',reverted)['kinematics']['collision_spheres']['link_2'][10:]
    if reverted!=baseline:raise ValueError('Only 12 appended link_2 spheres may differ; other robot settings changed')
    extra=actual[10:]
    for s in extra:
        if set(s)!= {'center','radius'} or len(s['center'])!=3:
            raise ValueError('Unexpected added sphere schema')
        values=[*s['center'],s['radius']]
        if any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in values) or s['radius']<=.010:
            raise ValueError('Invalid finite added sphere geometry')
    count=lambda k:sum(s['radius']>0 for name in k['collision_link_names'] if name!='attached_object' for s in k['collision_spheres'][name])
    if (count(bk),count(ck))!=(532,544):raise ValueError('Unexpected collision sphere count')
    if manifest.get('schema')!='depallet.robot_collision_candidate.v1':raise ValueError('Unexpected candidate manifest schema')
    for key,name in [('robot_config','candidate'),('baseline_config','baseline'),('urdf','urdf'),('validation','validation')]:
        if not _path_equals(manifest.get(key),paths[name]):raise ValueError('Manifest path binding mismatch: '+key)
        if manifest.get(key+'_sha256')!=PINNED_SHA256[name]:raise ValueError('Manifest SHA binding mismatch: '+key)
    if not _path_equals(ck['urdf_path'],paths['urdf']):raise ValueError('Candidate URDF must be the fixed baseline URDF')
    required={'original_positive_robot_spheres':532,'candidate_positive_robot_spheres':544,
              'original_link2_spheres':10,'candidate_link2_spheres':22,
              'geometric_numeric_enclosure_margin_m':.000005,'explicit_design_margin_m':.010}
    if any(type(manifest.get(k)) is bool or manifest.get(k)!=v for k,v in required.items()):
        raise ValueError('Candidate count or explicit design margin mismatch')
    if manifest.get('added_spheres')!=extra:raise ValueError('Added sphere manifest mismatch')
    if manifest.get('all_other_config_values_identical') is not True or manifest.get('design_margin_is_native_contact_offset_estimate') is not False:
        raise ValueError('Manifest scope mismatch')
    if manifest.get('physical_validated') is not False or manifest.get('continuous_path_validated') is not False:
        raise ValueError('CPU candidate cannot claim physical or continuous validation')
    if validation.get('schema')!='depallet.link2_hull_margin_candidate.validation.v1' or not _path_equals(validation.get('candidate_yaml'),paths['candidate']) or validation.get('candidate_sha256')!=PINNED_SHA256['candidate']:
        raise ValueError('Validation candidate binding mismatch')
    if validation.get('promoted') is not False or validation.get('physical_execution_validated') is not False:
        raise ValueError('Validation scope must remain CPU candidate only')
    enclosure=validation.get('slab_enclosure',{});slabs=enclosure.get('slabs',[])
    if enclosure.get('all_slabs_passed') is not True or len(slabs)!=12:
        raise ValueError('Missing full 12-slab enclosure evidence')
    for i,slab in enumerate(slabs):
        excess=slab.get('maximum_vertex_excess_m')
        if slab.get('index')!=i or slab.get('passed') is not True or not isinstance(excess,(int,float)) or not math.isfinite(excess) or excess>-.010+1e-10:
            raise ValueError('Slab does not preserve the explicit 10 mm margin')
        if i and abs(slabs[i-1]['slab_x_bounds_m'][1]-slab['slab_x_bounds_m'][0])>1e-12:
            raise ValueError('Hull slab intervals are not contiguous')
    full=validation.get('full03',{});curves=full.get('curves',[])
    if full.get('all_47_curves_checked') is not True or full.get('all_added_sphere_checks_passed') is not True or len(curves)!=47 or full.get('waypoints_checked')!=33747:
        raise ValueError('Missing saved-path validation evidence')
    if sum(c.get('waypoints_checked',0) for c in curves)!=33747 or any(c.get('passed') is not True or c.get('colliding_waypoint_counts_per_pair') for c in curves):
        raise ValueError('Saved-path validation contains a failed or incomplete curve')
    for curve in curves:
        for entry in curve['minimum_clearances'].values():
            clearance=entry.get('clearance_m')
            if clearance is not None and (not math.isfinite(clearance) or clearance<0):raise ValueError('Negative saved-path clearance')
    return True


def resolve_robot_collision_profile(profile, template_robot_config, *, workspace):
    """Return (path string, receipt), without writes, imports of Isaac, or GPU.

    The environment's JCLEE_WORKSPACE is the trusted workspace root. The caller
    supplies a named profile only; candidate paths and bytes are fixed here.
    Baseline returns the exact template string, preserving request semantics.
    """
    if profile=='link24_hull_margin10mm':
        return _resolve_link24(template_robot_config,workspace=workspace)
    if profile not in PROFILE_NAMES:raise ValueError('Unknown named robot collision profile')
    trusted=Path(os.environ.get('JCLEE_WORKSPACE','/home/jclee/workspace'))
    workspace=Path(workspace)
    if workspace.resolve(strict=True)!=trusted.resolve(strict=True):raise ValueError('Workspace must match the trusted workspace root')
    workspace=trusted  # Preserve the configured canonical spelling, never a caller's alias.
    data=workspace/'data/depallet_isaac_p0'
    folder=data/CANDIDATE_REL
    paths={'baseline':data/BASELINE_REL,'urdf':data/URDF_REL,'candidate':folder/'robot.yml','manifest':folder/'manifest.json','validation':folder/'validation.json'}
    allowed_template_paths={str(paths['baseline']),str(paths['baseline'].resolve(strict=True))}
    if not isinstance(template_robot_config,str) or template_robot_config not in allowed_template_paths or not _path_equals(template_robot_config,paths['baseline']):
        raise ValueError('Template config must be the fixed baseline config within this project')
    baseline_raw=_read_pinned(paths['baseline'],data,'baseline')
    receipt={'schema':'depallet.robot_collision_profile.v1','profile':profile,
        'template_robot_config':template_robot_config,'baseline_sha256':_sha(baseline_raw),
        'selected_robot_config':template_robot_config,'selected_robot_config_sha256':_sha(baseline_raw),
        'candidate_bundle_checked':False,'physical_validation_claimed':False}
    if profile=='baseline':
        receipt['baseline_path_string_preserved']=True
        return template_robot_config,receipt
    raw={key:_read_pinned(paths[key],data,key) for key in ('candidate','manifest','validation','urdf')}
    candidate=yaml.safe_load(raw['candidate']);baseline=yaml.safe_load(baseline_raw)
    manifest=json.loads(raw['manifest']);validation=json.loads(raw['validation'])
    _semantic_checks(baseline,candidate,manifest,validation,paths)
    selected=str(paths['candidate'])
    receipt.update(selected_robot_config=selected,selected_robot_config_sha256=PINNED_SHA256['candidate'],
        candidate_bundle_checked=True,candidate_manifest=str(paths['manifest']),candidate_manifest_sha256=PINNED_SHA256['manifest'],
        validation=str(paths['validation']),validation_sha256=PINNED_SHA256['validation'],
        urdf=str(paths['urdf']),urdf_sha256=PINNED_SHA256['urdf'],original_robot_spheres=532,selected_robot_spheres=544,
        added_link2_spheres=12,existing_spheres_and_other_config_preserved=True,explicit_design_margin_m=.010,
        design_margin_is_native_contact_offset_estimate=False,physical_execution_validated=False,continuous_path_validated=False)
    return selected,receipt


def _resolve_link24(template_robot_config, *, workspace):
    parent,receipt=resolve_robot_collision_profile('link2_hull12_margin10mm',template_robot_config,workspace=workspace)
    data=Path(os.environ.get('JCLEE_WORKSPACE','/home/jclee/workspace'))/'data/depallet_isaac_p0'
    folder=data/LINK24_REL
    raw={}
    for key,file in [('candidate','robot.yml'),('manifest','manifest.json'),('validation','validation.json')]:
        raw[key]=_regular(folder/file,data).read_bytes()
        if _sha(raw[key])!=LINK24_PINS[key]:raise ValueError('Link24 profile SHA mismatch: '+key)
    baseline=yaml.safe_load(Path(parent).read_text());candidate=yaml.safe_load(raw['candidate'])
    manifest=json.loads(raw['manifest']);validation=json.loads(raw['validation'])
    bk=baseline.get('robot_cfg',baseline)['kinematics'];ck=candidate.get('robot_cfg',candidate)['kinematics']
    before=bk['collision_spheres']['link_4'];after=ck['collision_spheres']['link_4']
    if after[:len(before)]!=before or len(after)!=len(before)+32 or after[len(before):]!=manifest['added_link4_spheres']:
        raise ValueError('Link24 must append exactly the pinned 32 spheres')
    reverted=copy.deepcopy(candidate);reverted.get('robot_cfg',reverted)['kinematics']['collision_spheres']['link_4']=before
    if reverted!=baseline:raise ValueError('Link24 changed settings other than appended link4 spheres')
    if validation.get('failure_pose_rejected') is not True or validation.get('physical_validated') is not False:
        raise ValueError('Link24 validation scope mismatch')
    receipt.update(profile='link24_hull_margin10mm',selected_robot_config=str(folder/'robot.yml'),
        selected_robot_config_sha256=LINK24_PINS['candidate'],selected_robot_spheres=576,
        added_link4_spheres=32,candidate_manifest=str(folder/'manifest.json'),
        candidate_manifest_sha256=LINK24_PINS['manifest'],validation=str(folder/'validation.json'),
        validation_sha256=LINK24_PINS['validation'])
    return str(folder/'robot.yml'),receipt
