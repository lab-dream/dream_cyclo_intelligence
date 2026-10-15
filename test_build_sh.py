"""Exercise bootstrap/update command sequences without touching the host Docker daemon."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent
FAKE_DOCKER = r'''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
state=Path(os.environ['BUILD_TEST_STATE']);data=json.loads(state.read_text());a=sys.argv[1:]
data['calls'].append(a)
rc=0
if a[:2]==['container','inspect']:
 rc=0 if a[2] in data['containers'] else 1
elif a[:2]==['image','inspect']:
 rc=0 if data.get('images') else 1
elif a and a[0]=='inspect':
 if '/workspace' in a[2]:print('/another/checkout' if data.get('wrong_mount') else os.environ['BUILD_TEST_ROOT']+'/docker/workspace')
 elif '.State.Health' in a[2]:print('healthy')
 else:print('image-id')
elif a and a[0]=='compose':
 if 'config' in a:print(json.dumps({'services':{s:{'image':s+':test'} for s in ('cyclo_intelligence','lerobot','groot')}}))
 elif 'pull' in a:
  rc=1 if data.get('pull_failure') else 0
  if not rc:data['images']=True
 elif 'build' in a:data['images']=True
 elif 'up' in a:
  data['containers'].append({'lerobot':'lerobot_server','groot':'groot_server'}.get(a[-1],a[-1]))
state.write_text(json.dumps(data));sys.exit(rc)
'''


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'checkout with spaces'
    root.mkdir()
    shutil.copy2(ROOT/'build.sh', root/'build.sh')
    (root/'docker').mkdir()
    (root/'docker/docker-compose.yml').write_text('services: {}\n')
    ui = root/'orchestrator/ui'
    ui.mkdir(parents=True)
    for name in ('package.json', 'package-lock.json'):
        (ui/name).write_text('{}')
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    for name, source in [('docker', FAKE_DOCKER), ('git', '#!/bin/sh\nexit 0\n')]:
        p = bin_dir/name
        p.write_text(source)
        p.chmod(0o755)
    state = tmp_path/'state.json'
    env = dict(os.environ, PATH=str(bin_dir)+':'+os.environ['PATH'],
               BUILD_TEST_STATE=str(state), BUILD_TEST_ROOT=str(root), CYCLO_START_LOCAL_ROUTER='0')
    return root, state, env


@pytest.mark.parametrize('existing', [False, True])
def test_bootstrap_and_existing_update_preserve_data(checkout, existing):
    root, state, env = checkout
    state.write_text(json.dumps(dict(calls=[], containers=['cyclo_intelligence', 'lerobot_server'] if existing else [], images=existing)))
    model = root/'docker/workspace/model/lerobot/user-model.bin'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'user weights')
    run = subprocess.run(['bash', str(root/'build.sh')], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    calls = json.loads(state.read_text())['calls']
    assert any(a[0] == 'exec' and 'colcon build' in ' '.join(a) for a in calls)
    assert any(a[0] == 'exec' and 'npm run build' in ' '.join(a) for a in calls)
    assert ['restart', 'cyclo_intelligence', 'lerobot_server'] in calls
    assert ['exec', 'cyclo_intelligence', '/command/s6-rc', '-u', 'change', 'orchestrator', 'cyclo_data'] in calls
    assert any('pull' in a for a in calls) is (not existing)
    assert any('up' in a for a in calls) is (not existing)
    assert model.read_bytes() == b'user weights'
    assert not any(a[0] in {'rm', 'system', 'volume'} for a in calls)


def test_registry_failure_falls_back_to_source_build(checkout):
    root, state, env = checkout
    state.write_text(json.dumps(dict(calls=[], containers=[], images=False, pull_failure=True)))
    run = subprocess.run(['bash', str(root/'build.sh')], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    assert any(a[0] == 'compose' and 'build' in a for a in json.loads(state.read_text())['calls'])


def test_wrong_checkout_mount_does_not_overwrite_container_or_data(checkout):
    root, state, env = checkout
    state.write_text(json.dumps(dict(calls=[], containers=['cyclo_intelligence'], images=True, wrong_mount=True)))
    run = subprocess.run(['bash', str(root/'build.sh')], env=env, capture_output=True, text=True)
    assert run.returncode != 0 and 'not this checkout' in run.stderr
    assert not any(a[0] in {'rm', 'restart'} for a in json.loads(state.read_text())['calls'])
