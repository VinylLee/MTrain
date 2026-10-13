"""A real interrupted supervisor must not launch queued GPU/API jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("interruption",[signal.SIGINT,signal.SIGTERM])
def test_supervisor_interrupt_stops_children_and_cancels_queued_datasets(tmp_path,interruption):
    project=Path(__file__).resolve().parents[1]
    fake=tmp_path/"fake_project";fake.mkdir()
    (fake/"run_nli_v3_4.py").write_text('''import os,sys,time
from pathlib import Path
name=sys.argv[sys.argv.index('--dataset')+1]
Path('started_'+name).write_text(str(os.getpid()))
try: time.sleep(120)
finally: Path('checkpoint_'+name).write_text('preserved')
''')
    manifest=tmp_path/"manifest.json";manifest.write_text('{}')
    output=tmp_path/"experiment"
    harness=tmp_path/"harness.py"
    harness.write_text(f'''import sys
from pathlib import Path
sys.path.insert(0,{str(project)!r})
import run_nli_v3_4_development as driver
driver.ROOT=Path({str(fake)!r})
sys.exit(driver.run(Path({str(manifest)!r}),Path({str(output)!r}),workers=2))
''')
    with (tmp_path/"harness.log").open("w") as log:
        parent=subprocess.Popen([sys.executable,str(harness)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+10
            while len(list(fake.glob('started_*')))<2 and time.monotonic()<deadline:
                assert parent.poll() is None,(tmp_path/"harness.log").read_text()
                time.sleep(.05)
            assert len(list(fake.glob('started_*')))==2
            os.kill(parent.pid,interruption)
            assert parent.wait(timeout=10)==130
            assert {p.name for p in fake.glob('started_*')}=={'started_train','started_snli'}
            assert {p.name for p in fake.glob('checkpoint_*')}=={'checkpoint_train','checkpoint_snli'}
            assert json.loads((output/'status.json').read_text())['state']=='interrupted'
            for p in fake.glob('started_*'):
                assert not Path('/proc/'+p.read_text()).exists()
        finally:
            if parent.poll() is None:
                os.killpg(parent.pid,signal.SIGKILL);parent.wait()
            for p in fake.glob('started_*'):
                try:
                    cmd=Path('/proc/'+p.read_text()+'/cmdline').read_bytes().decode()
                    if str(fake) in cmd:os.kill(int(p.read_text()),signal.SIGKILL)
                except FileNotFoundError:pass
