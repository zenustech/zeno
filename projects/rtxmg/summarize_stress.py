"""Summarize an actual validate_editor.py viewport-script run (no GPU work)."""
import argparse
import json
import re
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    return values[min(int((len(values)-1)*fraction), len(values)-1)] if values else None


def summarize(output, script_path):
    script = json.loads(script_path.read_text())
    log = (output/'optix-child.log').read_text(errors='replace')
    phases = re.findall(r'\[viewport script\] begin (cycle-\d+-step-\d+) reload=(\d+)(.*?)'
                        r'\[viewport script\] end \1', log, re.S)
    expected = [(f'cycle-{c:03d}-step-{s:02d}', step) for c in range(script.get('cycles', 1))
                for s, step in enumerate(script['steps'])]
    if [row[0] for row in phases] != [row[0] for row in expected]:
        raise RuntimeError('Script phases are missing or out of order')
    results = []
    for (label, reload, part), (_, step) in zip(phases, expected):
        viewport = re.findall(r'OptiX viewport: (\d+)x(\d+),.*?total=([\d.e+-]+) ms', part)
        updates = re.findall(r'CLAS update: ([\d.e+-]+) ms, planning=([\d.e+-]+) ms, loading=([\d.e+-]+) ms, AS=([\d.e+-]+) ms, settled=(\d+), pending=(\d+), plans=(\d+), staticSkips=(\d+), loads=(\d+), evictions=(\d+)', part)
        gpu = re.findall(r'CLAS GPU: LOD=([\d.e+-]+) ms, CLAS=([\d.e+-]+) ms, GAS=([\d.e+-]+) ms, IAS=([\d.e+-]+) ms, scene=([\d.e+-]+) ms', part)
        resource = re.findall(r'\[viewport resources\] label=.*?gpuUsed=(\d+), gpuTotal=(\d+)', part)
        if len(viewport) != step['ticks'] or len(updates) != len(viewport) or len(gpu) != len(viewport):
            raise RuntimeError(f'Wrong frame count in {label}')
        if bool(int(reload)) != bool(step.get('reload', False)):
            raise RuntimeError(f'Wrong reload flag in {label}')
        if step.get('pixels') and any(list(map(int, row[:2])) != step['pixels'] for row in viewport):
            raise RuntimeError(f'Resize not applied in {label}')
        tail = list(zip(updates, gpu))[-min(32, len(updates)):]
        idle = sum(u[4:6] == ('1','0') and all(float(v) == 0 for v in g) and
                   float(u[1]) == float(u[2]) == float(u[3]) == 0 for u, g in tail)
        times = [float(row[2]) for row in viewport]
        results.append({'label':label, 'ticks':len(times), 'reload':bool(int(reload)),
            'pixels':sorted(set(f'{r[0]}x{r[1]}' for r in viewport)),
            'viewport_ms':{'p50':percentile(times,.5),'p95':percentile(times,.95),'p99':percentile(times,.99),'max':max(times)},
            'settled_at_end':updates[-1][4:6] == ('1','0'), 'idle_tail_frames':idle,
            'tail_length':len(tail), 'loads':int(updates[-1][8]), 'evictions':int(updates[-1][9]),
            'gpu_used_end':int(resource[-1][0]) if resource else None,
            'gpu_scene_ms_max':max(float(g[4]) for g in gpu)})
    all_times = [float(v) for v in re.findall(r'OptiX viewport: .*?total=([\d.e+-]+) ms', log)]
    scene_times = [float(v) for v in re.findall(r'CLAS GPU: .*?scene=([\d.e+-]+) ms', log) if float(v)>0]
    host_times = [float(v) for v in re.findall(r'CLAS completed task: active host work=([\d.e+-]+) ms', log) if float(v)>0]
    host = json.loads((output/'host-resources.json').read_text())
    cycle_ends = [r for r in results if r['label'].endswith(f"step-{len(script['steps'])-1:02d}")]
    last = [r['gpu_used_end'] for r in cycle_ends[-5:]]
    report = {'cycles':script.get('cycles',1),'phases':len(results),'ticks':sum(r['ticks'] for r in results),
        'sanitizer': 'COMPUTE-SANITIZER' in log,
        'timings_are_performance_measurements': 'COMPUTE-SANITIZER' not in log,
        'reloads':sum(r['reload'] for r in results), 'maximum_evictions_per_residency':max(r['evictions'] for r in results),
        'viewport_ms':{k:percentile(all_times,f) for k,f in [('p50',.5),('p95',.95),('p99',.99)]},
        'published_gpu_scene_ms':{k:percentile(scene_times,f) for k,f in [('p50',.5),('p95',.95),('p99',.99)]},
        'completed_active_host_ms':{k:percentile(host_times,f) for k,f in [('p50',.5),('p95',.95),('p99',.99)]},
        'cycle_end_gpu_bytes': [r['gpu_used_end'] for r in cycle_ends],
        'settled_cycle_ends':sum(r['settled_at_end'] for r in cycle_ends),
        'fully_idle_cycle_ends':sum(r['idle_tail_frames']==r['tail_length'] for r in cycle_ends),
        'last_five_cycle_end_gpu_range_bytes':max(last)-min(last),
        'host_private_peak_bytes':max(r['private_bytes'] for r in host),
        'host_private_last_quarter_range_bytes':max(r['private_bytes'] for r in host[len(host)*3//4:])-min(r['private_bytes'] for r in host[len(host)*3//4:]),
        'host_private_final_bytes':host[-1]['private_bytes'], 'phase_results':results,
        'scope':'Frame/reload/resize counts, eviction, idle work and resource trends; published cuts are checked separately by ClusterValidate'}
    (output/'stress-summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('phase_results','cycle_end_gpu_bytes')},indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--script',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.output,args.script)
