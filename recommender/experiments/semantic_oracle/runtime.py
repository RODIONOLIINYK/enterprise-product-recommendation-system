"""Connected-runtime manifest handling; this module never reads evaluation targets."""

import argparse
from pathlib import Path

from .protocol import ROOT,read_json,save_json,file_sha,digest
from .agent import worker_prompt

PREDICTION_SOURCE=('protocol.py','history_tools.py','gateway.py','agent.py')


def refresh_protocol(root=ROOT,*,provider='codex',model='gpt-6.2-sol',options=None,model_label='GPT-6.2 Sol'):
    """Update prompts, session runtime metadata, and protocol lock ONLY before any prediction is sealed."""
    root=Path(root);options=dict(options or {'reasoningEffort':'medium'})
    if (root/'reveal.json').exists():raise PermissionError('Cannot refresh protocol after reveal.')
    jobs_payload=read_json(root/'jobs.json');jobs=jobs_payload['jobs']
    if any((Path(j['session_path'])/'seal.json').exists() for j in jobs):
        raise PermissionError('Cannot refresh protocol after predictions have been sealed.')
    manifest=read_json(root/'manifest.json')
    if file_sha(root/'vault'/'truth.json')!=manifest['truth_sha256']:
        raise ValueError('Evaluation vault truth checksum mismatch.')
    project_root=str(Path.cwd())
    for job in jobs:
        spath=Path(job['session_path'])
        session=read_json(spath/'session.json')
        for fname,sha in session['files_sha256'].items():
            if file_sha(spath/fname)!=sha:raise ValueError(f'Blind bundle checksum mismatch in {job["job_id"]}/{fname}.')
        for stale in ('tool_log.jsonl','context_view.txt','last_query_view.txt','answer.json'):
            st=spath/stale
            if st.exists():st.unlink()
        session['requested_provider']=provider
        session['requested_model']=model
        session['requested_options']=options
        save_json(spath/'session.json',session,exclusive=False)
        prompt=worker_prompt(str(spath.resolve()),project_root,model_label=model_label)
        ppath=Path(job['prompt_path'])
        ppath.write_text(prompt,encoding='utf-8');ppath.chmod(0o600)
    jobs_payload['requested_provider']=provider
    jobs_payload['requested_model']=model
    jobs_payload['requested_options']=options
    save_json(root/'jobs.json',jobs_payload,exclusive=False)
    manifest['jobs_sha256']=file_sha(root/'jobs.json')
    manifest['requested_provider']=provider
    manifest['requested_model']=model
    manifest['requested_options']=options
    save_json(root/'manifest.json',manifest,exclusive=False)
    return freeze(root,provider=provider,model=model,options=options,exclusive=False)


def freeze(root=ROOT,*,provider='codex',model='gpt-6.2-sol',options=None,exclusive=True):
    root=Path(root);jobs=read_json(root/'jobs.json')['jobs'];module=Path(__file__).parent
    options=dict(options or {'reasoningEffort':'medium'})
    value={'source_sha256':{str((module/name).resolve()):file_sha(module/name) for name in PREDICTION_SOURCE},
           'prompt_sha256':{job['job_id']:file_sha(job['prompt_path']) for job in jobs},
           'jobs_sha256':file_sha(root/'jobs.json'),'session_count':len(jobs),
           'provider':provider,'model':model,'options':options,
           'temperature':None,'decoding_seed':None,
           'limitations':['Fresh conversations and filtered bundles are not an OS filesystem sandbox.',
                          'Audit worker tool-call transcripts for access outside the gateway before revealing outcomes.']}
    value['protocol_digest']=digest(value);save_json(root/'protocol_lock.json',value,exclusive=exclusive)
    return value


def check_lock(root=ROOT):
    root=Path(root);locked=read_json(root/'protocol_lock.json')
    for path,sha in locked['source_sha256'].items():
        if file_sha(path)!=sha:raise ValueError('Prediction gateway/source changed after protocol lock.')
    if file_sha(root/'jobs.json')!=locked['jobs_sha256']:raise ValueError('Prediction jobs changed after protocol lock.')
    for job in read_json(root/'jobs.json')['jobs']:
        if file_sha(job['prompt_path'])!=locked['prompt_sha256'][job['job_id']]:raise ValueError('Oracle instructions changed after lock.')
    return locked


def pending(root=ROOT):
    root=Path(root);check_lock(root)
    return [job for job in read_json(root/'jobs.json')['jobs'] if not (Path(job['session_path'])/'seal.json').exists()]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(ROOT));p.add_argument('--freeze',action='store_true')
    p.add_argument('--refresh',action='store_true');p.add_argument('--provider',default='codex');p.add_argument('--model',default='gpt-6.2-sol')
    p.add_argument('--effort',default='medium');p.add_argument('--model-label',default='GPT-6.2 Sol')
    args=p.parse_args()
    opts={'reasoningEffort':args.effort} if 'codex' in args.provider else {'thinking_level':args.effort}
    if args.refresh:result=refresh_protocol(args.root,provider=args.provider,model=args.model,options=opts,model_label=args.model_label)
    elif args.freeze:result=freeze(args.root,provider=args.provider,model=args.model,options=opts)
    else:result={'pending_sessions':len(pending(args.root))}
    import json
    print(json.dumps(result))
