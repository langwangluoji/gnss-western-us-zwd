"""Publish recovery assets and verify GitHub's server-computed SHA256.

Git credentials stay in memory and are never written to logs or receipts.
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import subprocess
import time

import requests

from compact import HERE, sha

API='https://api.github.com/repos/langwangluoji/gnss-western-us-zwd'
TAG='archive/pre-data-compaction-20261127'


class UploadReader(io.FileIO):
    def __init__(self,path):
        super().__init__(path,'rb');self.last_report=time.monotonic()

    def read(self,*args):
        value=super().read(*args)
        if time.monotonic()-self.last_report>=30:
            print(json.dumps({'upload_progress':str(self.name).rsplit('\\',1)[-1], 'sent':self.tell()}),flush=True)
            self.last_report=time.monotonic()
        return value


def require(response, codes=(200,201)):
    if response.status_code not in codes:
        raise RuntimeError(f'GitHub HTTP {response.status_code}; no response or credentials logged')
    return response.json()


def incomplete_stub(info, expected_names):
    return info.get('state')=='starter' and info.get('name') in expected_names


def main():
    credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
                              text=True,capture_output=True,check=True)
    fields=dict(x.split('=',1) for x in credential.stdout.splitlines() if '=' in x)
    session=requests.Session()
    session.headers.update({'Authorization':'Bearer '+fields['password'],
                            'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    del fields, credential
    response=session.get(API+'/releases/tags/'+TAG,timeout=30)
    if response.status_code==404:
        release=require(session.post(API+'/releases',json={
            'tag_name':TAG,'name':'Retired experiment model archive — 2026-11-27',
            'body':'Recovery archive for retired experiment weights and lightweight scientific evidence. '
                   'Each model is content addressed by SHA256. See maintenance/compact_20261127 '
                   'for source paths, retention policy and verified deletion ledger. '
                   'Raw observations and retired large derived arrays are not release assets.',
            'draft':False,'prerelease':True},timeout=30))
    else:
        release=require(response)
    index=json.loads((HERE/'archive_index.json').read_text(encoding='utf-8'))
    remote={'release_url':release['html_url'],'tag':TAG,'assets':[],'models':{}}
    existing={a['name']:a for a in require(session.get(API+f"/releases/{release['id']}/assets",params={'per_page':100},timeout=30))}
    # Aborted HTTP bodies leave named 'starter' stubs, not uploaded recoveries.
    # Only this manifest's stubs may be removed, with the complete local archive.
    expected={a['name']:a for a in index['assets']};removed_stubs=[]
    for name,info in list(existing.items()):
        if incomplete_stub(info,expected):
            if sha(HERE/'staging'/name)!=expected[name]['sha256']:
                raise ValueError('local recovery archive required before retry')
            response=session.delete(API+f"/releases/assets/{info['id']}",timeout=30)
            if response.status_code!=204:
                raise RuntimeError(f'Incomplete upload cleanup HTTP {response.status_code}')
            removed_stubs.append({'asset_id':info['id'],'name':name,'state':'starter'})
            del existing[name]
    if removed_stubs:
        (HERE/'retried_upload_stubs.json').write_text(json.dumps(removed_stubs,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'retrying_incomplete_uploads':len(removed_stubs)}),flush=True)
    def transfer(asset):
        name=asset['name']; path=HERE/'staging'/name
        if sha(path)!=asset['sha256']:
            raise ValueError('local archive changed')
        info=existing.get(name)
        if info is None:
            print(json.dumps({'uploading':name,'bytes':asset['size']}),flush=True)
            # A session per worker; credentials remain in memory.
            worker=requests.Session();worker.headers.update(session.headers)
            with UploadReader(path) as f:
                info=require(worker.post(release['upload_url'].split('{')[0],params={'name':name},
                                          data=f,headers={'Content-Length':str(asset['size']),
                                          'Content-Type':'application/json' if name.endswith('.json') else 'application/gzip'},timeout=(30,1800)))
            worker.close()
        if info['size']!=asset['size']:
            raise ValueError('remote asset size mismatch')
        if info.get('digest') and info['digest']!='sha256:'+asset['sha256']:
            raise ValueError('GitHub asset digest mismatch')
        downloaded=False
        if asset['size']<100*2**20 or not info.get('digest'):
            print(json.dumps({'download_verifying':name}),flush=True)
            with requests.get(info['browser_download_url'],params={'verify':int(time.time())},stream=True,timeout=(30,1800)) as res:
                if res.status_code!=200:
                    raise RuntimeError(f'Archive download HTTP {res.status_code}')
                digest=hashlib.sha256(); total=0
                for chunk in res.iter_content(2**20):
                    digest.update(chunk);total+=len(chunk)
            if digest.hexdigest()!=asset['sha256'] or total!=asset['size']:
                raise ValueError('download SHA256 mismatch')
            downloaded=True
        print(json.dumps({'verified':name,'bytes':asset['size'],'downloaded':downloaded}),flush=True)
        return dict(asset,url=info['browser_download_url'],asset_id=info['id'],server_digest=info.get('digest'),
                    verified_remote=True,verified_download=downloaded)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(transfer,a) for a in index['assets']]):
            remote['assets'].append(future.result())
            (HERE/'remote_archive.json').write_text(json.dumps(remote,indent=2)+'\n',encoding='utf-8')
    verified={a['name'] for a in remote['assets'] if a['verified_remote']}
    remote['models']={h:dict(v,verified_remote=True) for h,v in index['models'].items() if v['asset'] in verified}
    (HERE/'remote_archive.json').write_text(json.dumps(remote,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'release':remote['release_url'],'verified_assets':len(verified),'models':len(remote['models'])}),flush=True)


if __name__=='__main__':
    main()
